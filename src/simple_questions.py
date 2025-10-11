import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.distributed as dist
import numpy as np
import os
import json
import yaml
from pathlib import Path
import argparse
from typing import Optional
import datetime
from datetime import timedelta
from torch.cuda.amp import autocast, GradScaler
import gc

import os
# NOTE: Avoid installing packages at runtime on clusters. Please
# prepare your environment (conda/venv or modules) before submitting.
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

from nn.llm import MultimodalLlamaModel
# from nn.models import MultiTaskRegressor
from llama3.llama.model import Transformer, ModelArgs
from data.dataset_interpert import StellarQuestionsDataset, create_stellar_dataloaders, collate_fn
from data.transforms import *
from nn.train import LLMTrainer
from util.utils import *
# from nn.multimodal import setup

JSON_PATH = '/data/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json'
COMPARATIVE_JSON_FILE='/data/TalkingLatents/data/dataset/comparative_dataset.json'
FEATURES_PATH = '/data/TalkingLatents/logs/2025-07-29/features.npy'  # Optional, can be None to load all features on-the-fly
MODEL_PATH = "/data/.llama/Llama3.1-8B"
TOKENIZER_PATH = "/data/.llama/Llama3.1-8B"
SPECTRA_CONFIG_PATH = "/data/DESA/logs/spec_decode2_2025-02-16/MultiTaskRegressor_spectra__decode_4_complete_config.yaml"
SPECTRA_WEIGHTS_PATH = "/data/DESA/logs/spec_decode2_2025-02-16/MultiTaskRegressor_spectra_decode_4.pth"

print("number of gpus: ", torch.cuda.device_count())
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Performance knobs: enable cuDNN autotuner and allow faster matmul where supported
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass


def setup():
    """
    Setup the distributed training environment (SLURM-friendly, V100s).
    - Robustly derives rank/world_size/local_rank from SLURM or torchrun env.
    - Initializes NCCL backend and sets CUDA device.
    - Attempts fairscale MP init if available, but does not require it.
    """
    import torch.distributed as dist

    # Derive distributed env from SLURM or torchrun
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    jobid = int(os.environ.get("SLURM_JOBID", 0))

    # Ensure master addr/port are present (even for single process)
    os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "127.0.0.1"))
    # Choose a port deterministically from job id when available
    default_port = 12910 + (jobid % 20000) if jobid else 12910
    os.environ.setdefault("MASTER_PORT", str(default_port))

    gpus_per_node = torch.cuda.device_count()
    print('jobid ', jobid)
    print('gpus per node ', gpus_per_node)
    print(
        f"Hello from rank {rank} of {world_size} where there are {gpus_per_node} allocated GPUs per node.",
        flush=True,
    )

    # Initialize process group (works for world_size==1 as well)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method="env://")

    if rank == 0:
        print(f"Group initialized? {dist.is_initialized()}", flush=True)

    # Map this process to its local GPU
    torch.cuda.set_device(local_rank)
    print(f"rank: {rank}, local_rank: {local_rank}")

    # Initialize fairscale model parallel if available (required for LLaMA even on 1 GPU)
    try:
        import fairscale.nn.model_parallel.initialize as fs_init
        if not fs_init.model_parallel_is_initialized():
            fs_init.initialize_model_parallel(1)  # 1 = no model parallel split
            if rank == 0:
                print("Fairscale model parallel initialized", flush=True)
    except Exception as _:
        if rank == 0:
            print("Fairscale not available; continuing without model parallel.")

    return local_rank, world_size, gpus_per_node

def _load_llm_model_with_error_handling(args) -> Transformer:
    """Load LLaMA model with better error handling for checkpoint loading"""
    model_path, tokenizer_path = get_model_path(args)
    max_batch_size, max_seq_len = args.batch_size, args.max_seq_length
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    with open(Path(model_path) / "params.json", "r") as f:
        params = json.loads(f.read())

    print(f"Model params from config: {params}")

    model_args = ModelArgs(
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        **params,
    )

    # Create model; Llama allocates some caches on GPU internally.
    # Keeping params on CPU initially reduces peak GPU usage during load.
    print("Creating LLaMA model on CPU...")
    model = Transformer(model_args)
    
    # Print model architecture info
    print(f"Model created with:")
    print(f"  Vocab size: {model_args.vocab_size}")
    print(f"  Dim: {model_args.dim}")
    print(f"  N layers: {model_args.n_layers}")
    print(f"  N heads: {model_args.n_heads}")
    print(f"  Multiple of: {model_args.multiple_of}")

    # Load checkpoints with better error handling
    checkpoints = sorted(Path(model_path).glob("*.pth"))
    if checkpoints:
        if dist.is_initialized() and rank != 0:
            print("Skipping checkpoint load on non-zero rank; weights will sync from rank 0 via DDP")
        else:
            print(f"Loading LLaMA checkpoint on rank {rank}: {checkpoints[0]}")
            try:
                # Load weights on CPU to avoid GPU OOM; immediately free after load.
                checkpoint = torch.load(checkpoints[0], map_location="cpu")
                
                # Print checkpoint info
                if isinstance(checkpoint, dict):
                    checkpoint_keys = list(checkpoint.keys())[:10]  # First 10 keys
                    print(f"Checkpoint keys (first 10): {checkpoint_keys}")
                    
                    # Check if this is a nested checkpoint
                    if 'model' in checkpoint:
                        checkpoint = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        checkpoint = checkpoint['state_dict']
                
                # Try to load with strict=False first
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
                
                if missing_keys:
                    print(f"Missing keys: {len(missing_keys)} (showing first 5)")
                    for key in missing_keys[:5]:
                        print(f"  Missing: {key}")
                
                if unexpected_keys:
                    print(f"Unexpected keys: {len(unexpected_keys)} (showing first 5)")
                    for key in unexpected_keys[:5]:
                        print(f"  Unexpected: {key}")
                
                print("✓ LLaMA model loaded successfully with partial weights on rank 0")
                # Free the checkpoint tensors to lower host RAM peak
                del checkpoint
                gc.collect()
                
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Proceeding with randomly initialized model...")
    else:
        print("No checkpoints found, using randomly initialized model")
    
    return model

def get_model_path(args):
    """Resolve LLaMA model directory robustly with multiple fallbacks.

    Order:
      1) --llm_path if provided and valid
      2) --llm_root/--llm_model
      3) --llm_root (directly)
      4) --llm_root/--llm_model/original
      5) --llm_root/original
    A valid directory contains params.json and tokenizer.model.
    """
    def is_valid_model_dir(p: str) -> bool:
        return os.path.isfile(os.path.join(p, 'params.json')) and os.path.isfile(os.path.join(p, 'tokenizer.model'))
    print('llm root: ', args.llm_root)
    candidates = []
    if hasattr(args, 'llm_path') and args.llm_path:
        candidates.append(args.llm_path)
    # typical layout root/name
    if hasattr(args, 'llm_root') and hasattr(args, 'llm_model'):
        candidates.append(os.path.join(args.llm_root, args.llm_model))
    # sometimes llm_root already points to the actual model dir
    if hasattr(args, 'llm_root'):
        candidates.append(args.llm_root)
    # huggingface original subfolder variants
    if hasattr(args, 'llm_root') and hasattr(args, 'llm_model'):
        candidates.append(os.path.join(args.llm_root, args.llm_model, 'original'))
    if hasattr(args, 'llm_root'):
        candidates.append(os.path.join(args.llm_root, 'original'))

    chosen = None
    for c in candidates:
        if c and is_valid_model_dir(c):
            chosen = c
            break

    if not chosen:
        msg = (
            "Could not resolve LLaMA model directory. Checked: "
            + ", ".join([str(c) for c in candidates if c])
            + ". Ensure the chosen directory contains params.json and tokenizer.model."
        )
        raise FileNotFoundError(msg)

    tokenizer_path = os.path.join(chosen, 'tokenizer.model')
    print(f"Using LLaMA model directory: {chosen}")
    return chosen, tokenizer_path



def print_detailed_memory():
    """Fixed memory reporting function"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        
        # FIX: Get total memory correctly
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        available = total_memory - allocated  # Available = Total - Currently Allocated
        
        print(f"GPU Memory Details:")
        print(f"  Total GPU Memory: {total_memory:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB") 
        print(f"  Max Allocated: {max_allocated:.2f} GB")
        print(f"  Available: {available:.2f} GB")
        print(f"  Free (Reserved - Allocated): {(reserved - allocated):.2f} GB")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train CLIP Multimodal Stellar Model')
    
    # Data paths
    parser.add_argument('--json_file', type=str, default=JSON_PATH,
                       help='Path to stellar descriptions JSON file')
    parser.add_argument('--features_file', type=str, default=FEATURES_PATH,
                       help='Path to spectral features numpy file')
    parser.add_argument('--output_dir', type=str, default='logs',
                       help='Output directory for logs and models')
    parser.add_argument('--exp_name', type=str, default='interpert',
                       help='Experiment name')
    
    # Model configuration
    parser.add_argument('--llm_root', type=str, default=os.environ.get('LLM_ROOT', '/data/.llama'),
                       help='Root directory containing LLaMA models (or set env LLM_ROOT)')
    parser.add_argument('--llm_model', type=str, default='Llama3.1-8B',
                       help='LLaMA model name under --llm_root')
    parser.add_argument('--llm_path', type=str, default=None,
                       help='Full path to the LLaMA model directory (contains params.json). Overrides llm_root/llm_model.')
    parser.add_argument('--llm_precision', type=str, default='fp16', choices=['fp32','fp16','bf16'],
                       help='Precision to hold LLM weights on GPU (fp16 recommended on V100)')
    parser.add_argument('--spectral_embedding_dim', type=int, default=2048,
                       help='Spectral model embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Common projection space dimension')
    parser.add_argument('--num_spectral_features', type=int, default=4,
                       help='Number of spectral features to integrate into LLM')
    parser.add_argument('--latent_ids', type=list, nargs='*', default=['Teff', 'logg', 'FeH'],
                       help='List of latent variable IDs to include (e.g., --latent_ids mass age metallicity)')
    parser.add_argument('--max_seq_length', type=int, default=128,
                       help='Maximum sequence length for text inputs'),
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                          help='Directory to load model checkpoint from, if any'),
    parser.add_argument('--train', type=bool, default=True,
                          help='Whether to train the model or just evaluate'),
    parser.add_argument('--use_cfm', action='store_true', default=False, 
                   help='Use Conditional Flow Matching')
    parser.add_argument('--cfm_weight', type=float, default=0.1,
                   help='Weight for CFM loss')
                    
                
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=1000,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                       help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                       help='Number of warmup epochs')
    parser.add_argument('--early_stopping', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--max_iter', type=int, default=-1,
                       help='Maximum training iterations per epoch (-1 for no cap)')

    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use Automatic Mixed Precision training')
    parser.add_argument('--amp_opt_level', type=str, default='O1',
                       choices=['O0', 'O1', 'O2', 'O3'],
                       help='AMP optimization level')
    parser.add_argument('--loss_scale', type=float, default=None,
                       help='Static loss scaling factor (None for dynamic)')

    parser.add_argument('--mode', type=str, choices=['single_star', 'two_star', 'combined'], 
                       default='combined', help='Training mode: single_star, two_star, or combined')
    
    parser.add_argument('--switch_epoch', type=int, default=7,
                       help='Epoch to switch from single_star to two_star in combined mode')
    
    parser.add_argument('--comparative_json_file', type=str, 
                       default='/data/TalkingLatents/data/dataset/comparative_dataset.json',
                       help='Path to comparative questions JSON file (used in two_star mode)')
    
    parser.add_argument('--single_sample_prob', type=float, default=1,
                       help='Probability of drawing a single-star sample when using the mixed dataset')
    
    parser.add_argument('--curriculum_decay_steps', type=int, default=1000,
                       help='Number of iterations between single_sample_prob decreases (0 = no curriculum)')
    
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.159, 0.5, 0.841],
                       help='Quantiles for CQR stellar parameter prediction (default: ~1-sigma + median)')
    
    # Memory optimization
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True,
                       help='Use gradient checkpointing to save memory')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')
    
    # Data splitting
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Test set ratio')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for data splitting')
    
    # Distributed training
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of dataloader workers')
    
    # Model freezing
    parser.add_argument('--freeze_llm', action='store_true', default=True,
                       help='Freeze LLM parameters')
    parser.add_argument('--freeze_spectral', action='store_true', default=True,
                       help='Freeze spectral model parameters')
    
    # Evaluation
    parser.add_argument('--eval_every', type=int, default=1,
                       help='Evaluate every N epochs')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--compute_retrieval_metrics', action='store_true',
                       help='Compute retrieval metrics during training')

    # Resume options
    parser.add_argument('--resume_path', type=str, default=None,
                        help='Path to a full training checkpoint (model+optimizer+scheduler+scaler) to resume')
    
    return parser.parse_args()

def create_datasets_and_loaders(args, device):
    """Create datasets and dataloaders with distributed sampling"""
    
    # Load spectral features if provided
    spectral_features = None
    if args.features_file and os.path.exists(args.features_file):
        print(f"Loading spectral features from {args.features_file}")
        spectral_features = np.load(args.features_file)
        print(f"Spectral features shape: {spectral_features.shape}")
    else:
        print("No spectral features file provided or file not found. Will use raw spectra on-the-fly.")
    model_path, tokenizer_path = get_model_path(args)
    transf = Compose([GeneralSpectrumPreprocessor(rv_norm=True), ToTensor()])

    # Create cache directory for split consistency
    # In DDP, avoid concurrent writes to the same npz file (race -> BadZipFile).
    # Use rank-specific cache dirs for ranks > 0 to prevent collisions.
    cache_dir_base = os.path.join(args.output_dir, 'cache')
    rank = dist.get_rank() if dist.is_initialized() else 0
    cache_dir = cache_dir_base if rank == 0 else f"{cache_dir_base}_r{rank}"
    os.makedirs(cache_dir, exist_ok=True)

    train_loader, val_loader, test_loader = create_stellar_dataloaders(
                                            json_file=args.json_file,
                                            features_array=spectral_features,
                                            spectral_transforms=transf,
                                            train_ratio=args.train_ratio,
                                            val_ratio=args.val_ratio,
                                            test_ratio=args.test_ratio,
                                            random_state=args.random_seed,
                                            num_spectral_features=args.num_spectral_features,
                                            cache_dir=cache_dir,
                                            tokenizer_path=tokenizer_path,  
                                            max_length=args.max_seq_length,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,)

    # Optional: if rank > 0 and you want to reuse rank 0 cache next time,
    # you can copy it or just rely on deterministic splitting.


    return train_loader, val_loader, test_loader

def create_model_memory_optimized(args, device):
    """Create model with aggressive memory optimization"""
    
    # Set memory allocation config
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
    
    # Clear all memory at start
    # torch.cuda.empty_cache()
    # gc.collect()
    
    # print("=== Initial Memory State ===")
    # print_detailed_memory()
    
    # Get rank info
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    print(f"Loading models on rank {rank}/{world_size}")

    if args.features_file is None:
    
        spectral_model = _load_spectra_model()

        
        # Keep spectral model on CPU if frozen to save GPU memory
        
        spectral_model = spectral_model.to(device)
    else:
        spectral_model = None
    
    print("=== After Spectral Model ===")
    print_detailed_memory()
    
    llm_model = _load_llm_model(args)

    # Downcast LLM weights to save memory on GPU
    if args.llm_precision == 'fp16':
        llm_model.half()
        print("✓ LLM weights cast to float16")
    elif args.llm_precision == 'bf16':
        # bfloat16 not supported on V100; guard to avoid silent slowdowns
        if torch.cuda.is_bf16_supported():
            llm_model.to(dtype=torch.bfloat16)
            print("✓ LLM weights cast to bfloat16")
        else:
            print("! bfloat16 not supported on this GPU; keeping default precision")
    
    if args.gradient_checkpointing:
        if hasattr(llm_model, 'gradient_checkpointing_enable'):
            llm_model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled for LLM")
    
        if hasattr(spectral_model, 'gradient_checkpointing_enable') and args.features_file is None:
            spectral_model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled for spectral model")
    # Keep LLM on CPU if frozen to save GPU memory
    
    llm_model = llm_model.to(device)
    
    print("=== After LLM Model ===")
    print_detailed_memory()
    
    # Create multimodal model with CPU/GPU hybrid approach
    print("Creating multimodal wrapper...")
    model = MultimodalLlamaModel(
        base_model=llm_model,
        fm_model=spectral_model,
        latent_dim=args.spectral_embedding_dim,
        hidden_dim=args.hidden_dim,
        num_spectral_features=args.num_spectral_features
    )

    model = model.to(device)
    
    print("=== After Multimodal Wrapper ===")
    print_detailed_memory()

    print(args.checkpoint_dir)
    
    # Load plain model checkpoint for warm-start if provided and not using exact resume
    if getattr(args, 'resume_path', None):
        print("Exact resume requested; skipping warm-start checkpoint load here.")
    elif args.checkpoint_dir and os.path.isdir(args.checkpoint_dir):
        candidates = [
            os.path.join(args.checkpoint_dir, f"{args.exp_name}.pth"),
            os.path.join(args.checkpoint_dir, 'interpert.pth'),
            os.path.join(args.checkpoint_dir, 'clip_stellar.pth'),
        ]
        # Fallback to any .pth in dir
        glob_any = sorted(Path(args.checkpoint_dir).glob('*.pth'))
        if glob_any:
            candidates.append(str(glob_any[0]))

        chosen = next((p for p in candidates if os.path.isfile(p)), None)
        if chosen:
            print(f"Warm-start: loading checkpoint weights from {chosen}")
            try:
                model = load_checkpoints_ddp(model, chosen)
            except Exception as e:
                print(f"Warning: failed to load checkpoint '{chosen}': {e}")
        else:
            print(f"No checkpoint file found in {args.checkpoint_dir}; continuing without warm-start")
        
    # Move trainable projection layers to GPU
    print("Moving projection layers to GPU...")
    if hasattr(model, 'text_projection') and model.text_projection is not None:
        model.text_projection = model.text_projection.to(device)
        print("✓ Text projection on GPU")
    
    if hasattr(model, 'spectral_projection') and model.spectral_projection is not None:
        model.spectral_projection = model.spectral_projection.to(device)
        print("✓ Spectral projection on GPU")
    
    print("=== After Projections ===")
    print_detailed_memory()
    
    # Freeze large submodules BEFORE wrapping with DDP to avoid gradient buckets
    if args.freeze_llm and hasattr(model, 'base_model') and model.base_model is not None:
        for p in model.base_model.parameters():
            p.requires_grad = False
        print("✓ LLM base_model frozen (no grads/buckets)")

    if args.freeze_spectral and hasattr(model, 'fm_model') and model.fm_model is not None:
        for p in model.fm_model.parameters():
            p.requires_grad = False
        print("✓ Spectral fm_model frozen (no grads/buckets)")

    # Apply DDP only if multi-GPU
    if world_size > 1:
        print(f"Applying DDP for distributed training (world_size={world_size})")
        # Ensure rank 0 finished loading before broadcasting parameters
        try:
            dist.barrier()
        except Exception:
            pass
        
        # Create a custom DDP that handles CPU/GPU hybrid models
        model = DDP(
            model,
            # Restore earlier behavior: only pin device_ids when unfrozen parts exist
            device_ids=[device] if not args.freeze_llm or not args.freeze_spectral else None,
            find_unused_parameters=True
        )
        print("✓ DDP applied")
    else:
        print("Single GPU training - no DDP needed")
    
    print("=== Final Memory State ===")
    print_detailed_memory()
    
    return model

def _load_spectra_model_cpu():
    """Load spectral model directly to CPU to save memory"""
    config = yaml.safe_load(open(SPECTRA_CONFIG_PATH, 'r'))
    config['model_args']['avg_output'] = False
    
    model = MultiTaskRegressor(Container(**config['model_args']), Container(**config['conformer_args']))
    
    checkpoint = torch.load(SPECTRA_WEIGHTS_PATH, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)  # Handle nested structure
    
    # Remove 'module.' prefix from DDP checkpoints
    state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model

def _load_llm_model_cpu(args) -> Transformer:
    """Load LLaMA model to CPU first"""
    model_path, tokenizer_path = get_model_path(args)
    max_batch_size, max_seq_len = args.batch_size, args.max_seq_length
    
    with open(Path(model_path) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args = ModelArgs(
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        **params,
    )

    # Create model on CPU
    print("Creating LLaMA model on CPU...")
    model = Transformer(model_args)

    # Load checkpoints to CPU first
    checkpoints = sorted(Path(model_path).glob("*.pth"))
    if checkpoints:
        checkpoint = torch.load(checkpoints[0], map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
    
    return model

# Update the existing functions to use CPU loading
def _load_spectra_model():
    return _load_spectra_model_cpu()

def _load_llm_model(args) -> Transformer:
    return _load_llm_model_with_error_handling(args)

def create_model(args, device):
    """Create model with proper DDP handling"""
    
    # Set memory allocation config
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Clear all GPU memory at start
    torch.cuda.empty_cache()
    gc.collect()
    
    print_detailed_memory()
    
    print("Loading LLM model...")
    llm_model = _load_llm_model(args)
    
    print_detailed_memory()
    
    print("Loading spectral model...")
    spectral_model = _load_spectra_model()

    print_detailed_memory()
    
    # # Move frozen models to CPU to save memory
    # if args.freeze_spectral:
    #     spectral_model = spectral_model.cpu()
    #     print("Moved frozen spectral model to CPU")
    
    # if args.freeze_llm:
    #     llm_model = llm_model.cpu()  
    #     print("Moved frozen LLM model to CPU")
    
    # Create multimodal model
    model = MultimodalLlamaModel(
        base_model=llm_model,
        fm_model=spectral_model,
        latent_dim=args.spectral_embedding_dim,
        hidden_dim=args.hidden_dim,
        num_spectral_features=args.num_spectral_features
    )

    print_detailed_memory()

    # Load checkpoint if exists
    if args.checkpoint_dir and os.path.exists(args.checkpoint_dir):
        checkpoint_path = os.path.join(args.checkpoint_dir, 'clip_stellar.pth')
        model = load_checkpoints_ddp(model, checkpoint_path)
    
    # CRITICAL FIX: Move model to device BEFORE DDP
    # Move only the parts that need to be on GPU
    print("Moving model components to device...")
    
    # # Always move trainable projection layers to GPU
    # if hasattr(model, 'projection_head'):
    #     model.projection_head = model.projection_head.to(device)
    # if hasattr(model, 'text_projection'):
    #     model.text_projection = model.text_projection.to(device)  
    # if hasattr(model, 'spectral_projection'):
    #     model.spectral_projection = model.spectral_projection.to(device)
    
    # # Move unfrozen models to GPU
    # if not args.freeze_llm and hasattr(model, 'base_model'):
    #     model.base_model = model.base_model.to(device)
    #     print("Moved unfrozen LLM to GPU")
        
    # if not args.freeze_spectral and hasattr(model, 'fm_model'):
    #     model.fm_model = model.fm_model.to(device)
    #     print("Moved unfrozen spectral model to GPU")
    
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()
    
    print_detailed_memory()

    model = model.to(device)
    
    # CRITICAL FIX: Only use DDP for multi-GPU/multi-node training
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    if world_size > 1:
        print(f"Using DDP for distributed training (world_size={world_size})")
        
        # Ensure model is properly on device before DDP
        model = model.to(device)
        
        model = DDP(
            model, 
            device_ids=[device], 
            find_unused_parameters=True,  # Keep True for mixed CPU/GPU model
            broadcast_buffers=False,      # Reduce communication overhead
            gradient_as_bucket_view=True  # More memory efficient
        )
    else:
        print("Single GPU training - skipping DDP")
        # For single GPU, just ensure model is on device
        # But keep frozen parts on CPU to save memory
        pass
    
    return model

def create_model_dataparallel(args, device):
    """
    Use DataParallel instead of DDP - simpler for interactive jobs
    """
    
    # Set memory allocation config
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    torch.cuda.empty_cache()
    
    print("Loading LLM model...")
    
    # CRITICAL: Initialize fairscale before loading LLaMA
    import fairscale.nn.model_parallel.initialize as fs_init
    import torch.distributed as dist
    
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("gloo", rank=0, world_size=1)
    
    if not fs_init.model_parallel_is_initialized():
        fs_init.initialize_model_parallel(1)
        print("Fairscale initialized for LLaMA")
    
    llm_model = _load_llm_model(args.batch_size, args.max_seq_length)
    spectral_model = _load_spectra_model()
    
    # Keep frozen models on CPU to save memory
    if args.freeze_spectral:
        spectral_model = spectral_model.cpu()
    if args.freeze_llm:
        llm_model = llm_model.cpu()
    
    model = MultimodalLlamaModel(
        base_model=llm_model,
        fm_model=spectral_model,
        latent_dim=args.spectral_embedding_dim,
        hidden_dim=args.hidden_dim,
        num_spectral_features=args.num_spectral_features
    )
    
    # Move trainable parts to primary GPU
    model = model.to(device)
    
    # Use DataParallel for multi-GPU (simpler than DDP)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Model wrapped with DataParallel across {torch.cuda.device_count()} GPUs")
    
    return model

# Add this detailed memory tracking to your model creation functions:

def create_optimizer_and_scheduler(model, args, train_loader):
    """Create optimizer and learning rate scheduler"""
    
    named_params = list(model.named_parameters())
    for name, param in named_params:
        if 'base_model' in name and args.freeze_llm:
            param.requires_grad = False
        if 'fm_model' in name and args.freeze_spectral:
            param.requires_grad = False

    # Collect only tensors (Parameters) with requires_grad=True
    opt_params = [p for n, p in named_params if isinstance(p, torch.nn.Parameter) and p.requires_grad]

    if len(opt_params) == 0:
        raise RuntimeError("No trainable parameters found for optimizer. Check freeze flags and model setup.")

    optimizer = torch.optim.AdamW(
        opt_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler with warmup
    total_steps = max(1, len(train_loader) * args.num_epochs)
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return step / float(warmup_steps)
        # Cosine decay after warmup
        denom = max(1, total_steps - max(0, warmup_steps))
        return 0.5 * (1 + np.cos(np.pi * (step - max(0, warmup_steps)) / denom))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=args.use_amp)

    return optimizer, scheduler, scaler

def create_trainer(model, optimizer, criterion, train_loader, val_loader,scaler, device, args):
    """Create CLIP trainer"""
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get world size
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    with open(os.path.join(ROOT_DIR, "src", "llm_config.json"), "r") as f:
        config = json.load(f)
    lora_params = config['lora_params']

    model_path, tokenizer_path = get_model_path(args)
    
    trainer = LLMTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,  # Not used for CLIP, but required by base class
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        world_size=world_size,
        output_dim=1,  # Single accuracy metric for CLIP
        scheduler=None,  # We'll handle scheduling manually
        max_iter=args.max_iter,
        log_path=args.output_dir,
        exp_name=args.exp_name,
        lora_params=lora_params,
        scaler=scaler,
        use_amp=args.use_amp,
        max_grad_norm=args.max_grad_norm 
    )
    
    return trainer

def save_config(args, output_dir):
    """Save training configuration"""
    config_path = os.path.join(output_dir, 'training_config.json')
    config_dict = vars(args)
    config_dict['llm_model'] = args.llm_model
    config_dict['json_path'] = JSON_PATH
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Training configuration saved to {config_path}.\nfull config: {config_dict}")


def main():
    """Main training function"""
    
    # Parse arguments
    args = parse_args()
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    args.output_dir = os.path.join(args.output_dir, date)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup distributed training
    local_rank, world_size, gpus_per_node = setup()
    # device = torch.cuda.current_device()
    
    print(f"Distributed training setup complete. Local rank: {local_rank}, World size: {world_size}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    
    # Create datasets and dataloaders
    print("Creating datasets and dataloaders...")
    train_loader, val_loader, test_loader = create_datasets_and_loaders(args, local_rank)

    tokenizer = train_loader.dataset.tokenizer
    
    # Create model
    print("Creating multimodal model...")
    # model = create_model(args, device)
    model = create_model_memory_optimized(args, local_rank)
    # model = create_model_dataparallel(args, device)
    # model = create_model_dataparallel_with_memory_tracking(args, device)

    # Handle DDP wrapper when accessing inner model attributes
    inner_model = model.module if isinstance(model, DDP) else model
    print(f"Model vocab_size: {inner_model.base_model.params.vocab_size}")

    # Create optimizer and scheduler
    print("Creating optimizer and scheduler...")
    optimizer, scheduler, scaler = create_optimizer_and_scheduler(model, args, train_loader)
    
    # Dummy criterion (not used for CLIP but required by base trainer)
    criterion = nn.CrossEntropyLoss()
    
    # Create trainer
    print("Creating trainer...")
    trainer = create_trainer(model, optimizer, criterion, train_loader, val_loader, scaler, local_rank, args)
    trainer.scheduler = scheduler  # Assign scheduler after trainer creation
    trainer.tokenizer = tokenizer  # Assign tokenizer for logging
    

    # Save configuration
    if local_rank == 0:  # Only save on main process
        save_config(args, args.output_dir)
    
    # Print training info
    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTraining Configuration:")
        print(f"  Experiment: {args.exp_name}")
        print(f"  Batch size: {args.batch_size} per GPU ({args.batch_size * world_size} total)")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Epochs: {args.num_epochs}")
        print(f"  Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"  Output directory: {args.output_dir}")
        print("-" * 60)
    
    # Resume if requested (model + optimizer + scheduler + scaler + epoch)
    start_epoch = 0
    if args.resume_path and os.path.exists(args.resume_path):
        print(f"Resuming from checkpoint: {args.resume_path}")
        try:
            ckpt = torch.load(args.resume_path, map_location='cpu')
        except Exception as e:
            print(f"Resume checkpoint appears corrupted or unreadable: {e}")
            # Try fallback to *_resume_best or warm-start from weights in the same dir
            resume_dir = os.path.dirname(args.resume_path)
            exp = os.path.basename(args.resume_path).split('_resume_')[0]
            fallbacks = [
                os.path.join(resume_dir, f"{exp}_resume_best.pth"),
                os.path.join(resume_dir, f"{exp}.pth"),
            ]
            alt = next((p for p in fallbacks if os.path.isfile(p)), None)
            if alt:
                print(f"Trying fallback resume from: {alt}")
                ckpt = torch.load(alt, map_location='cpu')
            else:
                print("No usable resume checkpoint found; proceeding without exact resume.")
                ckpt = None

        # Load model state
        state_dict = ckpt.get('model', ckpt) if ckpt is not None else None
        # Handle possible nested keys
        if state_dict is not None and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # Remove module. prefix if present/mismatched
        if state_dict is not None:
            new_state = {}
            for k, v in state_dict.items():
                nk = k
                if nk.startswith('module.'):
                    nk = nk[7:]
                new_state[nk] = v
            try:
                (model.module if isinstance(model, DDP) else model).load_state_dict(new_state, strict=False)
                print("✓ Model weights loaded from resume checkpoint")
            except Exception as e:
                print(f"Warning: failed to load full model state strictly ({e}); trying strict=False")
                (model.module if isinstance(model, DDP) else model).load_state_dict(new_state, strict=False)

        # Optimizer / scheduler / scaler
        if ckpt is not None:
            try:
                if 'optimizer' in ckpt and ckpt['optimizer'] is not None:
                    optimizer.load_state_dict(ckpt['optimizer'])
                    print("✓ Optimizer state loaded")
            except Exception as e:
                print(f"Warning loading optimizer state: {e}")
            try:
                if 'scheduler' in ckpt and ckpt['scheduler'] is not None and trainer.scheduler is not None:
                    trainer.scheduler.load_state_dict(ckpt['scheduler'])
                    print("✓ Scheduler state loaded")
            except Exception as e:
                print(f"Warning loading scheduler state: {e}")
            try:
                if 'scaler' in ckpt and ckpt['scaler'] is not None and scaler is not None:
                    scaler.load_state_dict(ckpt['scaler'])
                    print("✓ AMP scaler state loaded")
            except Exception as e:
                print(f"Warning loading scaler state: {e}")

        # Epoch / best metrics
        if ckpt is not None:
            start_epoch = int(ckpt.get('epoch', -1)) + 1
            initial_min_loss = ckpt.get('min_loss', None)
            initial_best_acc = ckpt.get('best_acc', None)
    else:
        initial_min_loss = None
        initial_best_acc = None

    # Train the model
    if args.train:
        if start_epoch == 0:
            trainer.evaluate_validation_samples(local_rank, 0)
        print(f"Starting training from epoch {start_epoch}...")
        results = trainer.fit(
            num_epochs=args.num_epochs,
            device=local_rank,
            early_stopping=args.early_stopping,
            best='loss',  # Use loss for early stopping
            start_epoch=start_epoch,
            initial_min_loss=initial_min_loss,
            initial_best_acc=initial_best_acc,
        )
    
    # Run final evaluation on test set
    if local_rank == 0:
        print("Running final evaluation on test set...")
        # Be robust to different LLMTrainer.predict signatures across versions
        try:
            test_results = trainer.predict(test_loader, local_rank, load_best=True, compute_embeddings=True)
        except TypeError:
            try:
                test_results = trainer.predict(test_loader, local_rank, tokenizer=tokenizer)
            except TypeError:
                test_results = trainer.predict(test_loader, local_rank)
        
        # Save test results
        test_results_path = os.path.join(args.output_dir, f'{args.exp_name}_test_results.json')
        
        # Convert tensors to lists for JSON serialization
        test_results_json = {
            'avg_loss': test_results['avg_loss'],
            'avg_accuracy': test_results['avg_accuracy'],
            'losses': test_results['losses'],
            'accuracies': test_results['accuracies'],
            'num_samples': len(test_results['descriptions'])
        }
        
        # Add retrieval metrics if computed
        if 'retrieval_metrics' in test_results:
            test_results_json['retrieval_metrics'] = test_results['retrieval_metrics']
        
        with open(test_results_path, 'w') as f:
            json.dump(test_results_json, f, indent=2)
        
        print(f"Test results saved to {test_results_path}")
        
        # Save embeddings separately (they're too large for JSON)
        if 'text_embeddings' in test_results:
            embeddings_path = os.path.join(args.output_dir, f'{args.exp_name}_test_embeddings.npz')
            np.savez(embeddings_path,
                    text_embeddings=test_results['text_embeddings'].numpy(),
                    spectral_embeddings=test_results['spectral_embeddings'].numpy(),
                    similarities=test_results['similarities'].numpy())
            print(f"Test embeddings saved to {embeddings_path}")
        
        print("\nTraining completed successfully!")
        print(f"Final test loss: {test_results['avg_loss']:.6f}")
        print(f"Final test accuracy: {test_results['avg_accuracy']:.4f}")
        
        if 'retrieval_metrics' in test_results:
            print("\nRetrieval Metrics:")
            for metric, value in test_results['retrieval_metrics'].items():
                if 'recall' in metric:
                    print(f"  {metric}: {value:.4f}")


def run_training_with_error_handling():
    """Wrapper to handle training with error recovery"""
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
    finally:
        # Cleanly tear down the process group on normal completion
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()


if __name__ == "__main__":
    """
    Example usage:
    
    # Single GPU training
    python clip_training_pipeline.py \
        --json_file data/stellar_descriptions.json \
        --features_file data/spectral_features.npy \
        --output_dir logs/clip_experiment \
        --exp_name stellar_clip_v1 \
        --batch_size 32 \
        --num_epochs 100 \
        --learning_rate 1e-4 \
        --projection_dim 256 \
        --learnable_temperature \
        --compute_retrieval_metrics
    
    # Multi-GPU SLURM training
    srun python clip_training_pipeline.py \
        --json_file data/stellar_descriptions.json \
        --features_file data/spectral_features.npy \
        --output_dir logs/clip_experiment \
        --exp_name stellar_clip_distributed \
        --batch_size 16 \
        --num_epochs 200 \
        --learning_rate 5e-4 \
        --projection_dim 512 \
        --learnable_temperature \
        --compute_retrieval_metrics
    """
    
    print("=" * 80)
    print("CLIP MULTIMODAL STELLAR MODEL TRAINING")
    print("=" * 80)
    
    run_training_with_error_handling()
