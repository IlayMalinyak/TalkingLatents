from __future__ import annotations

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.system('pip install tiktoken fairscale fire blobfile torchdiffeq ' \
 'torchcfm transformers bitsandbytes accelerate')

from src.simple_questions import (
    setup,
    create_optimizer_and_scheduler,
    save_config,
    _load_llm_model,
    _load_spectra_model,
    print_detailed_memory,
    get_model_path
)
from src.tokenizer_adapter import load_tokenizer_adapter
from nn.llm_multi import MultimodalLlamaModelMultiTokens
from nn.llm_hf_multi import HuggingFaceMultimodalModel
from nn.train import LLMTrainer
from data.dataset_interpert import create_stellar_dataloaders, StellarQuestionsDataset
from data.dataset_comparative import create_comparative_dataloaders
from data.dataset_advanced import AdvancedStellarQuestionsDataset
from data.dataset_mixed import create_mixed_dataloaders
from data.dataset_feature_pred import create_feature_prediction_dataloaders
from data.transforms import GeneralSpectrumPreprocessor, ToTensor, Compose
import numpy as np
import torch.distributed as dist
import gc
import pandas as pd

from src.llm_backend_config import LLMBackendConfig


 # Data paths - use the same defaults from simple_questions.py
JSON_PATH = '/home/ilay.kamai/work/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json'
JSON_PATH_LONG = '/home/ilay.kamai/work/TalkingLatents/data/dataset/stellar_descriptions_questions.json'
ADVANCED_JSON_PATH = '/home/ilay.kamai/work/TalkingLatents/data/dataset/caption_advanced_100k.json'
FEATURES_PATH = '/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/features.npy'  # Optional, can be None to load all features on-the-fly
MULTIMODAL_PATH = '/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/multimodal_features.npy'
MULTIMODAL_DF_PATH = '/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/info_full_multimodal.csv'
FEATURES_PATH_V2 = '/home/ilay.kamai/work/TalkingLatents/logs/2025-12-16/tokens.npy' # different model


def _normalize_backend(backend):
    """Map user-facing backend aliases to internal canonical strings."""
    value = (backend or "llama").lower()
    return "hf" if value == "qwen" else value


def _parse_device_map(device_map):
    if device_map is None:
        return None
    if isinstance(device_map, (dict, list)):
        return device_map
    if isinstance(device_map, str):
        try:
            return json.loads(device_map)
        except Exception:
            return device_map
    return device_map


def ensure_backend_config(args):
    """Create (or reuse) a backend config object for downstream components."""
    existing = getattr(args, "llm_backend_config", None)
    if isinstance(existing, LLMBackendConfig):
        return existing

    backend = _normalize_backend(getattr(args, 'llm_backend', 'llama'))
    tokenizer_backend = 'llama' if backend == 'llama' else 'hf'

    model_name = None
    tokenizer_path = getattr(args, "tokenizer_path_override", None)
    if backend == 'llama':
        model_path, tokenizer_path = get_model_path(args)
        model_name = model_path
    else:
        model_name = getattr(args, 'hf_model_name', None)

    config = LLMBackendConfig(
        backend=backend,
        tokenizer_backend=tokenizer_backend,
        model_name_or_path=model_name,
        tokenizer_path=tokenizer_path,
        quantization=getattr(args, 'hf_quantization', 'none') if backend != 'llama' else 'none',
        precision=getattr(args, 'llm_precision', 'fp16'),
        device_map=_parse_device_map(getattr(args, 'hf_device_map', None)),
        cache_dir=getattr(args, 'hf_cache_dir', None),
        trust_remote_code=getattr(args, 'hf_trust_remote_code', False),
        revision=getattr(args, 'hf_revision', None),
        max_memory_gb=getattr(args, 'hf_max_memory_gb', None),
        auth_token=getattr(args, 'hf_auth_token', None),
    )
    args.llm_backend_config = config
    return config


def parse_args(argv=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train CLIP Multimodal Stellar Model')
    
    
    parser.add_argument('--use_multimodal', action='store_true', default=False,
                        help='Use multimodal dataframe features and data for follow-up turns')
    
    parser.add_argument('--json_file', type=str, default=JSON_PATH,
                       help='Path to stellar descriptions JSON file')
    parser.add_argument('--advanced_json_file', type=str, default=ADVANCED_JSON_PATH,
                       help='Path to advanced stellar questions JSON file')
    parser.add_argument('--single_dataset_type', type=str, choices=['regular', 'advanced'],
                       default='regular', help='Select which single-star dataset variant to load')
    parser.add_argument('--features_file', type=str, default=None,
                       help='Path to spectral features numpy file')
    parser.add_argument('--feature_stats_file', type=str, default=None,
                       help='Path to feature statistics file (npz with mean/std) for on-the-fly normalization')
    parser.add_argument('--pooling_type', type=str, choices=['mean', 'sum'], default='mean',
                       help='Type of pooling for spectral features (mean/sum). Use "sum" for 2025-07-29 features, "mean" for 2025-12-16 tokens.')
    parser.add_argument('--output_dir', type=str, default='logs',
                       help='Output directory for logs and models')
    parser.add_argument('--exp_name', type=str, default='interpert',
                       help='Experiment name')
    parser.add_argument('--enable_followup_augmentation', action='store_true', default=False,
                        help='Append synthetic follow-up QA turns during training')
    parser.add_argument('--followup_prob', type=float, default=0.0,
                        help='Probability of augmenting a sample with follow-up QA')
    parser.add_argument('--max_followup_turns', type=int, default=1,
                        help='Maximum number of follow-up QA turns to append per sample')
    parser.add_argument('--followup_json_file', type=str, default=JSON_PATH_LONG,
                        help='json file for followup questions')
                        
    
    # Model configuration
    parser.add_argument('--llm_backend', type=str, choices=['llama', 'hf', 'qwen'],
                       default=os.environ.get('LLM_BACKEND', 'llama'),
                       help='Backbone family to use for the language model (Meta LLaMA or Hugging Face/Qwen).')
    parser.add_argument('--llm_root', type=str, default=os.environ.get('LLM_ROOT', '/home/ilay.kamai/work/.llama'),
                       help='Root directory containing LLaMA models (ignored for Hugging Face backend).')
    parser.add_argument('--llm_model', type=str, default='Llama3.1-8B',
                       help='LLaMA model name under --llm_root (ignored for Hugging Face backend).')
    parser.add_argument('--llm_path', type=str, default=None,
                       help='Full path to the LLaMA model directory (contains params.json). Overrides llm_root/llm_model.')
    parser.add_argument('--hf_model_name', type=str, default='Qwen/Qwen2.5-32B-Instruct',
                       help='Hugging Face model identifier to use when --llm_backend=hf.')
    parser.add_argument('--hf_revision', type=str, default=None,
                       help='Specific revision/commit for the Hugging Face model.')
    parser.add_argument('--hf_trust_remote_code', action='store_true', default=False,
                       help='Allow execution of custom model code when loading the Hugging Face tokenizer/model.')
    parser.add_argument('--hf_device_map', type=str, default=None,
                       help='Optional device map string/dict (passed through to AutoModel.from_pretrained).')
    parser.add_argument('--hf_quantization', type=str, choices=['none', '8bit', '4bit'],
                       default=os.environ.get('HF_QUANTIZATION', 'none'),
                       help='Quantization mode for the Hugging Face backbone (none/8bit/4bit).')
    parser.add_argument('--hf_cache_dir', type=str,
                       default=os.environ.get('HF_CACHE_DIR', os.environ.get('HF_HOME')),
                       help='Optional cache directory override for Hugging Face model/tokenizer files.')
    parser.add_argument('--hf_max_memory_gb', type=float, default=None,
                       help='Soft cap (in GB) per device when loading the Hugging Face model.')
    parser.add_argument('--hf_auth_token', type=str, default=os.environ.get('HF_TOKEN'),
                       help='Optional Hugging Face token for gated model downloads.')
    parser.add_argument('--hf_attn_implementation', type=str, default=None,
                       help='Hugging Face attention implementation (e.g., eager, flash_attention_2, sdpa)')
    parser.add_argument('--llm_precision', type=str, default='fp16', choices=['fp32','fp16','bf16'],
                       help='Precision to hold LLM weights on GPU (fp16 recommended on V100)')
    parser.add_argument('--spectral_embedding_dim', type=int, default=2048,
                       help='Spectral model embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Common projection space dimension')
    parser.add_argument('--num_spectral_features', type=int, default=8,
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

    # Feature prediction parameters
    parser.add_argument('--predict_stellar_params', action='store_true', default=False,
                   help='Enable stellar params prediction')
    parser.add_argument('--predict_features', action='store_true', default=False,
                   help='Enable feature prediction from hidden states')
    parser.add_argument('--feature_dim', type=int, default=2048,
                   help='Dimension of features to predict (default: 2048)')
    parser.add_argument('--feature_loss_weight', type=float, default=1.0,
                   help='Weight for feature prediction loss (default: 1.0)')
    parser.add_argument('--random_pairing', action='store_true', default=False,
                       help='pairing strategy - random (True) or nearest neighbor (False)')

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
    parser.add_argument('--loss_lambda', type=float, default=1.0,
                       help='Weight for cross-entropy loss when combining with stellar parameter loss')
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
    parser.add_argument('--enable_classification', action='store_true', default=True,
                       help='Enable classification head for comparative questions (default: True)')
    parser.add_argument('--disable_classification', action='store_true', default=False,
                       help='Disable classification head to save memory')
    
    parser.add_argument('--single_sample_prob', type=float, default=1.0,
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
    
    parser.add_argument('--v2', action='store_true', default=False,
                        help='Use V2 features path')
    
    args = parser.parse_args(argv)
    
    # Handle conflicting flags: disable takes precedence
    if args.disable_classification:
        args.enable_classification = False
        
    # Handle V2 features override
    if args.v2 and args.features_file:
        print(f"Using V2 features path: {FEATURES_PATH_V2}")
        args.features_file = FEATURES_PATH_V2
        
    return args


def resolve_resume_directory(args):
    """Determine resume checkpoint file and output directory if resuming."""
    resume_path = getattr(args, 'resume_path', None)
    if not resume_path:
        return None, None

    resume_path = os.path.expanduser(resume_path)
    resume_dir = None
    resolved_ckpt = None

    if os.path.isdir(resume_path):
        resume_dir = resume_path
    elif os.path.isfile(resume_path):
        resume_dir = os.path.dirname(resume_path)
        resolved_ckpt = resume_path
    else:
        print(f"Warning: resume_path '{resume_path}' not found.")
        return None, None

    if resolved_ckpt is None:
        exp_name = getattr(args, 'exp_name', None)
        candidates = []
        if exp_name:
            candidates.extend([
                os.path.join(resume_dir, f"{exp_name}_resume_last.pth"),
                os.path.join(resume_dir, f"{exp_name}_resume_best.pth"),
                os.path.join(resume_dir, f"{exp_name}.pth"),
            ])
        candidates.extend(
            str(p) for p in sorted(Path(resume_dir).glob("*resume*.pth"),
                                   key=lambda p: p.stat().st_mtime,
                                   reverse=True)
        )
        candidates.extend(
            str(p) for p in sorted(Path(resume_dir).glob("*.pth"),
                                   key=lambda p: p.stat().st_mtime,
                                   reverse=True)
        )
        for candidate in candidates:
            if os.path.isfile(candidate):
                resolved_ckpt = candidate
                break

    if resolved_ckpt is None:
        print(f"Warning: no checkpoint file found inside '{resume_dir}'. Proceeding without weight loading.")

    return resume_dir, resolved_ckpt


def load_weights_into_model(model, checkpoint_path, device, rank=0):
    """
    Load checkpoint weights into the provided model, mirroring src/inference.py semantics.

    Returns:
        ckpt_dict: dict with optimizer/scheduler/scaler metadata (empty if unavailable)
        load_info: tuple(missing_keys, unexpected_keys)
    """
    if rank == 0:
        print(f"Loading checkpoint weights from: {checkpoint_path}")

    if isinstance(device, int):
        device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    raw_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(raw_checkpoint, dict):
        ckpt_dict = raw_checkpoint
    else:
        ckpt_dict = {}

    if isinstance(raw_checkpoint, dict):
        if 'model_state_dict' in raw_checkpoint:
            state_dict = raw_checkpoint['model_state_dict']
        elif 'model' in raw_checkpoint and isinstance(raw_checkpoint['model'], dict):
            state_dict = raw_checkpoint['model']
            if 'state_dict' in state_dict and isinstance(state_dict['state_dict'], dict):
                state_dict = state_dict['state_dict']
        elif 'state_dict' in raw_checkpoint and isinstance(raw_checkpoint['state_dict'], dict):
            state_dict = raw_checkpoint['state_dict']
        else:
            state_dict = raw_checkpoint
    else:
        state_dict = raw_checkpoint

    cleaned_state = {}
    if isinstance(state_dict, dict):
        for key, value in state_dict.items():
            new_key = key[7:] if key.startswith('module.') else key
            cleaned_state[new_key] = value
    else:
        if rank == 0:
            print("Warning: resume checkpoint does not contain a recognized state_dict structure.")
        cleaned_state = {}

    target_model = model.module if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)) else model
    missing_keys, unexpected_keys = target_model.load_state_dict(cleaned_state, strict=False)

    if rank == 0:
        print("✓ Model weights loaded from checkpoint")
        if missing_keys:
            preview = missing_keys[:5]
            suffix = " ..." if len(missing_keys) > 5 else ""
            print(f"  Missing keys: {preview}{suffix}")
        if unexpected_keys:
            preview = unexpected_keys[:5]
            suffix = " ..." if len(unexpected_keys) > 5 else ""
            print(f"  Unexpected keys: {preview}{suffix}")

    del cleaned_state, state_dict
    model.to(device)

    return ckpt_dict, (missing_keys, unexpected_keys)


def prepare_training_with_resume(
    args,
    model,
    train_loader,
    val_loader,
    tokenizer,
    lora_params,
    local_rank,
    world_size,
    backend_config: LLMBackendConfig | None = None,
):
    """
    Create optimizer/scheduler/trainer and restore checkpoint weights and states if provided.
    Returns:
        optimizer, scheduler, scaler, trainer, start_epoch, initial_min_loss, initial_best_acc
    """
    resume_ckpt_path = getattr(args, 'resume_path', None)
    ckpt = {}
    start_epoch = 0
    initial_min_loss = None
    initial_best_acc = None

    def _load_checkpoint(path):
        if local_rank == 0:
            print(f"Attempting to load resume checkpoint: {path}")
        return load_weights_into_model(
            model=model,
            checkpoint_path=path,
            device=local_rank,
            rank=local_rank
        )[0]

    if resume_ckpt_path and os.path.isfile(resume_ckpt_path):
        try:
            ckpt = _load_checkpoint(resume_ckpt_path)
        except Exception as e:
            ckpt = {}
            if local_rank == 0:
                print(f"Failed to load checkpoint '{resume_ckpt_path}': {e}")
            resume_dir = os.path.dirname(resume_ckpt_path)
            exp = args.exp_name
            fallbacks = [
                os.path.join(resume_dir, f"{exp}_resume_best.pth"),
                os.path.join(resume_dir, f"{exp}.pth"),
            ]
            for candidate in fallbacks:
                if not os.path.isfile(candidate):
                    continue
                try:
                    ckpt = _load_checkpoint(candidate)
                    args.resume_path = candidate
                    if local_rank == 0:
                        print(f"Fallback checkpoint loaded: {candidate}")
                    break
                except Exception as inner_e:
                    if local_rank == 0:
                        print(f"Failed fallback checkpoint '{candidate}': {inner_e}")
            else:
                if local_rank == 0:
                    print("No usable resume checkpoint found; proceeding without loading weights.")
                ckpt = {}

    if local_rank == 0:
        print("Creating optimizer, scheduler, scaler, and trainer...")
    optimizer, scheduler, scaler = create_optimizer_and_scheduler(model, args, train_loader)

    trainer = LLMTrainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=local_rank,
        world_size=world_size,
        output_dim=1,
        scheduler=None,
        max_iter=args.max_iter,
        log_path=args.output_dir,
        exp_name=args.exp_name,
        lora_params=lora_params,
        scaler=scaler,
        use_amp=args.use_amp,
        max_grad_norm=args.max_grad_norm,
        mode=args.mode,
        curriculum_decay_steps=args.curriculum_decay_steps,
        quantiles=args.quantiles,
        loss_lambda=args.loss_lambda,
        backend_config=backend_config,
    )
    trainer.combined_mode = (args.mode == "combined")
    trainer.scheduler = scheduler
    trainer.tokenizer = tokenizer

    if isinstance(ckpt, dict) and ckpt:
        try:
            if 'optimizer' in ckpt and ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                if local_rank == 0:
                    print("✓ Optimizer state loaded")
        except Exception as e:
            if local_rank == 0:
                print(f"Warning loading optimizer state: {e}")
        try:
            if 'scheduler' in ckpt and ckpt['scheduler'] is not None and trainer.scheduler is not None:
                trainer.scheduler.load_state_dict(ckpt['scheduler'])
                if local_rank == 0:
                    print("✓ Scheduler state loaded")
        except Exception as e:
            if local_rank == 0:
                print(f"Warning loading scheduler state: {e}")
        try:
            if 'scaler' in ckpt and ckpt['scaler'] is not None and scaler is not None:
                scaler.load_state_dict(ckpt['scaler'])
                if local_rank == 0:
                    print("✓ AMP scaler state loaded")
        except Exception as e:
            if local_rank == 0:
                print(f"Warning loading scaler state: {e}")

        ckpt_backend = ckpt.get('backend_config')
        if backend_config is not None and ckpt_backend:
            try:
                ckpt_backend_name = ckpt_backend.get('backend')
            except AttributeError:
                ckpt_backend_name = ckpt_backend
            if ckpt_backend_name != backend_config.backend:
                if local_rank == 0:
                    print(f"Warning: Checkpoint backend '{ckpt_backend_name}' does not match current backend '{backend_config.backend}'.")
        start_epoch = int(ckpt.get('epoch', -1)) + 1 if 'epoch' in ckpt else 0
        initial_min_loss = ckpt.get('min_loss', None)
        initial_best_acc = ckpt.get('best_acc', None)
    else:
        if resume_ckpt_path and local_rank == 0:
            print("Resume checkpoint did not include optimizer/scheduler metadata; continuing with fresh states.")

    return optimizer, scheduler, scaler, trainer, start_epoch, initial_min_loss, initial_best_acc


def create_datasets_and_loaders(args, device, backend_config: LLMBackendConfig | None = None):
    """Create datasets and dataloaders with mode support - only on rank 0 for memory efficiency"""
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    # Handle missing attributes for backward compatibility with old configs
    random_pairing = getattr(args, 'random_pairing', False)
    
    # Only rank 0 loads spectral features to avoid OOM
    spectral_features = None

    # Handle multimodal override
    features_file = args.features_file
    multimodal_df = None
    if getattr(args, 'use_multimodal', False):
        print("Using MULTIMODAL mode")
        
        features_file = MULTIMODAL_PATH
        print(f"Loading multimodal dataframe from {MULTIMODAL_DF_PATH}")
        if os.path.exists(MULTIMODAL_DF_PATH):
            multimodal_df = pd.read_csv(MULTIMODAL_DF_PATH, index_col=0) # Assuming index is useful
            # Ensure index is int if possible for obsid matching
            try:
                multimodal_df.index = multimodal_df.index.astype(int)
            except:
                pass
        else:
            print(f"Warning: Multimodal DF path {MULTIMODAL_DF_PATH} not found!")

    print(features_file)
    if features_file and os.path.exists(features_file):
        print(f"Loading spectral features from {features_file}")
        spectral_features = np.load(features_file)
        print(f"Spectral features shape: {spectral_features.shape}")
    else:
        print("No spectral features file provided or file not found. Will use raw spectra on-the-fly.")
    dataset_type = getattr(args, "single_dataset_type", "regular")
    # if dataset_type == "advanced":
    #     single_dataset_cls = AdvancedStellarQuestionsDataset
    #     single_json_file = getattr(args, "advanced_json_file", None) or args.json_file
    # else:
    single_dataset_cls = StellarQuestionsDataset
    single_json_file = args.json_file

    # args.json_file = single_json_file

    followup_kwargs = dict(
        enable_followup=getattr(args, 'enable_followup_augmentation', False),
        followup_prob=getattr(args, 'followup_prob', 0.0),
        max_followup_turns=getattr(args, 'max_followup_turns', 1),
        followup_seed=getattr(args, 'random_seed', 42),
        followup_json_file=getattr(args, 'followup_json_file', None)
    )

    # # Synchronize before proceeding
    # if dist.is_initialized():
    #     dist.barrier()
    
    backend_config = backend_config or ensure_backend_config(args)
    tokenizer_backend = backend_config.tokenizer_backend
    tokenizer_path = backend_config.tokenizer_path
    model_path = backend_config.model_name_or_path
    if tokenizer_backend == 'llama' and tokenizer_path is None:
        model_path, tokenizer_path = get_model_path(args)
        backend_config.tokenizer_path = tokenizer_path
    tokenizer_adapter = load_tokenizer_adapter(
        backend=tokenizer_backend,
        tokenizer_path=tokenizer_path,
        hf_model_name=backend_config.model_name_or_path if tokenizer_backend != 'llama' else None,
        trust_remote_code=backend_config.trust_remote_code,
        hf_revision=backend_config.revision,
    )
    transf = Compose([GeneralSpectrumPreprocessor(rv_norm=True), ToTensor()])
    
    # Create cache directory for split consistency
    cache_dir_base = os.path.join(args.output_dir, 'cache')
    cache_dir = cache_dir_base if rank == 0 else f"{cache_dir_base}_r{rank}"
    os.makedirs(cache_dir, exist_ok=True)

            
    if rank == 0:
        print(f"Creating {dataset_type} single-star datasets from {single_json_file}...")

    train_loader, val_loader, test_loader = create_stellar_dataloaders(
        json_file=single_json_file,
        features_array=spectral_features,
        spectral_transforms=transf,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed,
        num_spectral_features=args.num_spectral_features,
        cache_dir=cache_dir,
        tokenizer_path=tokenizer_path,
        tokenizer=tokenizer_adapter,
        tokenizer_backend=tokenizer_backend,
        max_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers // world_size if world_size > 1 else args.num_workers,
        world_size=world_size,
        device=device,
        dataset_cls=single_dataset_cls,
        enable_followup=followup_kwargs['enable_followup'],
        followup_prob=followup_kwargs['followup_prob'],
        max_followup_turns=followup_kwargs['max_followup_turns'],
        followup_seed=followup_kwargs['followup_seed'],
        followup_json_file=followup_kwargs['followup_json_file'],
        multimodal_df=multimodal_df
    )

    # # Synchronize all processes after dataset creation
    # if dist.is_initialized():
    #     dist.barrier()

    for loader in (train_loader, val_loader, test_loader):
        if loader is not None and hasattr(loader, "dataset"):
            setattr(loader.dataset, "backend_config", backend_config)

    return train_loader, val_loader, test_loader


def build_model_multitok(args, local_rank, world_size=1, backend_config: LLMBackendConfig | None = None, feature_stats: Optional[Dict[str, Any]] = None):
    """Build the multimodal model with components"""
    pooling_type = getattr(args, 'pooling_type', 'mean')
    if args.use_multimodal:
        print("multimodal setting. Instead of explicit fm model, using features_file")
        fm = None
    else:
        print("loading fm model...")
        fm = None if (args.features_file and args.features_file != 'None') else _load_spectra_model(args)
    if fm is not None:
        fm = fm.to(local_rank)
    
    backend_config = backend_config or ensure_backend_config(args)
    backend = _normalize_backend(backend_config.backend)

    # OOM Fix: Force local device mapping for DDP with HF models
    # "auto" often leads to massive memory duplication or inefficient loading in DDP
    if backend != 'llama' and world_size > 1:
        current_map = getattr(args, 'hf_device_map', None)
        if current_map == 'auto' or current_map is None:
            print(f"[Rank {local_rank}] Overriding hf_device_map 'auto' to {{'': {local_rank}}} to prevent DDP OOM.")
            # Update args
            args.hf_device_map = {"": local_rank}
            # Update config object which might be used inside _load_llm_model or later
            if backend_config:
                backend_config.device_map = {"": local_rank}

    print(f"Loading LLM model (backend={backend_config.backend})...")
    
    # Serialized loading to prevent OOM when multiple processes try to load 32B model to CPU/RAM simultaneously
    if world_size > 1:
        for r in range(world_size):
            if r == local_rank:
                print(f"[Rank {local_rank}] Loading LLM...")
                llm = _load_llm_model(args)
            dist.barrier()
    else:
        llm = _load_llm_model(args)
    hf_quantization = getattr(args, "hf_quantization", "none")
    is_hf_backend = backend != 'llama'

    hf_quantization = (hf_quantization or "none").lower()
    if is_hf_backend and world_size > 1:
        print("Warning: HF/Qwen backend is only validated for single-GPU runs. Proceed with caution.")
    if is_hf_backend:
        if hasattr(llm, "config") and hasattr(llm.config, "use_cache"):
            llm.config.use_cache = False
        if args.gradient_checkpointing and hf_quantization in {'4bit', '8bit'}:
            print("Warning: Gradient checkpointing is not supported with 4/8-bit HF quantization. Disabling it.")
            args.gradient_checkpointing = False
        if args.gradient_checkpointing and hasattr(llm, "gradient_checkpointing_enable"):
            llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            print("✓ Enabled HF gradient checkpointing (use_reentrant=False)")
    else:
        if args.llm_precision == 'fp16':
            llm.half()
            print("✓ LLM weights cast to float16")
        elif args.llm_precision == 'bf16' and torch.cuda.is_bf16_supported():
            llm.to(dtype=torch.bfloat16)
            print("✓ LLM weights cast to bfloat16")

    # Disable gradient checkpointing for multi-GPU to avoid DDP issues on the LLaMA backend
    use_checkpoint = bool(args.gradient_checkpointing and (world_size == 1) and not is_hf_backend)
    if world_size > 1 and args.gradient_checkpointing and not is_hf_backend:
        print("Warning: Disabling gradient checkpointing for multi-GPU training to avoid DDP conflicts")

    if not is_hf_backend:
        llm = llm.to(local_rank)

    enable_cls = getattr(args, 'enable_classification', False)
    model_cls = MultimodalLlamaModelMultiTokens if not is_hf_backend else HuggingFaceMultimodalModel

    hf_device_map = getattr(args, "hf_device_map", None)

    model = model_cls(
        base_model=llm,
        fm_model=fm,
        latent_dim=args.spectral_embedding_dim,
        hidden_dim=args.hidden_dim,
        num_spectral_features=args.num_spectral_features,
        mode=args.mode,
        use_cfm=args.use_cfm,
        cfm_weight=args.cfm_weight,
        predict_stellar_params=args.predict_stellar_params,
        stellar_params=['Teff', 'logg', 'FeH'],
        quantiles=args.quantiles,
        use_checkpoint=use_checkpoint,
        enable_classification=enable_cls,
        predict_features=args.predict_features,
        feature_dim=args.feature_dim,
        feature_loss_weight=args.feature_loss_weight,
        feature_stats=feature_stats,
        pooling_type=pooling_type,
    )

    def _move_auxiliary_modules(mod: torch.nn.Module, target_device: torch.device) -> None:
        movable = [
            'projector', 'projector_a', 'projector_b',
            'stellar_predictor', 'stellar_transformer',
            'classification_head', 'feature_predictor',
            'flow_bridge'
        ]
        for name in movable:
            component = getattr(mod, name, None)
            if component is not None:
                component.to(target_device)
        if getattr(mod, 'fm_model', None) is not None:
            mod.fm_model.to(target_device)

    if is_hf_backend:
        requires_manual_placement = (
            hf_quantization in {'4bit', '8bit'} or
            (hf_device_map is not None and str(hf_device_map).lower() not in {'', 'none', 'cpu'})
        )
        if requires_manual_placement:
            _move_auxiliary_modules(model, local_rank)
            print(f"Keeping HF backbone on its device map; moved auxiliary heads to {local_rank}.")
        else:
            model = model.to(local_rank)
    else:
        model = model.to(local_rank)
    
    # Note: stellar predictor, stellar transformer, and feature predictor will be set to FP32 after DDP wrapping
    return model


def _apply_lora_with_regex(model, lora_config, local_rank=0):
    if local_rank == 0:
        print("Applying LoRA layers (pre-DDP/optimizer)...")
        
    # Get supported linear layer types
    try:
        from fairscale.nn.model_parallel.layers import RowParallelLinear, ColumnParallelLinear
        linear_types = (torch.nn.Linear, RowParallelLinear, ColumnParallelLinear)
    except ImportError:
        linear_types = (torch.nn.Linear,)

    # Identify all linear modules
    all_modules = []
    for name, module in model.named_modules():
        if isinstance(module, linear_types):
            all_modules.append(name)

    # Resolve wildcards
    target_modules = []
    import re
    patterns = lora_config.get('lora_target_modules', [])
    
    for pattern in patterns:
        if '*' in pattern:
            pattern_regex = pattern.replace('.', r'\.').replace('*', r'[^.]+')
            pattern_regex = f"^{pattern_regex}$"
            for name in all_modules:
                if re.match(pattern_regex, name):
                    target_modules.append(name)
        else:
            if pattern in all_modules:
                target_modules.append(pattern)

    # Apply LoRA
    from nn.lora import apply_lora_to_model
    applied = apply_lora_to_model(
        model,
        target_modules,
        rank=lora_config.get('lora_rank', 16),
        alpha=lora_config.get('lora_alpha', 16.0),
        dropout=lora_config.get('lora_dropout', 0.1)
    )
    if local_rank == 0:
        print(f"Pre-applied LoRA to {len(applied)} modules")
    return applied


def main():
    args = parse_args()
    resume_dir, resolved_resume = resolve_resume_directory(args)

    if resume_dir:
        args.output_dir = resume_dir
        if resolved_resume:
            args.resume_path = resolved_resume
        print(f"Resuming run; using output directory: {args.output_dir}")
        if resolved_resume:
            print(f"Resolved resume checkpoint: {resolved_resume}")
    else:
        date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        dataset_tag = getattr(args, "single_dataset_type", "regular")
        if getattr(args, "mode", "") != "single_star":
            dataset_tag = args.mode
        lambda_tag = str(args.loss_lambda).replace('.', 'p')
        run_name = f"{dataset_tag}_loss{lambda_tag}_{date}"
        args.output_dir = os.path.join(args.output_dir, run_name)
    
    local_rank, world_size, _ = setup()
    
    # Only rank 0 creates output directory to avoid multiple log dirs
    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Synchronize all processes after directory creation
    if dist.is_initialized():
        dist.barrier()
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    # Clear GPU memory and force garbage collection before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    backend_config = ensure_backend_config(args)
    if local_rank == 0:
        print(
            f"LLM backend configuration: backend={backend_config.backend}, "
            f"tokenizer_backend={backend_config.tokenizer_backend}, "
            f"model={backend_config.model_name_or_path}, "
            f"quantization={backend_config.quantization}"
        )

    print("Creating datasets and dataloaders...")
    train_loader, val_loader, test_loader = create_datasets_and_loaders(args, local_rank, backend_config)

    # Extract feature statistics for normalization
    feature_stats = None
    if local_rank == 0:
        # 1. Try to get stats from dataset (if features_file was provided, this is already populated)
        feature_stats = train_loader.dataset.get_feature_normalization_stats()
        
        # 2. If not in dataset (i.e. using on-the-fly), try loading from stats file
        if feature_stats is None and args.feature_stats_file:
            print(f"Loading feature stats from {args.feature_stats_file}")
            try:
                stats = np.load(args.feature_stats_file)
                feature_stats = {
                    'mean': stats['mean'],
                    'std': stats['std']
                }
                print("✓ Loaded feature stats from file")
            except Exception as e:
                print(f"Error loading feature stats file: {e}")
        
        # 3. If likely needed but missing, print warning (computation is complex due to fm_model dependency)
        if feature_stats is None and args.features_file is None:
             print("Warning: No feature stats found for on-the-fly generation. Features will NOT be normalized.")
             # NOTE: Implementing full on-the-fly computation here is tricky because fm_model is not loaded yet.
             # Ideally one would load fm_model, run a pass on train_loader, compute stats, and then build full model.
             # For now, we rely on the user providing a stats file if they want normalization.
    
    # Broadcast stats to other ranks if needed (simple approximation for now: just pass None on non-zero ranks
    # or rely on model doing its own things. But properly we should broadcast. 
    # For now, build_model_multitok runs on all ranks, so all ranks need the stats.
    # However, create_datasets_and_loaders logic for stats is cleaner on rank 0.
    # Let's simple pass feature_stats to build_model_multitok and let it handle distribution or just pass on all ranks if available.
    # Since dataset is created on all ranks (with different samplers), train_loader.dataset should have access to stats
    # IF features_loader was used. If features_loader was NOT used, then train_loader.dataset.get_feature_normalization_stats() is None.
    # So if we loaded from file on rank 0, we might need to broadcast or just load on all ranks.
    # Given args are same, all ranks can load from args.feature_stats_file.
    
    if feature_stats is None and args.feature_stats_file and os.path.exists(args.feature_stats_file):
         # Redundant load for all ranks if not already handled
         try:
             stats = np.load(args.feature_stats_file)
             feature_stats = {'mean': stats['mean'], 'std': stats['std']}
         except: pass

    
    # Clear memory after dataset creation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Handle tokenizer extraction for combined mode
    if args.mode == "combined":
        # For mixed datasets, get tokenizer from the single dataset inside the mixed dataset
        tokenizer = train_loader.dataset.single_dataset.tokenizer
    else:
        tokenizer = train_loader.dataset.tokenizer

    print("Creating multitoken multimodal model...")
    model = build_model_multitok(args, local_rank, world_size, backend_config, feature_stats=feature_stats)

    # Clear memory after model creation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Model mode after creation: {model.mode}")
    if hasattr(model, 'module'):
        print(f"Model.module mode: {model.module.mode}")

    # Freeze large submodules BEFORE wrapping with DDP so the reducer
    # only tracks truly trainable parameters (avoids unused-grad errors).
    base = model
    if isinstance(base, DDP):
        base = base.module
    if hasattr(base, 'base_model') and base.base_model is not None:
        for p in base.base_model.parameters():
            p.requires_grad = False
        print("✓ Frozen base LLaMA parameters")
    if hasattr(base, 'fm_model') and base.fm_model is not None:
        for p in base.fm_model.parameters():
            p.requires_grad = False
        print("✓ Frozen spectral FM parameters")
    
    # Clear memory after freezing parameters
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print_detailed_memory()

    # Freeze large submodules BEFORE wrapping with DDP...
    # (already done above)
    
    # Load tuned LoRA config (attention-only by default)
    # Moved here to apply LoRA BEFORE DDP wrapping
    tuned_cfg_path = os.path.join(ROOT_DIR, 'src', 'llm_config_tuned.json')
    if os.path.isfile(tuned_cfg_path):
        with open(tuned_cfg_path, 'r') as f:
            tuned_cfg = json.load(f)
        lora_params = tuned_cfg.get('lora_params', {})
    else:
        # Fallback to base config if tuned not found
        base_cfg_path = os.path.join(ROOT_DIR, 'src', 'llm_config.json')
        with open(base_cfg_path, 'r') as f:
            base_cfg = json.load(f)
        lora_params = base_cfg.get('lora_params', {})

    # Apply LoRA before DDP wrapping if enabled
    if lora_params and lora_params.get('freeze_strategy') == 'lora':
         print(f"Applying LoRA at init (start_epoch={lora_params.get('lora_start_epoch')}) to ensure parameter tracking.")
         _ = _apply_lora_with_regex(model, lora_params, local_rank)
             
    # Clear memory after LoRA application
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("fm model in final model: ", model.fm_model)

    if world_size > 1:
        print(f"Wrapping with DDP (world_size={world_size})")
        # All large modules are frozen; but LoRA may be applied later and
        # not participate immediately. Use find_unused_parameters=True to
        # avoid reducer errors in early epochs.
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
            broadcast_buffers=False,
            bucket_cap_mb=25,
        )
    else:
        print("Single process - no DDP")

    # Keep stellar predictor, transformer, and feature predictor in FP32 for numerical stability (after DDP wrapping)
    base_model = model.module if isinstance(model, DDP) else model
    if hasattr(base_model, 'stellar_predictor') and base_model.stellar_predictor is not None:
        base_model.stellar_predictor.float()
        print("✓ Stellar predictor set to float32")
    if hasattr(base_model, 'stellar_transformer') and base_model.stellar_transformer is not None:
        for layer in base_model.stellar_transformer:
            layer.float()
        print("✓ Stellar transformer set to float32")
    if hasattr(base_model, 'feature_predictor') and base_model.feature_predictor is not None:
        base_model.feature_predictor.float()
        print("✓ Feature predictor set to float32")
    # Note: Classification head remains in model precision (FP16/BF16) for memory efficiency
    if hasattr(base_model, 'classification_head') and base_model.classification_head is not None:
        print("✓ Classification head using model precision for memory efficiency")
        
    # Ensure projectors are in float32 for stability if they exist
    for proj_name in ['projector', 'projector_a', 'projector_b']:
        if hasattr(base_model, proj_name) and getattr(base_model, proj_name) is not None:
            mod = getattr(base_model, proj_name)
            mod.float()
            # for p in mod.parameters():
            #     p.requires_grad = True
            print(f"✓ {proj_name} set to float32")

    print("Preparing trainer configuration...")
    # LoRA params already loaded above
    
    optimizer, scheduler, scaler, trainer, start_epoch, initial_min_loss, initial_best_acc = prepare_training_with_resume(
        args=args,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        lora_params=lora_params,
        local_rank=local_rank,
        world_size=world_size,
        backend_config=backend_config,
    )

    if local_rank == 0:
        backend_cfg_dict = backend_config.to_dict() if hasattr(backend_config, "to_dict") else backend_config
        save_config(args, args.output_dir, backend_config=backend_cfg_dict)

    if args.train:
        if start_epoch == 0:
            trainer.evaluate_validation_samples(local_rank, 0)
        fit_res = trainer.fit(
            num_epochs=args.num_epochs,
            device=local_rank,
            early_stopping=args.early_stopping,
            best='loss',
            start_epoch=start_epoch,
            initial_min_loss=initial_min_loss,
            initial_best_acc=initial_best_acc
        )
        # Only rank 0 saves training results to avoid multiple files
        if local_rank == 0:
            output_filename = f'{args.output_dir}/fit_res.json'
            with open(output_filename, "w") as f:
                json.dump(fit_res, f, indent=2) 


if __name__ == '__main__':
    print("="*80)
    print("MULTIMODAL STELLAR MODEL TRAINING (Multi spectral tokens)")
    print("Supports both single-star and two-star comparative modes")
    print("="*80)
    main()
