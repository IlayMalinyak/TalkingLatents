#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
import random
from tqdm import tqdm
from typing import Dict, Any, List, Optional
import math

# Add ROOT_DIR to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.simple_questions_multitok import (
    build_model_multitok,
    ensure_backend_config,
    LLMBackendConfig
)
from src.tokenizer_adapter import load_tokenizer_adapter
from data.dataset_interpert import create_stellar_dataloaders, StellarQuestionsDataset
from data.transforms import GeneralSpectrumPreprocessor, ToTensor, Compose
from src.follow_up_templates import create_follow_up_specs

def parse_args():
    parser = argparse.ArgumentParser(description="Steer LLM inference with concept directions")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--features_file', type=str, required=True, help='Path to features.npy')
    parser.add_argument('--concept_file', type=str, required=True, help='Path to concept_directions.pt')
    parser.add_argument('--output_dir', type=str, default='steering_results', help='Output directory')
    parser.add_argument('--json_file', type=str, default='data/dataset/stellar_descriptions_questions_short.json')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to test')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alphas', type=str, default='-5,-2,0,2,5', help='Comma separated list of alphas')
    parser.add_argument('--max_seq_length', type=int, default=512, help='Maximum sequence length')
    
    # Model args needed for build_model_multitok
    parser.add_argument('--llm_backend', type=str, default='llama')
    parser.add_argument('--llm_root', type=str, default='/home/ilay.kamai/work/.llama')
    parser.add_argument('--llm_model', type=str, default='Llama3.1-8B')
    parser.add_argument('--spectral_embedding_dim', type=int, default=2048) 
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_spectral_features', type=int, default=8)
    parser.add_argument('--use_cfm', action='store_true', default=False)
    parser.add_argument('--cfm_weight', type=float, default=0.1)
    parser.add_argument('--predict_stellar_params', action='store_true', default=False)
    parser.add_argument('--enable_classification', action='store_true', default=True)
    parser.add_argument('--predict_features', action='store_true', default=False)
    parser.add_argument('--feature_dim', type=int, default=2048)
    parser.add_argument('--feature_loss_weight', type=float, default=1.0)
    parser.add_argument('--mode', type=str, default='single_star')
    parser.add_argument('--llm_precision', type=str, default='fp16')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False)
    parser.add_argument('--hf_quantization', type=str, default='none')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.159, 0.5, 0.841], help='Quantiles for CQR')
    
    return parser.parse_args()

def load_concepts(path, device):
    return torch.load(path, map_location=device)

def extract_params_from_text(text: str) -> Dict[str, Optional[float]]:
    """Simple heuristic extraction of parameters from generated text."""
    params = {'Teff': None, 'logg': None, 'FeH': None}
    
    # Example patterns to look for. This needs to be robust for the actual output format.
    # The model usually outputs: "Teff = 5778 K, logg = 4.44 dex, [Fe/H] = 0.00 dex"
    
    text_lower = text.lower()
    
    # Teff
    try:
        if 'teff' in text_lower:
            # Find number after teff
            parts = text_lower.split('teff')
            after = parts[1]
            # Extract number
            import re
            match = re.search(r"[-+]?\d*\.\d+|\d+", after)
            if match:
                params['Teff'] = float(match.group())
    except: pass
    
    # logg
    try:
        if 'logg' in text_lower:
            parts = text_lower.split('logg')
            after = parts[1]
            import re
            match = re.search(r"[-+]?\d*\.\d+|\d+", after)
            if match:
                params['logg'] = float(match.group())
    except: pass

    # FeH
    try:
        if 'feh' in text_lower or '[fe/h]' in text_lower:
             # handle both
             if '[fe/h]' in text_lower:
                 parts = text_lower.split('[fe/h]')
             else:
                 parts = text_lower.split('feh')
             after = parts[1]
             import re
             match = re.search(r"[-+]?\d*\.\d+|\d+", after)
             if match:
                 params['FeH'] = float(match.group())
    except: pass
    
    return params


from src.follow_up_inference import (
    build_generation_context, 
    generate_text_for_group
)

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Initialize distributed environment for FairScale
    if not torch.distributed.is_initialized():
        print("Initializing distributed environment...")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        torch.distributed.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    
    from fairscale.nn.model_parallel.initialize import initialize_model_parallel
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        # Initialize with 1 GPU for inference
        try:
            initialize_model_parallel(1)
            print("FairScale model parallel initialized (size=1)")
        except Exception as e:
            print(f"Warning initializing model parallel: {e}")
            pass
    
    # Create backend config
    backend_config = ensure_backend_config(args)
    
    # Load Model (this will use build_model_multitok)
    print("Loading model...")
    model = build_model_multitok(args, device, backend_config=backend_config)
    
    # Load Weights
    print(f"Loading weights from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Clean state dict keys
    cleaned_sdict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            cleaned_sdict[k[7:]] = v
        else:
            cleaned_sdict[k] = v
            
    model.load_state_dict(cleaned_sdict, strict=False)
    model.to(device)
    model.eval()
    
    # Load Concepts
    print(f"Loading concepts from {args.concept_file}")
    concepts = load_concepts(args.concept_file, device)
    print(f"Available concepts: {list(concepts.keys())}")
    
    # Load Data (Test set)
    print("Loading data...")
    spectral_features = np.load(args.features_file)
    transf = Compose([GeneralSpectrumPreprocessor(rv_norm=True), ToTensor()])
    cache_dir = os.path.join(args.output_dir, 'cache')
    
    tokenizer_adapter = load_tokenizer_adapter(
       backend=backend_config.tokenizer_backend,
       tokenizer_path=backend_config.tokenizer_path, 
       hf_model_name=backend_config.model_name_or_path
    )

    print("Creating dataloaders...")
    _, _, test_loader = create_stellar_dataloaders(
        json_file=args.json_file,
        features_array=spectral_features,
        spectral_transforms=transf,
        batch_size=args.batch_size,
        num_workers=4,
        tokenizer=tokenizer_adapter,
        max_length=512,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        # normalize_features=True by default
    )
    
    # Get stats
    stats = test_loader.dataset.get_feature_normalization_stats()
    if stats:
        print("Feature normalization stats found.")
        sigma = torch.from_numpy(stats['std']).to(device)
    else:
        print("No feature normalization stats found. Assuming identity.")
        sigma = torch.ones(1).to(device)

    alphas = [float(x) for x in args.alphas.split(',')]
    
    results = []
    
    print(f"Starting generation for {args.num_samples} samples...")
    for i, batch in enumerate(tqdm(test_loader, total=min(len(test_loader), args.num_samples))):
        if i >= args.num_samples: break
        
        # Move batch to device for manual tensor ops
        h_base = batch['masked_spectra'].to(device) # [B, D]

        # Ground truth
        params_gt = {
            'Teff': batch['y_numeric'][0][0].item(), 
            'logg': batch['y_numeric'][0][1].item(), 
            'FeH': batch['y_numeric'][0][2].item()
        }
        
        # Info for this star
        try:
            obsid = batch['metadata'][0]['obsid']
        except:
            obsid = f"sample_{i}"
            
        for concept_name, direction in concepts.items():
            v = direction.to(device)
            v_norm_space = v / sigma
            
            for alpha in alphas:
                # Steer
                h_steered = h_base + alpha * v_norm_space
                
                # Clone batch
                batch_steered = {}
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch_steered[k] = v.to(device)
                    else:
                        batch_steered[k] = v

                batch_steered['masked_spectra'] = h_steered # Already on device
                
                # Prepare context
                # batch_steered needs to look like batch_data in loop
                # We are processing batch_size=1 so batch_idx=0
                # ctx, input_text, target_text = build_generation_context(batch_steered, 0)
                
                # Generate use model method
                new_text, input_text, target_text, _ = model.generate_response_from_batch(
                    batch_steered, 
                    batch_idx=0, 
                    tokenizer=tokenizer_adapter
                )
                
                # Extract params
                params_extracted = extract_params_from_text(new_text)
                
                results.append({
                    'obsid': obsid,
                    'concept': concept_name,
                    'alpha': alpha,
                    'params_gt': params_gt,
                    'params_extracted': params_extracted,
                    'input_text': input_text,
                    'target_text': target_text,
                    'generated_text': new_text
                })
                
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    save_file = os.path.join(args.output_dir, 'steering_results.json')
    with open(save_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {save_file}")

if __name__ == "__main__":
    main()
