#!/usr/bin/env python3
"""
Training script for late fusion model.
Demonstrates how to train the spectral-text alignment using CLIP loss.
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from nn.late_fusion import LateFusionModel, create_default_config
from src.simple_questions import setup, _load_llm_model, _load_spectra_model, get_model_path
from data.dataset_interpert import create_stellar_dataloaders
from data.transforms import GeneralSpectrumPreprocessor, ToTensor, Compose


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Late Fusion Multimodal Model')
    
    # Data paths
    parser.add_argument('--json_file', type=str, 
                       default='/data/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json',
                       help='Path to stellar descriptions JSON file')
    parser.add_argument('--features_file', type=str, 
                       default='/data/TalkingLatents/logs/2025-07-29/features.npy',
                       help='Path to spectral features numpy file')
    parser.add_argument('--config_file', type=str,
                       default='/data/TalkingLatents/configs/late_fusion_config.json',
                       help='Path to late fusion configuration file')
    parser.add_argument('--output_dir', type=str, default='logs',
                       help='Output directory for logs and models')
    parser.add_argument('--exp_name', type=str, default='late_fusion',
                       help='Experiment name')
    
    # Model configuration
    parser.add_argument('--llm_root', type=str, default=os.environ.get('LLM_ROOT', '/data/.llama'),
                       help='Root directory containing LLaMA models')
    parser.add_argument('--llm_model', type=str, default='Llama3.1-8B',
                       help='LLaMA model name')
    parser.add_argument('--llm_precision', type=str, default='fp16', choices=['fp32','fp16','bf16'],
                       help='Precision for LLM weights')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='Number of warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')
    
    # Data parameters
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Test set ratio')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--max_seq_length', type=int, default=128,
                       help='Maximum sequence length')
    
    # System parameters
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of dataloader workers')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use Automatic Mixed Precision')
    
    return parser.parse_args()


def create_data_loaders(args, tokenizer_path):
    """Create data loaders for training"""
    import numpy as np
    
    # Load spectral features if available
    spectral_features = None
    if args.features_file and os.path.exists(args.features_file):
        print(f"Loading spectral features from {args.features_file}")
        spectral_features = np.load(args.features_file)
        print(f"Spectral features shape: {spectral_features.shape}")
    
    # Create transforms
    transforms = Compose([GeneralSpectrumPreprocessor(rv_norm=True), ToTensor()])
    
    # Create cache directory
    cache_dir = os.path.join(args.output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_stellar_dataloaders(
        json_file=args.json_file,
        features_array=spectral_features,
        spectral_transforms=transforms,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed,
        num_spectral_features=4,  # Not used in late fusion, but required by dataset
        cache_dir=cache_dir,
        tokenizer_path=tokenizer_path,
        max_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        world_size=1,  # Single GPU for simplicity
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    return train_loader, val_loader, test_loader


def create_model(args, config):
    """Create the late fusion model"""
    print("Loading LLM model...")
    llm = _load_llm_model(args)
    
    # Apply precision settings
    if args.llm_precision == 'fp16':
        llm.half()
        print("✓ LLM weights cast to float16")
    elif args.llm_precision == 'bf16' and torch.cuda.is_bf16_supported():
        llm.to(dtype=torch.bfloat16)
        print("✓ LLM weights cast to bfloat16")
    
    # Load spectral model if needed
    fm = None
    if not args.features_file:
        fm = _load_spectra_model()
        print("✓ Loaded spectral flow model")
    else:
        print("✓ Using pre-computed features, no spectral model needed")
    
    # Move models to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    llm = llm.to(device)
    if fm is not None:
        fm = fm.to(device)
    
    # Create late fusion model
    model = LateFusionModel(llm, fm, config)
    model = model.to(device)
    
    print(f"✓ Late fusion model created with shared dim: {config['shared_dim']}")
    
    return model


def train_epoch(model, train_loader, optimizer, scaler, device, args):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Prepare input for late fusion model
        late_fusion_batch = {
            'input_ids': batch['input_ids'],
            'spectral_data': batch.get('masked_spectra', batch.get('features', torch.randn(batch['input_ids'].size(0), 2048, device=device)))
        }
        
        optimizer.zero_grad()
        
        if args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(late_fusion_batch)
                loss = outputs['contrastive_loss']
        else:
            outputs = model(late_fusion_batch)
            loss = outputs['contrastive_loss']
        
        # Backward pass
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Prepare input for late fusion model
            late_fusion_batch = {
                'input_ids': batch['input_ids'],
                'spectral_data': batch.get('masked_spectra', batch.get('features', torch.randn(batch['input_ids'].size(0), 2048, device=device)))
            }
            
            outputs = model(late_fusion_batch)
            loss = outputs['contrastive_loss']
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    args = parse_args()
    
    # Create output directory
    date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    args.output_dir = os.path.join(args.output_dir, f"{args.exp_name}_{date}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    
    # Load configuration
    if os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        print(f"✓ Loaded configuration from {args.config_file}")
    else:
        print(f"Configuration file not found, creating default config...")
        config = create_default_config()
        os.makedirs(os.path.dirname(args.config_file), exist_ok=True)
        with open(args.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Saved default configuration to {args.config_file}")
    
    # Get tokenizer path
    model_path, tokenizer_path = get_model_path(args)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(args, tokenizer_path)
    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Val samples: {len(val_loader.dataset)}")
    
    # Create model
    model = create_model(args, config)
    
    # Create optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    total_steps = len(train_loader) * args.num_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=total_steps//4, eta_min=args.learning_rate*0.1
    )
    
    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training loop
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, args)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
    
    print(f"\n✓ Training completed! Best val loss: {best_val_loss:.4f}")
    print(f"✓ Models saved to: {args.output_dir}")


if __name__ == '__main__':
    main()