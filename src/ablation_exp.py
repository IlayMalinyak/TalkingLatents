#!/usr/bin/env python3
"""
Ablation Study: Testing the Impact of num_spectral_features

This script runs multiple training experiments with different values of num_spectral_features
to understand how the number of spectral tokens affects model performance.
"""
from __future__ import annotations

import os
import sys
import json
import copy
import time
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Reuse the project's existing pipeline
from src.simple_questions_multitok import (
    ensure_backend_config,
    create_datasets_and_loaders,
    build_model_multitok,
    prepare_training_with_resume,
)
from src.simple_questions_multitok import parse_args as base_parse_args
from torch.nn.parallel import DistributedDataParallel as DDP
from src.simple_questions import setup


@dataclass
class SpectralTokensAblationConfig:
    """Configuration for a single ablation experiment."""
    num_spectral_features: int
    label: str
    
    def __post_init__(self):
        if not self.label:
            self.label = f"spectral_tokens_{self.num_spectral_features}"


def make_base_args(output_dir: str) -> Any:
    """
    Create base arguments matching cluster/followup_train_athena.sbatch.
    
    Parameters
    ----------
    output_dir : str
        Directory to save experiment outputs
        
    Returns
    -------
    args : Namespace
        Parsed arguments object
    """
    args = base_parse_args([])
    
    # Match sbatch parameters
    args.llm_path = '/home/ilay.kamai/work/.llama/Llama3.1-8B'
    args.json_file = '/home/ilay.kamai/work/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json'
    args.followup_json_file = '/home/ilay.kamai/work/TalkingLatents/data/dataset/stellar_descriptions_questions.json'
    args.features_file = '/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/features.npy'
    args.feature_stats_file = '/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/feature_stats_corrected.npz'
    
    # Training mode
    args.mode = 'single_star'
    args.single_dataset_type = 'regular'
    
    # Hyperparameters from sbatch (with short training)
    args.batch_size = 8
    args.num_epochs = 6  # Short ablation
    args.learning_rate = 1e-4
    args.weight_decay = 1e-3
    args.max_seq_length = 256
    args.num_workers = 4
    args.pooling_type = 'sum'
    args.loss_lambda = 1.0
    args.llm_precision = 'fp16'
    
    # Training stability
    args.use_amp = True
    args.gradient_checkpointing = True
    args.max_grad_norm = 1.0
    
    # Disable classification for this ablation
    args.enable_classification = False
    args.disable_classification = True
    
    # Enable follow-up augmentation as in sbatch
    args.enable_followup_augmentation = True
    args.followup_prob = 0.7
    args.max_followup_turns = 1
    
    # Other settings
    args.random_seed = 42
    args.output_dir = output_dir
    
    # Backend
    args.llm_backend = 'llama'
    args.hf_quantization = 'none'
    
    return args


def apply_ablation_config(args: Any, cfg: SpectralTokensAblationConfig) -> Any:
    """
    Apply ablation-specific configuration to base args.
    
    Parameters
    ----------
    args : Namespace
        Base arguments
    cfg : SpectralTokensAblationConfig
        Ablation configuration
        
    Returns
    -------
    args : Namespace
        Modified arguments
    """
    args = copy.deepcopy(args)
    
    # Set the number of spectral features
    args.num_spectral_features = cfg.num_spectral_features
    
    # Create informative experiment name
    args.exp_name = f"ablation_spectral_tokens_{cfg.num_spectral_features}"
    
    return args


def run_single_experiment(
    cfg: SpectralTokensAblationConfig,
    base_args: Any,
    device: int = 0,
    world_size: int = 1
) -> Dict[str, Any]:
    """
    Run a single ablation experiment.
    
    Parameters
    ----------
    cfg : SpectralTokensAblationConfig
        Configuration for this experiment
    base_args : Namespace
        Base arguments
    device : int
        Device rank
    world_size : int
        Number of GPUs
        
    Returns
    -------
    results : dict
        Dictionary containing training results and metrics
    """
    if device == 0:
        print(f"\n{'='*80}")
        print(f"Running Experiment: {cfg.label}")
        print(f"num_spectral_features = {cfg.num_spectral_features}")
        print(f"{'='*80}\n")
    
    args = apply_ablation_config(base_args, cfg)
    
    # Create output directory for this experiment
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    args.output_dir = exp_dir
    
    # Initialize backend config
    backend_config = ensure_backend_config(args)
    
    # Create datasets and loaders
    if device == 0:
        print("Creating datasets and dataloaders...")
    train_loader, val_loader, test_loader = create_datasets_and_loaders(
        args, device=device, backend_config=backend_config
    )
    
    # Build model
    if device == 0:
        print("Building model...")
    model = build_model_multitok(
        args, 
        local_rank=device, 
        world_size=world_size, 
        backend_config=backend_config,
        feature_stats=None
    )
    
    # Get tokenizer
    if args.mode == "combined":
        tokenizer = train_loader.dataset.single_dataset.tokenizer
    else:
        tokenizer = train_loader.dataset.tokenizer
    
    # Load LoRA config
    tuned_cfg_path = os.path.join(ROOT_DIR, 'src', 'llm_config_tuned.json')
    if os.path.isfile(tuned_cfg_path):
        with open(tuned_cfg_path, 'r') as f:
            tuned_cfg = json.load(f)
        lora_params = tuned_cfg.get('lora_params', {})
    else:
        base_cfg_path = os.path.join(ROOT_DIR, 'src', 'llm_config.json')
        with open(base_cfg_path, 'r') as f:
            base_cfg = json.load(f)
        lora_params = base_cfg.get('lora_params', {})
    
    # Prepare training
    if device == 0:
        print("Preparing trainer...")
    optimizer, scheduler, scaler, trainer, start_epoch, initial_min_loss, initial_best_acc = prepare_training_with_resume(
        args=args,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        lora_params=lora_params,
        local_rank=device,
        world_size=world_size,
        backend_config=backend_config,
    )
    
    # Run training
    if device == 0:
        print(f"Starting training for {args.num_epochs} epochs...")
    
    fit_res = trainer.fit(
        num_epochs=args.num_epochs,
        device=device,
        early_stopping=args.num_epochs + 1,  # No early stopping for short ablation
        best='loss',
        start_epoch=start_epoch,
        initial_min_loss=initial_min_loss,
        initial_best_acc=initial_best_acc
    )
    
    # Save fit results
    if device == 0:
        output_filename = os.path.join(exp_dir, 'fit_res.json')
        with open(output_filename, "w") as f:
            json.dump(fit_res, f, indent=2)
        print(f"Saved training results to {output_filename}")
    
    # Gather summary
    summary = {
        'config': asdict(cfg),
        'exp_name': args.exp_name,
        'output_dir': exp_dir,
        'fit_results': fit_res,
        'num_spectral_features': cfg.num_spectral_features,
    }
    
    # Clean up
    del model, trainer, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return summary


def gather_results(all_results: List[Dict[str, Any]], output_dir: str) -> pd.DataFrame:
    """
    Gather and organize results from all experiments.
    
    Parameters
    ----------
    all_results : list
        List of result dictionaries from all experiments
    output_dir : str
        Directory to save results
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with organized results
    """
    rows = []
    
    for result in all_results:
        if 'error' in result:
            continue
            
        num_tokens = result['num_spectral_features']
        fit_res = result.get('fit_results', {})
        
        # Extract losses
        train_losses = fit_res.get('train_loss', [])
        val_losses = fit_res.get('val_loss', [])
        
        # Get final and best losses
        final_train_loss = train_losses[-1] if train_losses else None
        final_val_loss = val_losses[-1] if val_losses else None
        best_val_loss = min(val_losses) if val_losses else None
        
        row = {
            'num_spectral_features': num_tokens,
            'exp_name': result['exp_name'],
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'best_val_loss': best_val_loss,
            'avg_train_loss': np.mean(train_losses) if train_losses else None,
            'avg_val_loss': np.mean(val_losses) if val_losses else None,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows).sort_values('num_spectral_features')
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'ablation_results_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary to {csv_path}")
    
    return df


def create_comparison_plots(all_results: List[Dict[str, Any]], output_dir: str):
    """
    Create visualization plots comparing experiments.
    
    Parameters
    ----------
    all_results : list
        List of result dictionaries from all experiments
    output_dir : str
        Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Ablation Study: Impact of num_spectral_features', fontsize=16, fontweight='bold')
    
    # Prepare data
    data_by_tokens = {}
    for result in all_results:
        if 'error' in result:
            continue
        num_tokens = result['num_spectral_features']
        fit_res = result.get('fit_results', {})
        data_by_tokens[num_tokens] = fit_res
    
    # Sort by number of tokens
    sorted_tokens = sorted(data_by_tokens.keys())
    
    # Plot 1: Training Loss Curves
    ax = axes[0, 0]
    for num_tokens in sorted_tokens:
        fit_res = data_by_tokens[num_tokens]
        train_losses = fit_res.get('train_loss', [])
        if train_losses:
            epochs = range(1, len(train_losses) + 1)
            ax.plot(epochs, train_losses, marker='o', label=f'{num_tokens} tokens', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Loss over Epochs', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss Curves
    ax = axes[0, 1]
    for num_tokens in sorted_tokens:
        fit_res = data_by_tokens[num_tokens]
        val_losses = fit_res.get('val_loss', [])
        if val_losses:
            epochs = range(1, len(val_losses) + 1)
            ax.plot(epochs, val_losses, marker='s', label=f'{num_tokens} tokens', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Validation Loss over Epochs', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Final Loss Comparison (Bar Plot)
    ax = axes[1, 0]
    final_train = []
    final_val = []
    labels = []
    for num_tokens in sorted_tokens:
        fit_res = data_by_tokens[num_tokens]
        train_losses = fit_res.get('train_loss', [])
        val_losses = fit_res.get('val_loss', [])
        final_train.append(train_losses[-1] if train_losses else None)
        final_val.append(val_losses[-1] if val_losses else None)
        labels.append(str(num_tokens))
    
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, final_train, width, label='Train Loss', alpha=0.8)
    ax.bar(x + width/2, final_val, width, label='Val Loss', alpha=0.8)
    ax.set_xlabel('Number of Spectral Tokens', fontsize=12)
    ax.set_ylabel('Final Loss', fontsize=12)
    ax.set_title('Final Loss Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Best Validation Loss vs num_spectral_features
    ax = axes[1, 1]
    best_val_losses = []
    for num_tokens in sorted_tokens:
        fit_res = data_by_tokens[num_tokens]
        val_losses = fit_res.get('val_loss', [])
        best_val = min(val_losses) if val_losses else None
        best_val_losses.append(best_val)
    
    ax.plot(sorted_tokens, best_val_losses, marker='D', linewidth=2, markersize=8, color='crimson')
    ax.set_xlabel('Number of Spectral Tokens', fontsize=12)
    ax.set_ylabel('Best Validation Loss', fontsize=12)
    ax.set_title('Best Val Loss vs. num_spectral_features', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Highlight the best configuration
    if best_val_losses:
        best_idx = np.argmin(best_val_losses)
        best_tokens = sorted_tokens[best_idx]
        best_loss = best_val_losses[best_idx]
        ax.axvline(best_tokens, color='green', linestyle='--', alpha=0.5, label=f'Best: {best_tokens} tokens')
        ax.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'ablation_comparison_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plots to {plot_path}")
    plt.close()


def main():
    """Main ablation study execution."""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    root_out = os.path.join('/home/ilay.kamai/work/TalkingLatents/logs', f'ablation_spectral_tokens_{timestamp}')
    os.makedirs(root_out, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"ABLATION STUDY: num_spectral_features")
    print(f"Output directory: {root_out}")
    print(f"{'='*80}\n")
    
    # Initialize distributed training (if applicable)
    local_rank, world_size, _ = setup()
    device = local_rank if isinstance(local_rank, int) else 0
    
    # Base arguments
    base_args = make_base_args(output_dir=root_out)
    
    # Define ablation grid - test different numbers of spectral tokens
    token_counts = [4, 8, 16, 32]
    
    ablation_configs = [
        SpectralTokensAblationConfig(
            num_spectral_features=n,
            label=f"spectral_tokens_{n}"
        )
        for n in token_counts
    ]
    
    if device == 0:
        print(f"Running {len(ablation_configs)} experiments with token counts: {token_counts}\n")
    
    all_results: List[Dict[str, Any]] = []
    
    # Run experiments
    for i, cfg in enumerate(ablation_configs):
        if device == 0:
            print(f"\n[{i+1}/{len(ablation_configs)}] Starting experiment: {cfg.label}")
        
        try:
            result = run_single_experiment(cfg, base_args, device=device, world_size=world_size)
            all_results.append(result)
            
            # Save intermediate results
            if device == 0:
                partial_path = os.path.join(root_out, 'ablation_results_partial.json')
                with open(partial_path, 'w') as f:
                    json.dump(all_results, f, indent=2)
                print(f"✓ Saved intermediate results to {partial_path}")
                
        except Exception as e:
            if device == 0:
                print(f"✗ Experiment '{cfg.label}' failed with error: {e}")
            result = {
                'config': asdict(cfg),
                'error': str(e),
                'num_spectral_features': cfg.num_spectral_features,
            }
            all_results.append(result)
    
    # Only rank 0 performs analysis and plotting
    if device == 0:
        print(f"\n{'='*80}")
        print("All experiments completed. Analyzing results...")
        print(f"{'='*80}\n")
        
        # Save final results
        final_path = os.path.join(root_out, 'ablation_results_final.json')
        with open(final_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved final results to {final_path}")
        
        # Gather and summarize
        df = gather_results(all_results, root_out)
        print("\nResults Summary:")
        print(df.to_string(index=False))
        
        # Create comparison plots
        create_comparison_plots(all_results, root_out)
        
        # Print best configuration
        if not df.empty and 'best_val_loss' in df.columns:
            best_row = df.loc[df['best_val_loss'].idxmin()]
            print(f"\n{'='*80}")
            print("BEST CONFIGURATION:")
            print(f"  num_spectral_features = {int(best_row['num_spectral_features'])}")
            print(f"  Best Validation Loss = {best_row['best_val_loss']:.4f}")
            print(f"  Final Validation Loss = {best_row['final_val_loss']:.4f}")
            print(f"  Final Training Loss = {best_row['final_train_loss']:.4f}")
            print(f"{'='*80}\n")
        
        print(f"\nAll outputs saved to: {root_out}")


if __name__ == '__main__':
    main()
