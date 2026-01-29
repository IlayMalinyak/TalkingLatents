#!/usr/bin/env python3
"""
Baseline MLP Inference Script with Steering Support

Loads a trained MLP baseline model and performs inference on test data.
Supports steering experiments where concept vectors are added to features
to observe their effect on predictions.

Usage:
    # Basic inference
    python src/baseline_inference.py \
        --checkpoint_path logs/baseline_mlp/best_model.pth \
        --json_file data/dataset/stellar_descriptions_questions_short.json \
        --features_file logs/2025-07-29/features.npy \
        --output_dir logs/baseline_inference
    
    # Steering experiment
    python src/baseline_inference.py \
        --checkpoint_path logs/baseline_mlp/best_model.pth \
        --json_file data/dataset/stellar_descriptions_questions_short.json \
        --features_file logs/2025-07-29/features.npy \
        --steering_concept_file logs/concepts/concept_directions.pt \
        --steering_alphas "-5,-2,0,2,5" \
        --output_dir logs/baseline_steering
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from data.dataset_interpert import create_stellar_dataloaders


# Import MLP model and utilities from baseline_mlp
from src.baseline_mlp import (
    MLPRegressor,
    StellarFeatureDataset,
    split_dataset,
    DenormalizationSpec,
    build_target_denormalizers,
    extract_batch_data,
    normalize_batch,
)


PHYSICAL_BOUNDS = {
    'Teff': (3000.0, 7500.0),
    'logg': (0.0, 5.0),
    'FeH': (-3.0, 0.5),
}

DEFAULT_TARGETS = ['Teff', 'logg', 'FeH']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with trained MLP baseline model, with optional steering."
    )
    
    # Model checkpoint
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to trained MLP checkpoint (.pth file)'
    )
    
    # Data source (new pipeline)
    parser.add_argument(
        '--json_file',
        type=str,
        default=None,
        help='Path to JSON dataset (new pipeline)'
    )
    parser.add_argument(
        '--features_file',
        type=str,
        required=True,
        help='Path to features .npy file'
    )
    parser.add_argument(
        '--feature_stats_file',
        type=str,
        default=None,
        help='Path to normalization stats .npz file'
    )
    parser.add_argument(
        '--index_df_file',
        type=str,
        default=None,
        help='Path to obsid mapping CSV'
    )
    
    # Legacy pipeline support
    parser.add_argument(
        '--info_file',
        type=str,
        default=None,
        help='Path to info CSV (legacy pipeline)'
    )
    
    # Output
    parser.add_argument(
        '--output_dir',
        type=str,
        default='baseline_inference_output',
        help='Directory for output files'
    )
    parser.add_argument(
        '--output_filename',
        type=str,
        default='predictions.json',
        help='Filename for predictions JSON'
    )
    
    # Steering
    parser.add_argument(
        '--steering_concept_file',
        type=str,
        default=None,
        help='Path to concept directions (.pt file)'
    )
    parser.add_argument(
        '--steering_alphas',
        type=str,
        default='0',
        help='Comma-separated alphas (e.g., "-5,-2,0,2,5")'
    )
    
    # Inference settings
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Limit number of samples to process'
    )
    parser.add_argument(
        '--targets',
        nargs='+',
        default=DEFAULT_TARGETS,
        help='Target parameters to predict'
    )
    parser.add_argument(
        '--num_spectral_features',
        type=int,
        default=8,
        help='Number of spectral feature tokens (for new pipeline)'
    )
    
    # Other
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, default: auto-detect)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(
    checkpoint_path: str,
    input_dim: int,
    output_dim: int,
    device: torch.device
) -> Tuple[nn.Module, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load MLP model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model hyperparameters from checkpoint if available
    # Otherwise use defaults
    hidden_dim = checkpoint.get('hidden_dim', 512)
    hidden_layers = checkpoint.get('hidden_layers', 3)
    dropout = checkpoint.get('dropout', 0.1)
    
    # Create model
    model = MLPRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        dropout=dropout
    )
    
    # Load state dict
    state_dict = checkpoint.get('model_state', checkpoint)
    model.load_state_dict(state_dict)
    
    # Load normalization stats if available
    feature_mean = checkpoint.get('feature_mean')
    feature_std = checkpoint.get('feature_std')
    
    if feature_mean is not None:
        feature_mean = feature_mean.to(device)
    if feature_std is not None:
        feature_std = feature_std.to(device)
    
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded (input_dim={input_dim}, output_dim={output_dim})")
    if feature_mean is not None:
        print("✓ Loaded feature normalization stats from checkpoint")
    
    return model, feature_mean, feature_std


def denormalize_predictions(
    predictions: np.ndarray,
    target_names: List[str],
    bounds: Dict[str, Tuple[float, float]] = PHYSICAL_BOUNDS
) -> np.ndarray:
    """Denormalize predictions from [0,1] to physical units."""
    denorm_preds = predictions.copy()
    for i, target in enumerate(target_names):
        if target in bounds:
            low, high = bounds[target]
            denorm_preds[:, i] = denorm_preds[:, i] * (high - low) + low
    return denorm_preds


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    feature_mean: Optional[torch.Tensor],
    feature_std: Optional[torch.Tensor],
    use_new_pipeline: bool,
    steering_vector: Optional[torch.Tensor] = None,
    steering_alpha: float = 0.0,
    sigma: Optional[torch.Tensor] = None,
    max_samples: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any]]:
    """
    Run inference on dataloader.
    
    Returns:
        predictions: List of prediction arrays
        targets: List of target arrays
        obsids: List of obsids
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_obsids = []
    
    total_processed = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Running inference"):
            # Extract features and targets
            features, targets = extract_batch_data(batch, use_new_pipeline)
            
            # Get obsids if available
            if use_new_pipeline:
                obsids = batch.get('obsids', [None] * features.shape[0])
            else:
                obsids = [None] * features.shape[0]
            
            # Move to device first
            features = features.to(device)
            targets = targets.to(device)
            
            # Apply steering if enabled (before normalization)
            if steering_vector is not None and steering_alpha != 0.0:
                # Steering in normalized space: h' = h + alpha * (v / sigma)
                # Steering in normalized space: h' = h + alpha * (v / sigma)
                if sigma is not None:
                    # Use epsilon for numerical stability
                    EPS = 1e-6
                    safe_sigma = torch.where(sigma < EPS, torch.tensor(1.0, device=sigma.device), sigma)
                    if (sigma < EPS).any():
                        print(f"WARNING: sigma contains {torch.sum(sigma < EPS)} values close to zero! Using 1.0 for these.")
                    
                    v_norm = steering_vector / safe_sigma
                else:
                    v_norm = steering_vector
                
                # Debug stats
                if torch.isnan(v_norm).any() or torch.isinf(v_norm).any():
                     print(f"WARNING: v_norm contains non-finite values!")
                     print(f"  Max val: {v_norm.max()}, Min val: {v_norm.min()}")
                features = features + steering_alpha * v_norm
                
                # Debug: Check for NaN after steering
                if torch.isnan(features).any():
                    print(f"WARNING: NaN detected in features after steering!")
                    print(f"  steering_alpha: {steering_alpha}")
                    print(f"  v_norm contains NaN: {torch.isnan(v_norm).any()}")
                    print(f"  sigma contains NaN: {torch.isnan(sigma).any() if sigma is not None else 'N/A'}")
            
            # Normalize features ONLY for legacy pipeline (new pipeline features are already normalized)
            if not use_new_pipeline:
                 features = normalize_batch(features, feature_mean, feature_std)
            
            # Debug: Check for NaN after normalization
            if torch.isnan(features).any():
                print(f"WARNING: NaN detected in features after normalization!")
                print(f"  feature_mean contains NaN: {torch.isnan(feature_mean).any() if feature_mean is not None else 'N/A'}")
                print(f"  feature_std contains NaN: {torch.isnan(feature_std).any() if feature_std is not None else 'N/A'}")
            
            # Run model
            predictions = model(features)
            
            # Debug: Check for NaN in predictions
            if torch.isnan(predictions).any():
                print(f"WARNING: NaN detected in model predictions!")
                print(f"  Input features contain NaN: {torch.isnan(features).any()}")
            
            # Store results
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_obsids.extend(obsids)
            
            total_processed += features.shape[0]
            if max_samples is not None and total_processed >= max_samples:
                break
    
    return all_preds, all_targets, all_obsids


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_names: List[str]
) -> Dict[str, float]:
    """Calculate metrics for predictions."""
    metrics = {}
    
    # Overall metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    medae = np.median(np.abs(predictions - targets))
    
    metrics['mse'] = float(mse)
    metrics['mae'] = float(mae)
    metrics['medae'] = float(medae)
    metrics['rmse'] = float(np.sqrt(mse))
    
    # Per-parameter metrics
    for i, target in enumerate(target_names):
        param_mse = np.mean((predictions[:, i] - targets[:, i]) ** 2)
        param_mae = np.mean(np.abs(predictions[:, i] - targets[:, i]))
        param_medae = np.median(np.abs(predictions[:, i] - targets[:, i]))
        
        metrics[f'{target}_mse'] = float(param_mse)
        metrics[f'{target}_mae'] = float(param_mae)
        metrics[f'{target}_medae'] = float(param_medae)
        metrics[f'{target}_rmse'] = float(np.sqrt(param_mse))
    
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine pipeline type
    use_new_pipeline = args.json_file is not None
    
    # Load data
    if use_new_pipeline:
        print("Using NEW data pipeline...")
        
        # Load features
        features_array = np.load(args.features_file)
        print(f"Loaded features: {features_array.shape}")
        
        # Load feature stats
        feature_stats = None
        if args.feature_stats_file and os.path.exists(args.feature_stats_file):
            print(f"Loading feature stats from {args.feature_stats_file}")
            stats_data = np.load(args.feature_stats_file)
            feature_stats = {'mean': stats_data['mean'], 'std': stats_data['std']}
        
        # Load index_df
        index_df = None
        if args.index_df_file and os.path.exists(args.index_df_file):
            print(f"Loading index_df from {args.index_df_file}")
            index_df = pd.read_csv(args.index_df_file)
        
        # Create dataloaders
        _, _, test_loader = create_stellar_dataloaders(
            json_file=args.json_file,
            features_array=features_array,
            batch_size=args.batch_size,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=args.seed,
            num_workers=0,
            num_spectral_features=args.num_spectral_features,
            cache_dir=os.path.join(args.output_dir, 'cache'),
            feature_stats=feature_stats,
            index_df=index_df,
            device=device,
        )
        
        # Get dimensions from first batch
        sample_batch = next(iter(test_loader))
        input_dim = sample_batch['features'].shape[1]
        output_dim = len(args.targets)
        
        # Get sigma for steering
        sigma = None
        if feature_stats is not None:
            sigma = torch.from_numpy(feature_stats['std']).to(device)
        
    else:
        print("Using LEGACY data pipeline...")
        
        # Load dataset
        info_path = args.info_file or os.path.join(
            os.path.dirname(args.features_file), "info.csv"
        )
        
        dataset = StellarFeatureDataset(
            features_path=args.features_file,
            info_path=info_path,
            target_columns=args.targets,
            max_samples=args.max_samples,
        )
        
        # Split dataset
        _, _, test_set = split_dataset(
            dataset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=args.seed,
        )
        
        # Create dataloader
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == 'cuda',
        )
        
        input_dim = dataset.input_dim
        output_dim = dataset.output_dim
        sigma = None  # Will compute from data if needed
    
    print(f"Test set size: {len(test_loader.dataset)}")
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    
    # Load model
    model, checkpoint_mean, checkpoint_std = load_model(
        args.checkpoint_path,
        input_dim,
        output_dim,
        device
    )
    
    # Use checkpoint stats if available, otherwise None
    feature_mean = checkpoint_mean
    feature_std = checkpoint_std
    
    # Load steering concepts if provided
    steering_enabled = False
    concepts = {}
    alphas = [0.0]
    
    if args.steering_concept_file:
        print(f"Loading steering concepts from {args.steering_concept_file}")
        concepts = torch.load(args.steering_concept_file, map_location=device)
        steering_enabled = True
        alphas = [float(x) for x in args.steering_alphas.split(',')]
        print(f"Steering enabled with {len(concepts)} concepts and {len(alphas)} alphas")
        print(f"Concepts: {list(concepts.keys())}")
        print(f"Alphas: {alphas}")
    else:
        concepts = {'none': None}
        alphas = [0.0]
    
    # Compute sigma from test data if not available and steering is enabled
    if steering_enabled and sigma is None:
        print("Computing sigma from test data for steering...")
        all_features = []
        for batch in test_loader:
            features, _ = extract_batch_data(batch, use_new_pipeline)
            all_features.append(features.numpy())
        all_features = np.concatenate(all_features, axis=0)
        sigma = torch.from_numpy(np.std(all_features, axis=0).astype(np.float32)).to(device)
        print(f"✓ Computed sigma: shape={sigma.shape}")
    
    # Run inference for all steering configurations
    all_results = []
    
    for concept_name, concept_vec in concepts.items():
        if concept_vec is not None:
            concept_vec = concept_vec.to(device)
            print(f"\nConcept '{concept_name}' shape: {concept_vec.shape}")
            print(f"Expected feature dim: {input_dim}")
            print(f"Sigma shape: {sigma.shape if sigma is not None else 'None'}")
            
            # Validate dimensions
            if concept_vec.shape[0] != input_dim:
                print(f"WARNING: Concept vector dimension {concept_vec.shape[0]} doesn't match input_dim {input_dim}")
                print("Skipping this concept...")
                continue
        
        for alpha in alphas:
            print(f"\n{'='*60}")
            print(f"Concept: {concept_name}, Alpha: {alpha}")
            print(f"{'='*60}")
            
            # Run inference
            preds_list, targets_list, obsids_list = run_inference(
                model=model,
                loader=test_loader,
                device=device,
                feature_mean=feature_mean,
                feature_std=feature_std,
                use_new_pipeline=use_new_pipeline,
                steering_vector=concept_vec,
                steering_alpha=alpha,
                sigma=sigma,
                max_samples=args.max_samples,
            )
            
            # Concatenate results
            predictions = np.concatenate(preds_list, axis=0)
            targets = np.concatenate(targets_list, axis=0)
            
            # Denormalize
            predictions_phys = denormalize_predictions(predictions, args.targets)
            targets_phys = denormalize_predictions(targets, args.targets)
            
            # Calculate metrics
            metrics_norm = calculate_metrics(predictions, targets, args.targets)
            metrics_phys = calculate_metrics(predictions_phys, targets_phys, args.targets)
            
            print(f"\nMetrics (normalized):")
            print(f"  MSE: {metrics_norm['mse']:.6f}, MAE: {metrics_norm['mae']:.6f}, MedAE: {metrics_norm['medae']:.6f}")
            for target in args.targets:
                print(f"  {target}: MAE={metrics_norm[f'{target}_mae']:.6f}, MedAE={metrics_norm[f'{target}_medae']:.6f}")
            
            print(f"\nMetrics (physical units):")
            print(f"  MSE: {metrics_phys['mse']:.2f}, MAE: {metrics_phys['mae']:.2f}, MedAE: {metrics_phys['medae']:.2f}")
            for target in args.targets:
                print(f"  {target}: MAE={metrics_phys[f'{target}_mae']:.2f}, MedAE={metrics_phys[f'{target}_medae']:.2f}")
            
            # Store results
            for i in range(len(predictions)):
                result = {
                    'obsid': str(obsids_list[i]) if obsids_list[i] is not None else None,
                    'steering_concept': concept_name,
                    'steering_alpha': float(alpha),
                    'predictions': {
                        target: float(predictions_phys[i, j])
                        for j, target in enumerate(args.targets)
                    },
                    'predictions_normalized': {
                        target: float(predictions[i, j])
                        for j, target in enumerate(args.targets)
                    },
                    'targets': {
                        target: float(targets_phys[i, j])
                        for j, target in enumerate(args.targets)
                    },
                    'targets_normalized': {
                        target: float(targets[i, j])
                        for j, target in enumerate(args.targets)
                    },
                    'errors': {
                        target: float(predictions_phys[i, j] - targets_phys[i, j])
                        for j, target in enumerate(args.targets)
                    },
                }
                all_results.append(result)
    
    # Save results
    output_path = os.path.join(args.output_dir, args.output_filename)
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'checkpoint_path': args.checkpoint_path,
            'total_samples': len(all_results) // (len(concepts) * len(alphas)),
            'concepts': list(concepts.keys()),
            'alphas': alphas,
            'targets': args.targets,
            'steering_enabled': steering_enabled,
        },
        'results': all_results,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Saved {len(all_results)} results to {output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
