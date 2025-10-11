#!/usr/bin/env python3
"""
Inference script for the multimodal stellar model.
Loads a trained model, runs predictions on test set, and saves results.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add root directory to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.simple_questions import (
    parse_args as base_parse_args,
    _load_llm_model,
    _load_spectra_model,
    get_model_path
)
from src.simple_questions_multitok import (
    create_datasets_and_loaders,
    build_model_multitok
)
from nn.llm_multi import MultimodalLlamaModelMultiTokens
from data.transforms import GeneralSpectrumPreprocessor, ToTensor, Compose
import torch.nn.functional as F


def parse_inference_args():
    """Parse command line arguments for inference"""
    parser = argparse.ArgumentParser(description='Inference script for stellar model')
    
    # Model and data arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the saved model checkpoint')
    parser.add_argument('--config_path', type=str, 
                        help='Path to training config file (auto-detected from checkpoint dir if not provided)')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save inference results')
    
    # Data arguments (can be overridden from config)
    parser.add_argument('--json_file', type=str, 
                        default='/data/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json',
                        help='Path to main dataset JSON file')
    parser.add_argument('--comparative_json_file', type=str,
                        default='/data/TalkingLatents/data/dataset/comparative_dataset.json',
                        help='Path to comparative dataset JSON file')
    parser.add_argument('--features_file', type=str,
                        default='/data/TalkingLatents/logs/2025-07-29/features.npy',
                        help='Path to pre-computed spectral features')
    
    # Inference settings
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on')
    parser.add_argument('--save_predictions', action='store_true', default=True,
                        help='Save predictions to JSON file')
    parser.add_argument('--plot_results', action='store_true', default=True,
                        help='Generate plots of true vs predicted values')
    
    return parser.parse_args()


def load_config_from_checkpoint_dir(checkpoint_path: str) -> Dict:
    """Load training config from checkpoint directory"""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, 'training_config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"No training config found at {config_path}")


def create_args_from_config(config: Dict, inference_args: argparse.Namespace) -> argparse.Namespace:
    """Create args namespace from training config, overriding with inference args"""
    
    # Create a base args object
    args = argparse.Namespace()
    
    # Set all config values as attributes
    for key, value in config.items():
        setattr(args, key, value)
    
    # Override with inference-specific arguments
    for key, value in vars(inference_args).items():
        if value is not None:
            setattr(args, key, value)
    
    # Ensure required attributes for inference
    if not hasattr(args, 'mode'):
        args.mode = 'combined'
    if not hasattr(args, 'num_workers'):
        args.num_workers = 4
    if not hasattr(args, 'train_ratio'):
        args.train_ratio = 0.8
    if not hasattr(args, 'val_ratio'):
        args.val_ratio = 0.1
    if not hasattr(args, 'test_ratio'):
        args.test_ratio = 0.1
    if not hasattr(args, 'random_seed'):
        args.random_seed = 42
    
    return args


def load_model(checkpoint_path: str, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    """Load the trained model from checkpoint"""
    
    print(f"Loading model from {checkpoint_path}")
    
    # Build the model architecture
    model = build_model_multitok(args, device, world_size=1)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel/DistributedDataParallel prefixes
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    # Load state dict
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    print("✓ Model loaded successfully")
    return model


def predict_on_testset(model: torch.nn.Module, test_loader, device: torch.device) -> Tuple[List, List, List]:
    """Run predictions on test set"""
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_stellar_predictions = []
    all_stellar_targets = []
    
    print("Running inference on test set...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move batch to device
            if isinstance(batch, dict):
                for key in ['input_ids', 'attention_mask', 'spectral_features']:
                    if key in batch and batch[key] is not None:
                        batch[key] = batch[key].to(device)
                
                # Handle stellar parameters if present
                if 'stellar_params' in batch and batch['stellar_params'] is not None:
                    if isinstance(batch['stellar_params'], dict):
                        stellar_targets = {k: v.to(device) for k, v in batch['stellar_params'].items()}
                    else:
                        stellar_targets = batch['stellar_params'].to(device)
                else:
                    stellar_targets = None
            
            # Forward pass
            try:
                outputs = model(
                    input_ids=batch.get('input_ids'),
                    attention_mask=batch.get('attention_mask'),
                    spectral_features=batch.get('spectral_features'),
                    labels=batch.get('labels')
                )
                
                # Extract predictions
                if hasattr(outputs, 'stellar_predictions') and outputs.stellar_predictions is not None:
                    stellar_preds = outputs.stellar_predictions
                    if isinstance(stellar_preds, dict):
                        all_stellar_predictions.append({k: v.cpu().numpy() for k, v in stellar_preds.items()})
                    else:
                        all_stellar_predictions.append(stellar_preds.cpu().numpy())
                
                if stellar_targets is not None:
                    if isinstance(stellar_targets, dict):
                        all_stellar_targets.append({k: v.cpu().numpy() for k, v in stellar_targets.items()})
                    else:
                        all_stellar_targets.append(stellar_targets.cpu().numpy())
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(test_loader)}")
    
    # Combine all predictions
    if all_stellar_predictions:
        if isinstance(all_stellar_predictions[0], dict):
            # Handle dict format (parameter-wise predictions)
            combined_stellar_preds = {}
            for param in all_stellar_predictions[0].keys():
                combined_stellar_preds[param] = np.concatenate([pred[param] for pred in all_stellar_predictions])
        else:
            # Handle tensor format
            combined_stellar_preds = np.concatenate(all_stellar_predictions)
    else:
        combined_stellar_preds = None
    
    if all_stellar_targets:
        if isinstance(all_stellar_targets[0], dict):
            # Handle dict format
            combined_stellar_targets = {}
            for param in all_stellar_targets[0].keys():
                combined_stellar_targets[param] = np.concatenate([target[param] for target in all_stellar_targets])
        else:
            # Handle tensor format
            combined_stellar_targets = np.concatenate(all_stellar_targets)
    else:
        combined_stellar_targets = None
    
    return all_predictions, all_targets, combined_stellar_preds, combined_stellar_targets


def save_predictions(predictions: Dict, targets: Dict, stellar_preds: Dict, stellar_targets: Dict, 
                    output_path: str):
    """Save predictions to JSON file"""
    
    results = {
        'stellar_predictions': {},
        'stellar_targets': {},
        'metadata': {
            'num_samples': len(list(stellar_preds.values())[0]) if stellar_preds else 0,
            'stellar_parameters': list(stellar_preds.keys()) if stellar_preds else []
        }
    }
    
    # Convert numpy arrays to lists for JSON serialization
    if stellar_preds:
        if isinstance(stellar_preds, dict):
            for param, values in stellar_preds.items():
                results['stellar_predictions'][param] = values.tolist()
        else:
            results['stellar_predictions']['values'] = stellar_preds.tolist()
    
    if stellar_targets:
        if isinstance(stellar_targets, dict):
            for param, values in stellar_targets.items():
                results['stellar_targets'][param] = values.tolist()
        else:
            results['stellar_targets']['values'] = stellar_targets.tolist()
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Predictions saved to {output_path}")


def plot_stellar_parameters(stellar_preds: Dict, stellar_targets: Dict, output_dir: str):
    """Plot true vs predicted stellar parameters"""
    
    if not stellar_preds or not stellar_targets:
        print("No stellar parameters to plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameter labels and units
    param_info = {
        'Teff': {'label': 'Effective Temperature (K)', 'range': (3000, 8000)},
        'logg': {'label': 'Surface Gravity (log g)', 'range': (0, 5)},
        'FeH': {'label': 'Metallicity [Fe/H]', 'range': (-3, 1)}
    }
    
    if isinstance(stellar_preds, dict) and isinstance(stellar_targets, dict):
        # Plot each parameter separately
        for param in stellar_preds.keys():
            if param not in stellar_targets:
                continue
                
            preds = stellar_preds[param]
            targets = stellar_targets[param]
            
            plt.figure(figsize=(8, 6))
            plt.scatter(targets, preds, alpha=0.6, s=20)
            
            # Add diagonal line for perfect prediction
            min_val = min(np.min(targets), np.min(preds))
            max_val = max(np.max(targets), np.max(preds))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            
            plt.xlabel(f'True {param_info.get(param, {}).get("label", param)}')
            plt.ylabel(f'Predicted {param_info.get(param, {}).get("label", param)}')
            plt.title(f'True vs Predicted {param}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Calculate metrics
            mae = np.mean(np.abs(preds - targets))
            rmse = np.sqrt(np.mean((preds - targets)**2))
            r_squared = np.corrcoef(targets, preds)[0, 1]**2
            
            # Add metrics to plot
            plt.text(0.05, 0.95, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r_squared:.3f}',
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{param}_true_vs_predicted.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Plot saved for {param}")
    
    # Create combined plot
    if isinstance(stellar_preds, dict) and len(stellar_preds) == 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, param in enumerate(['Teff', 'logg', 'FeH']):
            if param not in stellar_preds or param not in stellar_targets:
                continue
                
            preds = stellar_preds[param]
            targets = stellar_targets[param]
            
            axes[i].scatter(targets, preds, alpha=0.6, s=20)
            
            min_val = min(np.min(targets), np.min(preds))
            max_val = max(np.max(targets), np.max(preds))
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            axes[i].set_xlabel(f'True {param_info.get(param, {}).get("label", param)}')
            axes[i].set_ylabel(f'Predicted {param_info.get(param, {}).get("label", param)}')
            axes[i].set_title(f'{param}')
            axes[i].grid(True, alpha=0.3)
            
            # Calculate and display metrics
            mae = np.mean(np.abs(preds - targets))
            rmse = np.sqrt(np.mean((preds - targets)**2))
            r_squared = np.corrcoef(targets, preds)[0, 1]**2
            
            axes[i].text(0.05, 0.95, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r_squared:.3f}',
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'stellar_parameters_combined.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Combined plot saved")


def main():
    """Main inference function"""
    
    # Parse arguments
    inference_args = parse_inference_args()
    
    # Load training config
    if inference_args.config_path:
        with open(inference_args.config_path, 'r') as f:
            config = json.load(f)
    else:
        config = load_config_from_checkpoint_dir(inference_args.checkpoint_path)
    
    # Create full args from config and inference args
    args = create_args_from_config(config, inference_args)
    
    # Set up device
    device = torch.device(inference_args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(inference_args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(inference_args.checkpoint_path, args, device)
    
    # Create dataloaders (we only need test_loader)
    print("Creating test dataloader...")
    _, _, test_loader = create_datasets_and_loaders(args, device)
    
    # Run predictions
    _, _, stellar_preds, stellar_targets = predict_on_testset(model, test_loader, device)
    
    # Save predictions
    if inference_args.save_predictions and stellar_preds is not None:
        predictions_path = os.path.join(inference_args.output_dir, 'test_predictions.json')
        save_predictions({}, {}, stellar_preds, stellar_targets, predictions_path)
    
    # Generate plots
    if inference_args.plot_results and stellar_preds is not None:
        plots_dir = os.path.join(inference_args.output_dir, 'plots')
        plot_stellar_parameters(stellar_preds, stellar_targets, plots_dir)
    
    print("✓ Inference completed successfully!")


if __name__ == '__main__':
    main()