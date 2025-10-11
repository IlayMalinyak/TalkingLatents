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
from typing import Dict, List, Tuple, Optional, Union

# Add root directory to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.simple_questions import (
    parse_args as base_parse_args,
    _load_llm_model,
    _load_spectra_model,
    get_model_path,
    create_optimizer_and_scheduler,
    setup
)
from src.simple_questions_multitok import (
    create_datasets_and_loaders,
    build_model_multitok
)
from nn.llm_multi import MultimodalLlamaModelMultiTokens
from nn.train import LLMTrainer
from nn.optim import CQR
from data.transforms import GeneralSpectrumPreprocessor, ToTensor, Compose
import torch.nn.functional as F
import re


def unnormalize_stellar_params(normalized_params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Unnormalize stellar parameters using the same bounds as in dataset_mixed.py"""
    
    # Stellar parameter bounds from dataset_mixed.py
    BOUNDS = {
        'MAX_TEFF': 7500, 'MIN_TEFF': 3000,
        'MAX_LOGG': 5.0, 'MIN_LOGG': 0,
        'MAX_FE_H': 0.5, 'MIN_FE_H': -3
    }
    
    unnormalized = {}
    
    for param, values in normalized_params.items():
        if param == 'Teff':
            unnormalized[param] = values * (BOUNDS['MAX_TEFF'] - BOUNDS['MIN_TEFF']) + BOUNDS['MIN_TEFF']
        elif param == 'logg':
            unnormalized[param] = values * (BOUNDS['MAX_LOGG'] - BOUNDS['MIN_LOGG']) + BOUNDS['MIN_LOGG']
        elif param == 'FeH':
            unnormalized[param] = values * (BOUNDS['MAX_FE_H'] - BOUNDS['MIN_FE_H']) + BOUNDS['MIN_FE_H']
        else:
            # For unknown parameters, return as-is
            unnormalized[param] = values
    
    return unnormalized


def extract_stellar_params_from_text(text: str) -> Dict[str, float]:
    """Extract stellar parameters from generated text"""
    
    params = {}
    
    # Define patterns for different parameter formats (improved to better capture floats)
    patterns = {
        'Teff': [
            r'(?:T_?eff|effective temperature|temperature)[\s:=~]*(?:is\s+)?(?:approximately\s+)?(\d+\.?\d*)\s*K?',
            r'(\d+\.?\d*)\s*K',  # Simple pattern for temperature
            r'(\d+\.\d+)',  # Explicit decimal pattern
        ],
        'logg': [
            r'(?:log\s*g|logg|surface gravity)[\s:=~]*(?:is\s+)?(?:approximately\s+)?(\d+\.?\d*)',
            r'(?:log\s*g|logg)[\s:=~]*(\d+\.?\d*)',
            r'(\d+\.\d+)',  # Explicit decimal pattern for logg
        ],
        'FeH': [
            r'(?:\[Fe/H\]|FeH|metallicity|metal)[\s:=~]*(?:is\s+)?(?:approximately\s+)?([+-]?\d+\.?\d*)',
            r'\[Fe/H\][\s:=~]*([+-]?\d+\.?\d*)',
            r'([+-]?\d+\.\d+)',  # Explicit decimal pattern for FeH
        ]
    }
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    for param, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                try:
                    # Take the first valid match
                    value = float(matches[0])
                    params[param] = value
                    # Debug: print extracted values for first few samples
                    if len(params) <= 3:  # Only for first few extractions to avoid spam
                        print(f"  Extracted {param} = {value} (type: {type(value)}) from pattern '{pattern}' in text snippet: '{text_lower[:100]}...'")
                    break  # Stop after finding first valid match for this parameter
                except (ValueError, IndexError):
                    continue
    
    return params


def plot_stellar_vs_text_consistency(stellar_preds_quantiles: Dict[str, np.ndarray], 
                                   text_predictions: List[str], 
                                   output_dir: str):
    """Plot consistency between stellar parameter predictions and text-extracted predictions"""
    
    if not stellar_preds_quantiles or not text_predictions:
        print("No data available for consistency plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract median predictions from quantiles for consistency comparison
    stellar_medians = {}
    for param, quantiles in stellar_preds_quantiles.items():
        median_idx = quantiles.shape[1] // 2
        stellar_medians[param] = quantiles[:, median_idx]
    
    # Extract parameters from all text predictions
    text_extracted = {'Teff': [], 'logg': [], 'FeH': []}
    stellar_values = {'Teff': [], 'logg': [], 'FeH': []}
    
    for i, text in enumerate(text_predictions):
        text_params = extract_stellar_params_from_text(text)
        
        # Only include samples where we have both stellar and text predictions
        include_sample = True
        for param in ['Teff', 'logg', 'FeH']:
            if param not in text_params or i >= len(stellar_medians[param]):
                include_sample = False
                break
        
        if include_sample:
            for param in ['Teff', 'logg', 'FeH']:
                text_extracted[param].append(text_params[param])
                stellar_values[param].append(stellar_medians[param][i])
    
    # Convert to numpy arrays
    for param in ['Teff', 'logg', 'FeH']:
        text_extracted[param] = np.array(text_extracted[param])
        stellar_values[param] = np.array(stellar_values[param])
    
    # Parameter info for labeling
    param_info = {
        'Teff': {'label': 'Effective Temperature (K)', 'range': (3000, 8000)},
        'logg': {'label': 'Surface Gravity (log g)', 'range': (0, 5)},
        'FeH': {'label': 'Metallicity [Fe/H]', 'range': (-3, 1)}
    }
    
    # Create individual plots
    for param in ['Teff', 'logg', 'FeH']:
        if len(text_extracted[param]) == 0:
            print(f"Warning: No extracted {param} values from text, skipping consistency plot")
            continue
            
        stellar_vals = stellar_values[param]
        text_vals = text_extracted[param]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(stellar_vals, text_vals, alpha=0.6, s=20)
        
        # Add diagonal line for perfect consistency
        min_val = min(np.min(stellar_vals), np.min(text_vals))
        max_val = max(np.max(stellar_vals), np.max(text_vals))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Consistency')
        
        plt.xlabel(f'Stellar Prediction - {param_info[param]["label"]}')
        plt.ylabel(f'Text Extracted - {param_info[param]["label"]}')
        plt.title(f'Stellar vs Text Prediction Consistency - {param}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculate consistency metrics
        mae = np.mean(np.abs(stellar_vals - text_vals))
        rmse = np.sqrt(np.mean((stellar_vals - text_vals)**2))
        r_squared = np.corrcoef(stellar_vals, text_vals)[0, 1]**2 if len(stellar_vals) > 1 else 0
        
        # Add metrics to plot
        plt.text(0.05, 0.95, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r_squared:.3f}\nN: {len(stellar_vals)}',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{param}_consistency.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Consistency plot saved for {param}")
    
    # Create combined consistency plot
    if all(len(text_extracted[param]) > 0 for param in ['Teff', 'logg', 'FeH']):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, param in enumerate(['Teff', 'logg', 'FeH']):
            stellar_vals = stellar_values[param]
            text_vals = text_extracted[param]
            
            axes[i].scatter(stellar_vals, text_vals, alpha=0.6, s=20)
            
            min_val = min(np.min(stellar_vals), np.min(text_vals))
            max_val = max(np.max(stellar_vals), np.max(text_vals))
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            axes[i].set_xlabel(f'Stellar - {param_info[param]["label"]}')
            axes[i].set_ylabel(f'Text - {param_info[param]["label"]}')
            axes[i].set_title(f'{param} Consistency')
            axes[i].grid(True, alpha=0.3)
            
            # Calculate and display metrics
            mae = np.mean(np.abs(stellar_vals - text_vals))
            rmse = np.sqrt(np.mean((stellar_vals - text_vals)**2))
            r_squared = np.corrcoef(stellar_vals, text_vals)[0, 1]**2 if len(stellar_vals) > 1 else 0
            
            axes[i].text(0.05, 0.95, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r_squared:.3f}\nN: {len(stellar_vals)}',
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'stellar_text_consistency_combined.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Combined consistency plot saved")


def llm_predict_stellar_bulk(trainer, data_loader, device, quantiles, max_iter=np.inf):
    """
    Bulk prediction method for LLMTrainer to extract stellar parameter predictions
    following the pattern from APOGEE script and MaskedRegressorTrainer
    """
    trainer.model.eval()
    
    # Initialize arrays similar to MaskedRegressorTrainer.predict
    num_params = 3  # Teff, logg, FeH
    num_quantiles = len(quantiles)
    
    preds = np.zeros((0, num_params, num_quantiles))  # [samples, params, quantiles]
    targets = np.zeros((0, num_params))              # [samples, params]
    obsids = []  # Track obsids for verification
    
    print(f"Running bulk stellar prediction...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Move batch to device
            if isinstance(batch, dict):
                for key in batch:
                    if key in batch and batch[key] is not None and torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(device)
            
            try:
                # Get model outputs
                outputs = trainer.get_logits(batch, device, val=True)
                
                # Extract stellar predictions
                stellar_preds_tensor = None
                if isinstance(outputs, dict) and 'stellar_predictions' in outputs:
                    stellar_preds_tensor = outputs['stellar_predictions']
                elif hasattr(outputs, 'stellar_predictions'):
                    stellar_preds_tensor = outputs.stellar_predictions
                
                if stellar_preds_tensor is not None:
                    batch_size = stellar_preds_tensor.shape[0]
                    
                    # Reshape to [batch_size, num_params, num_quantiles] 
                    preds_reshaped = stellar_preds_tensor.view(batch_size, num_params, num_quantiles)
                    
                    # Append to predictions array
                    preds = np.concatenate([preds, preds_reshaped.cpu().numpy()], axis=0)
                    
                    # Extract obsids for this batch
                    batch_obsids = batch.get('obsids', [])
                    obsids.extend(batch_obsids)
                
                # Extract targets following the same logic as training
                batch_targets = np.full((batch_size, num_params), np.nan)
                
                # Single-star mode
                if 'y_numeric' in batch and batch['y_numeric'] is not None:
                    gt_params = batch['y_numeric']
                    gt_mask = batch['y_numeric_present']
                    
                    if gt_mask.any():
                        valid_indices = gt_mask.cpu().numpy()
                        valid_gt = gt_params[gt_mask].cpu().numpy()
                        batch_targets[valid_indices] = valid_gt
                
                # Two-star mode (use star A)
                elif 'y_numeric_a' in batch and batch['y_numeric_a'] is not None:
                    gt_params_a = batch['y_numeric_a']
                    gt_mask_a = batch['y_numeric_a_present']
                    
                    if gt_mask_a.any():
                        valid_indices_a = gt_mask_a.cpu().numpy()
                        valid_gt_a = gt_params_a[gt_mask_a].cpu().numpy()
                        batch_targets[valid_indices_a] = valid_gt_a
                
                targets = np.concatenate([targets, batch_targets], axis=0)
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(data_loader)}")
                
            if batch_idx > max_iter:
                break
    
    print(f"✓ Bulk prediction completed: preds shape {preds.shape}, targets shape {targets.shape}, obsids: {len(obsids)}")
    return preds, targets, obsids


def llm_predict_with_text_generation(trainer, data_loader, device, max_iter=np.inf, 
                                   generate_text=True, max_new_tokens=128, args=None):
    """
    Enhanced prediction that collects both stellar predictions and generated text
    """
    trainer.model.eval()
    
    # Explicitly load tokenizer using get_model_path
    tokenizer = None
    if generate_text and args is not None:
        try:
            from llama3.llama.tokenizer import Tokenizer
            _, tokenizer_path = get_model_path(args)
            tokenizer = Tokenizer(model_path=tokenizer_path)
            print(f"✓ Loaded tokenizer from {tokenizer_path}")
        except Exception as e:
            print(f"Warning: Failed to load tokenizer: {e}")
            generate_text = False
    elif generate_text:
        print("Warning: No args provided for tokenizer loading")
        generate_text = False
    
    # Collect predictions
    all_text_predictions = []
    all_text_targets = []
    text_obsids = []  # Track obsids for verification
    
    print(f"Running predictions with text generation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Move batch to device
            if isinstance(batch, dict):
                for key in batch:
                    if key in batch and batch[key] is not None and torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(device)
            
            batch_size = batch['input_ids'].shape[0] if 'input_ids' in batch else 1
            
            # Extract obsids for this batch
            batch_obsids = batch.get('obsids', [])
            
            # Generate text for each sample in batch
            if generate_text:
                for sample_idx in range(batch_size):
                    try:
                        # Get the model (handle DataParallel/DistributedDataParallel)
                        model = trainer.model
                        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                            model = model.module
                        
                        # Generate response for this sample
                        generated_text, input_text, target_text, _ = model.generate_response_from_batch(
                            batch_data=batch,
                            batch_idx=sample_idx,
                            tokenizer=tokenizer,
                            max_new_tokens=max_new_tokens,
                            temperature=0.2,
                            top_p=0.8
                        )
                        all_text_predictions.append(generated_text)
                        all_text_targets.append(target_text)
                        
                        # Add corresponding obsid
                        if sample_idx < len(batch_obsids):
                            text_obsids.append(batch_obsids[sample_idx])
                        else:
                            text_obsids.append(None)
                        
                    except Exception as e:
                        print(f"Error generating text for sample {sample_idx} in batch {batch_idx}: {e}")
                        all_text_predictions.append("")
                        all_text_targets.append("")
                        
                        # Add corresponding obsid even for failed cases
                        if sample_idx < len(batch_obsids):
                            text_obsids.append(batch_obsids[sample_idx])
                        else:
                            text_obsids.append(None)
            
            if batch_idx % 1 == 0:
                print(f"Processed batch {batch_idx}/{len(data_loader)}")
                
            if batch_idx > max_iter:
                break
    
    print(f"✓ Text generation completed: {len(all_text_predictions)} samples, obsids: {len(text_obsids)}")
    return all_text_predictions, all_text_targets, text_obsids


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
    parser.add_argument('--llm_root', type=str, default=os.environ.get('LLM_ROOT', '/data/.llama'),
                       help='Root directory containing LLaMA models (or set env LLM_ROOT)')
    parser.add_argument('--llm_model', type=str, default='Llama3.1-8B',)
    
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
    
    # Start with inference_args as the base
    args = argparse.Namespace(**vars(inference_args))
    
    # Update with all config values (config takes precedence for any overlaps)
    for key, value in config.items():
        setattr(args, key, value)
    
    return args


def load_model(checkpoint_path: str, args: argparse.Namespace, device: Union[int, torch.device]) -> torch.nn.Module:
    """Load the trained model from checkpoint"""
    
    print(f"Loading model from {checkpoint_path}")

    if not hasattr(args, 'llm_precision'):
        args.llm_precision = 'fp16'
    if not hasattr(args, 'gradient_checkpointing'):
        args.gradient_checkpointing = False
    
    # Convert device to torch.device if it's an int (local rank)
    if isinstance(device, int):
        device = torch.device(f'cuda:{device}')
    
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




def save_predictions(text_predictions: List[str], text_targets: List[str], stellar_preds_quantiles: Dict, stellar_targets: Dict, 
                    output_path: str):
    """Save predictions to JSON file"""
    
    # Determine number of samples
    num_samples = 0
    if text_predictions:
        num_samples = len(text_predictions)
    elif stellar_preds_quantiles:
        num_samples = len(list(stellar_preds_quantiles.values())[0])
    
    results = {
        'text_predictions': text_predictions if text_predictions else [],
        'text_targets': text_targets if text_targets else [],
        'stellar_predictions_quantiles': {},
        'stellar_predictions_median': {},
        'stellar_targets': {},
        'metadata': {
            'num_samples': num_samples,
            'stellar_parameters': list(stellar_preds_quantiles.keys()) if stellar_preds_quantiles else [],
            'has_text_predictions': bool(text_predictions),
            'has_stellar_predictions': bool(stellar_preds_quantiles),
            'num_quantiles': stellar_preds_quantiles[list(stellar_preds_quantiles.keys())[0]].shape[1] if stellar_preds_quantiles else 0
        }
    }
    
    # Convert numpy arrays to lists for JSON serialization
    if stellar_preds_quantiles:
        for param, values in stellar_preds_quantiles.items():
            # Save all quantiles: [num_samples, num_quantiles]
            results['stellar_predictions_quantiles'][param] = values.tolist()
            
            # Also save median for convenience: [num_samples]
            median_idx = values.shape[1] // 2
            results['stellar_predictions_median'][param] = values[:, median_idx].tolist()
    
    if stellar_targets:
        for param, values in stellar_targets.items():
            results['stellar_targets'][param] = values.tolist()
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Predictions saved to {output_path}")
    if text_predictions:
        print(f"  - {len(text_predictions)} text predictions and targets")
    if stellar_preds_quantiles:
        num_quantiles = stellar_preds_quantiles[list(stellar_preds_quantiles.keys())[0]].shape[1]
        print(f"  - {num_samples} stellar parameter predictions with {num_quantiles} quantiles each")


def plot_stellar_parameters(stellar_preds_quantiles: Dict, stellar_targets: Dict, output_dir: str):
    """Plot true vs predicted stellar parameters with confidence intervals"""
    
    if not stellar_preds_quantiles or not stellar_targets:
        print("No stellar parameters to plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameter labels and units
    param_info = {
        'Teff': {'label': 'Effective Temperature (K)', 'range': (3000, 8000)},
        'logg': {'label': 'Surface Gravity (log g)', 'range': (0, 5)},
        'FeH': {'label': 'Metallicity [Fe/H]', 'range': (-3, 1)}
    }
    
    if isinstance(stellar_preds_quantiles, dict) and isinstance(stellar_targets, dict):
        # Plot each parameter separately
        for param in stellar_preds_quantiles.keys():
            if param not in stellar_targets:
                continue
                
            preds_quantiles = stellar_preds_quantiles[param]  # [num_samples, num_quantiles]
            targets = stellar_targets[param]  # [num_samples]
            
            # Extract median and confidence intervals
            median_idx = preds_quantiles.shape[1] // 2
            median_preds = preds_quantiles[:, median_idx]
            
            # Use 10th and 90th percentiles for confidence interval (assuming quantiles are evenly spaced)
            num_quantiles = preds_quantiles.shape[1]
            lower_idx = max(0, int(0.1 * num_quantiles))
            upper_idx = min(num_quantiles - 1, int(0.9 * num_quantiles))
            lower_preds = preds_quantiles[:, lower_idx]
            upper_preds = preds_quantiles[:, upper_idx]
            
            # Filter out NaN values for plotting and metrics
            valid_mask = ~(np.isnan(targets) | np.isnan(median_preds))
            if not valid_mask.any():
                print(f"Warning: No valid data for parameter {param}, skipping plot")
                continue
                
            valid_targets = targets[valid_mask]
            valid_median_preds = median_preds[valid_mask]
            valid_lower_preds = lower_preds[valid_mask]
            valid_upper_preds = upper_preds[valid_mask]
            
            plt.figure(figsize=(10, 8))
            
            # Sort by targets for better confidence interval visualization
            sort_idx = np.argsort(valid_targets)
            sorted_targets = valid_targets[sort_idx]
            sorted_median = valid_median_preds[sort_idx]
            sorted_lower = valid_lower_preds[sort_idx]
            sorted_upper = valid_upper_preds[sort_idx]
            
            # Plot confidence interval
            plt.fill_between(sorted_targets, sorted_lower, sorted_upper, alpha=0.3, 
                           label=f'1-sigma Confidence Interval', color='lightblue')
            
            # Plot median predictions
            plt.scatter(valid_targets, valid_median_preds, alpha=0.7, s=25, color='blue', label='Median Prediction')
            
            # Add diagonal line for perfect prediction
            min_val = min(np.min(valid_targets), np.min(valid_median_preds))
            max_val = max(np.max(valid_targets), np.max(valid_median_preds))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
            
            plt.xlabel(f'True {param_info.get(param, {}).get("label", param)}')
            plt.ylabel(f'Predicted {param_info.get(param, {}).get("label", param)}')
            plt.title(f'True vs Predicted {param} (CQR Calibrated)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Calculate metrics on valid data only
            mae = np.mean(np.abs(valid_median_preds - valid_targets))
            rmse = np.sqrt(np.mean((valid_median_preds - valid_targets)**2))
            r_squared = np.corrcoef(valid_targets, valid_median_preds)[0, 1]**2
            
            # Calculate coverage (what fraction of targets fall within confidence interval)
            coverage = np.mean((valid_targets >= valid_lower_preds) & (valid_targets <= valid_upper_preds))
            
            # Add metrics to plot
            plt.text(0.05, 0.95, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r_squared:.3f}\nCoverage: {coverage:.1%}\nN: {len(valid_targets)}',
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{param}_true_vs_predicted_cqr_calibrated.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Plot with CQR calibrated confidence intervals saved for {param}")
    
    # Create combined plot with confidence intervals
    if isinstance(stellar_preds_quantiles, dict) and len(stellar_preds_quantiles) == 3:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for i, param in enumerate(['Teff', 'logg', 'FeH']):
            if param not in stellar_preds_quantiles or param not in stellar_targets:
                continue
                
            preds_quantiles = stellar_preds_quantiles[param]
            targets = stellar_targets[param]
            
            # Extract median and confidence intervals
            median_idx = preds_quantiles.shape[1] // 2
            median_preds = preds_quantiles[:, median_idx]
            
            num_quantiles = preds_quantiles.shape[1]
            lower_idx = max(0, int(0.1 * num_quantiles))
            upper_idx = min(num_quantiles - 1, int(0.9 * num_quantiles))
            lower_preds = preds_quantiles[:, lower_idx]
            upper_preds = preds_quantiles[:, upper_idx]
            
            # Filter out NaN values
            valid_mask = ~(np.isnan(targets) | np.isnan(median_preds))
            if not valid_mask.any():
                print(f"Warning: No valid data for parameter {param}, skipping from combined plot")
                continue
                
            valid_targets = targets[valid_mask]
            valid_median_preds = median_preds[valid_mask]
            valid_lower_preds = lower_preds[valid_mask]
            valid_upper_preds = upper_preds[valid_mask]
            
            # Sort for better visualization
            sort_idx = np.argsort(valid_targets)
            sorted_targets = valid_targets[sort_idx]
            sorted_lower = valid_lower_preds[sort_idx]
            sorted_upper = valid_upper_preds[sort_idx]
            
            # Plot confidence interval
            axes[i].fill_between(sorted_targets, sorted_lower, sorted_upper, alpha=0.3, color='lightblue')
            
            # Plot median predictions
            axes[i].scatter(valid_targets, valid_median_preds, alpha=0.7, s=20, color='blue')
            
            min_val = min(np.min(valid_targets), np.min(valid_median_preds))
            max_val = max(np.max(valid_targets), np.max(valid_median_preds))
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            axes[i].set_xlabel(f'True {param_info.get(param, {}).get("label", param)}')
            axes[i].set_ylabel(f'Predicted {param_info.get(param, {}).get("label", param)}')
            axes[i].set_title(f'{param}')
            axes[i].grid(True, alpha=0.3)
            
            # Calculate and display metrics
            mae = np.mean(np.abs(valid_median_preds - valid_targets))
            rmse = np.sqrt(np.mean((valid_median_preds - valid_targets)**2))
            r_squared = np.corrcoef(valid_targets, valid_median_preds)[0, 1]**2
            coverage = np.mean((valid_targets >= valid_lower_preds) & (valid_targets <= valid_upper_preds))
            
            axes[i].text(0.05, 0.95, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r_squared:.3f}\nCov: {coverage:.1%}\nN: {len(valid_targets)}',
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'stellar_parameters_combined_cqr_calibrated.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Combined plot with CQR calibrated confidence intervals saved")


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
    device, world_size, gpus_per_node = setup()
    print(f"Using device: {device}")
    
    # Create output directory relative to checkpoint path
    checkpoint_parent = os.path.dirname(inference_args.checkpoint_path)
    output_dir = os.path.join(checkpoint_parent, inference_args.output_dir)
    inference_args.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_model(inference_args.checkpoint_path, args, device)

    model.eval()
    
    # Create dataloaders (we only need test_loader)
    print("Creating test dataloader...")
    train_loader, val_loader, test_loader = create_datasets_and_loaders(args, device)

    optimizer, scheduler, scaler = create_optimizer_and_scheduler(model, args, train_loader)

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
    
    # Run predictions
    trainer = LLMTrainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        world_size=world_size,
        output_dim=1,
        scheduler=None,
        max_iter=args.max_iter,
        log_path=args.output_dir,
        exp_name=args.exp_name,
        lora_params=lora_params,
        scaler=None,
        use_amp=args.use_amp,
        max_grad_norm=args.max_grad_norm,
        mode=args.mode,
        curriculum_decay_steps=args.curriculum_decay_steps,
        quantiles=args.quantiles,
    )
    
    trainer.combined_mode = (args.mode == "combined")
    
    # Generate text predictions separately
    print("\n" + "="*60)
    print("GENERATING TEXT PREDICTIONS")
    print("="*60)
    text_preds, text_targets, obsids_text = llm_predict_with_text_generation(
        trainer, test_loader, device, max_iter=2, generate_text=True, args=args
    )

    # Get quantiles from args for CQR calibration
    quantiles = getattr(args, 'quantiles', [0.159, 0.5, 0.841])  # Default: ~1-sigma + median
    print(f"Using quantiles for CQR calibration: {quantiles}")
    
    # Create CQR loss function (same as used in training)
    loss_fn = CQR(quantiles=quantiles, reduction='none')
    
    print("\n" + "="*60)
    print("BULK PREDICTIONS")
    print("="*60)
    print("len val/test/train dataloader: ", len(val_loader), len(test_loader), "N/A")

    
    # Step 1: Get validation predictions for calibration
    preds_val, targets_val, obsids_val = llm_predict_stellar_bulk(trainer, val_loader, device, quantiles, max_iter=100)
    print(f"Validation predictions shape: {preds_val.shape}, targets: {targets_val.shape}")
    
    # Step 2: Get test predictions  
    preds, targets, obsids_stellar = llm_predict_stellar_bulk(trainer, test_loader, device, quantiles, max_iter=100)
    print(f"Test predictions shape: {preds.shape}, targets: {targets.shape}")
    
    # Step 3: Check coverage before calibration
    low_q = preds[:, :, 0]  # First quantile
    high_q = preds[:, :, -1]  # Last quantile
    print("before calibration - first sample predictions: ", low_q[0,0], high_q[0,0])
    coverage = np.mean((targets >= low_q) & (targets <= high_q))
    print('coverage before calibration: ', coverage)
    
    # Step 4: CQR Calibration (following APOGEE script exactly)
    print("\nApplying CQR calibration...")
    cqr_errs = loss_fn.calibrate(preds_val, targets_val)
    print(f"CQR errors shape: {cqr_errs.shape}")
    print(f"Targets shape: {targets.shape}, Preds shape: {preds.shape}")
    preds_cqr = loss_fn.predict(preds, cqr_errs)
    
    # Step 5: Check coverage after calibration
    low_q = preds_cqr[:, :, 0]
    high_q = preds_cqr[:, :, -1]
    print("after calibration shape: ", preds_cqr.shape)
    print("after calibration - first 10 samples low_q[:10,0]: ", low_q[:10,0])
    print("after calibration - first 10 samples high_q[:10,0]: ", high_q[:10,0])
    coverage = np.mean((targets >= low_q) & (targets <= high_q))
    print('coverage after calibration: ', coverage)
    
    
    # Verify sample alignment by checking obsids
    print(f"\n" + "="*60)
    print("VERIFYING SAMPLE ALIGNMENT")
    print("="*60)
    print(f"Stellar predictions: {len(obsids_stellar)} samples")
    print(f"Text predictions: {len(obsids_text)} samples")
    
    if len(obsids_stellar) == len(obsids_text):
        # Check if obsids match exactly
        mismatches = []
        for i, (stellar_obsid, text_obsid) in enumerate(zip(obsids_stellar, obsids_text)):
            if stellar_obsid != text_obsid:
                mismatches.append((i, stellar_obsid, text_obsid))
        
        if len(mismatches) == 0:
            print("✓ All obsids match perfectly - samples are aligned")
        else:
            print(f"⚠ Found {len(mismatches)} mismatched obsids:")
            for i, stellar_obsid, text_obsid in mismatches[:10]:  # Show first 10
                print(f"  Index {i}: stellar={stellar_obsid}, text={text_obsid}")
            if len(mismatches) > 10:
                print(f"  ... and {len(mismatches) - 10} more")
    else:
        print(f"⚠ Sample count mismatch: {len(obsids_stellar)} vs {len(obsids_text)}")
        
    print(f"First 5 stellar obsids: {obsids_stellar[:5]}")
    print(f"First 5 text obsids: {obsids_text[:5]}")
    
    # Step 7: Convert to format expected by plotting/saving functions
    # Unnormalize the calibrated predictions and convert to dict format
    stellar_preds_unnormalized = {}
    stellar_targets_unnormalized = {}
    
    # Unnormalize predictions (calibrated quantiles)
    for param_idx, param in enumerate(['Teff', 'logg', 'FeH']):
        if param == 'Teff':
            stellar_preds_unnormalized[param] = preds_cqr[:, param_idx, :] * (7500 - 3000) + 3000
            stellar_targets_unnormalized[param] = targets[:, param_idx] * (7500 - 3000) + 3000
        elif param == 'logg':
            stellar_preds_unnormalized[param] = preds_cqr[:, param_idx, :] * (5.0 - 0) + 0
            stellar_targets_unnormalized[param] = targets[:, param_idx] * (5.0 - 0) + 0
        elif param == 'FeH':
            stellar_preds_unnormalized[param] = preds_cqr[:, param_idx, :] * (0.5 - (-3)) + (-3)
            stellar_targets_unnormalized[param] = targets[:, param_idx] * (0.5 - (-3)) + (-3)
    
    # Use the calibrated predictions
    stellar_preds = stellar_preds_unnormalized
    stellar_targets = stellar_targets_unnormalized
    
    print("✓ CQR calibration completed! Confidence intervals are now properly calibrated.")
    print("Final shapes: ", {param: arr.shape for param, arr in stellar_preds.items()})
    # Save predictions
    if inference_args.save_predictions and stellar_preds is not None:
        print("saving predictions...")
        predictions_path = os.path.join(inference_args.output_dir, 'test_predictions.json')
        save_predictions(text_preds, text_targets, stellar_preds, stellar_targets, predictions_path)
    
    # Generate plots
    if inference_args.plot_results and stellar_preds is not None:
        plots_dir = os.path.join(inference_args.output_dir, 'plots')
        plot_stellar_parameters(stellar_preds, stellar_targets, plots_dir)
        
        # Generate consistency plots between stellar and text predictions
        if text_preds:
            consistency_dir = os.path.join(inference_args.output_dir, 'consistency_plots')
            plot_stellar_vs_text_consistency(stellar_preds, text_preds, consistency_dir)
    
    print("✓ Inference completed successfully!")


if __name__ == '__main__':
    main()