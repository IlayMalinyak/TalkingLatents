
import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from torch.utils.data import DataLoader
from collections import OrderedDict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from nn.spectra_model import MultiTaskRegressor
from nn.train import MaskedRegressorTrainer
from util.utils import Container, collate_with_idx, load_checkpoints_ddp
from nn.optim import CQR
from data.transforms import GeneralSpectrumPreprocessor, ToTensor, Compose
from data.spectra_dataset import SpectraDataset

# Bounds matching generate_features.py
BOUNDS = {'MAX_TEFF': 7500, 'MIN_TEFF': 3000, 'MAX_LOGG': 5.0, 'MIN_LOGG': 0, 'MAX_FEH': 0.5, 'MIN_FEH': -3}

def load_json_as_df(json_file: str) -> pd.DataFrame:
    """Load JSON file and convert to DataFrame with proper stellar parameters."""
    import json
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    records = []
    for item in data:
        record = {
            'obsid': item.get('obsid', item.get('id', '')),
            'teff': item.get('Teff'),
            'logg': item.get('logg'),
            'feh': item.get('FeH'),
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Apply normalization matching generate_features.py
    print("\nNormalizing parameters...")
    print("Values ranges before normalization:")
    for c in ['teff', 'logg', 'feh']:
        if c in df.columns:
            C = c.upper()
            if c != 'feh':
                df = df[df[c] > 0]
            print(f"{c}: {df[c].min():.2f} - {df[c].max():.2f}, nans: {df[c].isna().sum()}")
            
            # Only Teff is divided by 5778 (matching line 45 in generate_features.py)
            if c == 'teff':
                df[c] = df[c] / 5778.0
            
            # Then normalize using BOUNDS (matching lines 54 in generate_features.py)
            df[c] = (df[c] - BOUNDS[f'MIN_{C}']) / (BOUNDS[f'MAX_{C}'] - BOUNDS[f'MIN_{C}'])
            print(f"{c} (normalized): {df[c].min():.4f} - {df[c].max():.4f}, nans: {df[c].isna().sum()}")
    
    return df

def denormalize(val: np.ndarray, param: str) -> np.ndarray:
    """Denormalize values back to physical units."""
    param_upper = param.upper()
    low = BOUNDS[f'MIN_{param_upper}']
    high = BOUNDS[f'MAX_{param_upper}']
    
    # Reverse the normalization
    val_denorm = val * (high - low) + low
    
    # Reverse Teff division by 5778
    if param.lower() == 'teff':
        val_denorm = val_denorm * 5778.0
    
    return val_denorm

def predict(model: torch.nn.Module, test_dl: DataLoader, device: torch.device, max_iter: int = 100):
    """Run prediction using MaskedRegressorTrainer (matching generate_features.py lines 116-132)."""
    loss_fn = CQR(quantiles=[0.1, 0.25, 0.5, 0.75, 0.9], reduction='none')
    ssl_loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)

    trainer = MaskedRegressorTrainer(
        model=model, 
        optimizer=optimizer,
        criterion=loss_fn, 
        ssl_criterion=ssl_loss_fn,
        output_dim=3, 
        scaler=None,
        scheduler=None, 
        train_dataloader=None,
        val_dataloader=None, 
        device=device,
        num_quantiles=5,
        log_path=None, 
        range_update=None,
        accumulation_step=1, 
        max_iter=max_iter, 
        w_name=None,
        exp_name="spectral_regression_inference"
    )

    return trainer.predict(test_dl, device=device)

def main():
    parser = argparse.ArgumentParser(description='Spectral Model Regression Inference')
    
    # Model Args (from generate_features.py)
    parser.add_argument('--model_config', type=str, 
                        default='/home/ilay.kamai/work/MultiDESA/configs/lamost.yaml', 
                        help='Path to spectral model config (yaml)')
    parser.add_argument('--weights_path', type=str, 
                        default='/home/ilay.kamai/work/MultiDESA/pretrained_models/spectra.pth', 
                        help='Path to spectral model weights')
    
    # Data Args
    parser.add_argument('--json_file', type=str, 
                        default='/home/ilay.kamai/work/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json', 
                        help='Path to stellar descriptions JSON file')
    parser.add_argument('--spectra_dir', type=str, 
                        default=None,
                        help='Path to spectra directory (if None, use from config)')
    parser.add_argument('--output_dir', type=str, 
                        default='spectral_regression_results', 
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Num workers')
    parser.add_argument('--max_iter', type=int, default=100, help='Max iterations for prediction')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load config
    print(f"Loading config from: {args.model_config}")
    with open(args.model_config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 2. Load Model (matching generate_features.py lines 106-115)
    print("Loading MultiTaskRegressor model...")
    spectra_args = Container(**cfg['MultiTaskRegressor'])
    conformer_args = Container(**cfg['Conformer'])
    model = MultiTaskRegressor(spectra_args, conformer_args)
    
    # Load weights (matching generate_features.py line 162)
    model = load_checkpoints_ddp(model, args.weights_path)
    model.to(device)
    model.eval()
    
    # 3. Load and prepare data (matching generate_features.py lines 165-182)
    print(f"Loading data from: {args.json_file}")
    df = load_json_as_df(args.json_file)
    
    # Use spectra_dir from args or config
    spectra_dir = args.spectra_dir if args.spectra_dir else cfg['Data']['spectra_dir']
    
    # Transformations (matching generate_features.py line 170)
    # For JSON data, rv_norm should be False (like simulation data)
    rv_norm = False
    transf = Compose([GeneralSpectrumPreprocessor(rv_norm=rv_norm), ToTensor()])
    
    # Create dataset (matching generate_features.py lines 171-178)
    test_ds = SpectraDataset(
        spectra_dir,
        transforms=transf, 
        df=df,
        max_len=cfg['Data']['max_len'],
        target_norm=cfg['Data']['target_norm'],
        id='obsid',
        labels=cfg['Data']['prediction_labels'],
        store_wv=True
    )
    
    # Create dataloader (matching generate_features.py lines 179-182)
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        collate_fn=collate_with_idx,
        num_workers=args.num_workers
    )
    
    # 4. Run prediction (matching generate_features.py line 190)
    print("Starting inference...")
    preds, targets, features, tokens, decodes, xs, info, mean_loss = predict(
        model, test_dl, device, max_iter=args.max_iter
    )
    
    print(f"Predictions shape: {preds.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Mean loss: {mean_loss:.4f}")
    
    # 5. Calculate metrics
    # preds shape: [N, 3, 5] (samples, params, quantiles)
    # targets shape: [N, 3] (samples, params)
    
    # Extract median predictions (quantile index 2 for 5 quantiles)
    median_idx = 2
    preds_median = preds[:, :, median_idx]  # [N, 3]
    
    print("\n" + "="*80)
    print("SPECTRAL MODEL REGRESSION METRICS REPORT")
    print("="*80)
    print(f"{'Parameter':<10} | {'RMSE (norm)':<12} | {'MAE (norm)':<12} | {'RMSE (phys)':<12} | {'MAE (phys)':<12}")
    print("-" * 80)
    
    param_names = ['teff', 'logg', 'feh']
    for i, param in enumerate(param_names):
        y_true_norm = targets[:, i]
        y_pred_norm = preds_median[:, i]
        
        # Metrics in normalized space
        rmse_norm = np.sqrt(mean_squared_error(y_true_norm, y_pred_norm))
        mae_norm = mean_absolute_error(y_true_norm, y_pred_norm)
        
        # Denormalize and compute physical metrics
        y_true_phys = denormalize(y_true_norm, param)
        y_pred_phys = denormalize(y_pred_norm, param)
        
        rmse_phys = np.sqrt(mean_squared_error(y_true_phys, y_pred_phys))
        mae_phys = mean_absolute_error(y_true_phys, y_pred_phys)
        
        print(f"{param:<10} | {rmse_norm:<12.4f} | {mae_norm:<12.4f} | {rmse_phys:<12.2f} | {mae_phys:<12.2f}")
    
    print("-" * 80)
    
    # 6. Save results
    print(f"\nSaving results to: {args.output_dir}")
    np.save(os.path.join(args.output_dir, 'preds.npy'), preds)
    np.save(os.path.join(args.output_dir, 'targets.npy'), targets)
    np.save(os.path.join(args.output_dir, 'features.npy'), features)
    np.save(os.path.join(args.output_dir, 'tokens.npy'), tokens)
    
    # Save info as CSV if available
    if info:
        max_length = max(len(v) for v in info.values())
        for key in info:
            current_length = len(info[key])
            if current_length < max_length:
                info[key].extend([np.nan] * (max_length - current_length))
        info_df = pd.DataFrame(info)
        info_df.to_csv(os.path.join(args.output_dir, 'info.csv'), index=False)
    
    print("Done!")

if __name__ == "__main__":
    main()
