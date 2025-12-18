
import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm
import yaml
from pathlib import Path

# Add root directory to path to allow imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.system('pip install tiktoken fairscale fire blobfile torchdiffeq torchcfm transformers bitsandbytes accelerate')

from src.simple_questions import _load_spectra_model, create_stellar_dataloaders
from data.transforms import Compose, GeneralSpectrumPreprocessor, ToTensor
from data.dataset_interpert import StellarQuestionsDataset

def get_args():
    parser = argparse.ArgumentParser(description="Extract concept directions")
    parser.add_argument("--json_file", type=str, default='/home/ilay.kamai/work/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json')
    parser.add_argument("--features_file", type=str, default='/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/features.npy')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default='/home/ilay.kamai/work/TalkingLatents/concepts')
    parser.add_argument("--tokenizer_path", type=str, default="/home/ilay.kamai/work/.llama/Llama3.1-8B/tokenizer.model")
    return parser.parse_args()

def is_dwarf(teff, logg):
    """
    Condition for dwarfs (Main Sequence) vs Giants.
    Returns True if Dwarf, False if Giant.
    """
    thresh = torch.zeros_like(teff)
    
    # Vectorized implementation
    mask_hot = teff >= 6000
    mask_cool = teff <= 4250
    mask_mid = ~(mask_hot | mask_cool)
    
    thresh[mask_hot] = 3.5
    thresh[mask_cool] = 4.0
    thresh[mask_mid] = 5.2 - (2.8e-4 * teff[mask_mid])
    
    return logg >= thresh

def extract_from_loader(model, loader, device, name="Split"):
    """
    Extracts latents and parameters from a dataloader.
    Returns clean tensors (NaNs removed).
    """
    print(f"Extracting latents from {name}...")
    all_latents = []
    all_params = [] 
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=name):
            spectra = batch['spectra']
            
            if model is None:
                # Features are pre-computed and stored in 'spectra'
                x_enc = spectra
            else:
                spectra = spectra.to(device)
                if len(spectra.shape) == 2:
                    spectra = spectra.unsqueeze(1)
                x_enc, _ = model.encoder(spectra)
            
            # Save cpu tensor
            all_latents.append(x_enc.cpu())
            all_params.append(batch['y_numeric'])

    if not all_latents:
         return torch.empty(0), torch.empty(0)

    latents = torch.cat(all_latents, dim=0)
    params = torch.cat(all_params, dim=0)
    
    # Remove NaNs
    valid_mask = ~torch.isnan(params).any(dim=1)
    latents = latents[valid_mask]
    params = params[valid_mask]
    
    print(f"  [{name}] Extracted {latents.shape[0]} valid samples.")
    return latents, params

def compute_directions(latents, params, numeric_bounds):
    """
    Computes concept directions from latents and params.
    """
    print("Computing concept directions...")
    
    # Denormalize params
    teff_norm = params[:, 0]
    logg_norm = params[:, 1]
    feh_norm = params[:, 2]
    
    def denorm(val, key):
        low, high = numeric_bounds[key]
        return val * (high - low) + low

    teff = denorm(teff_norm, 'Teff')
    logg = denorm(logg_norm, 'logg')
    feh = denorm(feh_norm, 'FeH')
    
    concepts = {}
    
    # --- Evolution: Giant vs Dwarf ---
    is_dw = is_dwarf(teff, logg)
    mask_dwarf = is_dw
    mask_giant = ~is_dw
    
    print(f"  Evolution - Giant: {mask_giant.sum()}, Dwarf: {mask_dwarf.sum()}")
    if mask_giant.sum() > 0 and mask_dwarf.sum() > 0:
        mu_giant = latents[mask_giant].mean(dim=0)
        mu_dwarf = latents[mask_dwarf].mean(dim=0)
        concepts['evolution_giant_minus_dwarf'] = mu_giant - mu_dwarf

    # --- Teff: Hot (> 6250) vs Cool (< 4500) ---
    mask_hot = teff > 6250 
    mask_cool = teff < 4500 
    print(f"  Teff - Hot: {mask_hot.sum()}, Cool: {mask_cool.sum()}")
    if mask_hot.sum() > 0 and mask_cool.sum() > 0:
        mu_hot = latents[mask_hot].mean(dim=0)
        mu_cool = latents[mask_cool].mean(dim=0)
        concepts['teff_hot_minus_cool'] = mu_hot - mu_cool

    # --- logg: High (> 4.5) vs Low (< 3) ---
    mask_logg_high = logg > 4.5
    mask_logg_low = logg < 3
    print(f"  logg - High: {mask_logg_high.sum()}, Low: {mask_logg_low.sum()}")
    if mask_logg_high.sum() > 0 and mask_logg_low.sum() > 0:
        mu_high = latents[mask_logg_high].mean(dim=0)
        mu_low = latents[mask_logg_low].mean(dim=0)
        concepts['logg_high_minus_low'] = mu_high - mu_low

    # --- FeH: Rich (> 0.2) vs Poor (< -0.2) ---
    mask_rich = feh > 0.2
    mask_poor = feh < -0.2
    print(f"  FeH - Rich: {mask_rich.sum()}, Poor: {mask_poor.sum()}")
    if mask_rich.sum() > 0 and mask_poor.sum() > 0:
        mu_rich = latents[mask_rich].mean(dim=0)
        mu_poor = latents[mask_poor].mean(dim=0)
        concepts['feh_rich_minus_poor'] = mu_rich - mu_poor

    return concepts

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Model (Optional if features exist)
    model = None
    if os.path.exists(args.features_file):
        print(f"Found pre-computed features at {args.features_file}")
        print("Skipping model loading and using features directly.")
    else:
        print("Loading spectra model...")
        model = _load_spectra_model()
        model = model.to(device)
        model.eval()

    # 2. Load Data
    print("Loading data...")
    class DataArgs:
        json_file = args.json_file
        features_file = args.features_file 
        output_dir = args.output_dir
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15
        random_seed = 42
        num_spectral_features = 1 
        max_seq_len = 512
        batch_size = args.batch_size
        num_workers = 4
        
    data_args = DataArgs()
    
    spectral_features = None
    if os.path.exists(args.features_file):
         spectral_features = np.load(args.features_file)
    
    transf = Compose([GeneralSpectrumPreprocessor(rv_norm=True), ToTensor()])
    cache_dir = os.path.join(args.output_dir, 'cache')
    
    # Pass normalize_features=False if using pre-computed features to preserve latent space geometry
    normalize_features = (spectral_features is None)

    train_loader, val_loader, test_loader = create_stellar_dataloaders(
        json_file=data_args.json_file,
        features_array=spectral_features,
        spectral_transforms=transf,
        train_ratio=data_args.train_ratio,
        val_ratio=data_args.val_ratio,
        test_ratio=data_args.test_ratio,
        random_state=data_args.random_seed,
        num_spectral_features=data_args.num_spectral_features,
        cache_dir=cache_dir,
        tokenizer_path=args.tokenizer_path,
        max_length=data_args.max_seq_len,
        batch_size=data_args.batch_size,
        num_workers=data_args.num_workers,
        normalize_features=normalize_features
    )
    
    numeric_bounds = train_loader.dataset.numeric_bounds

    # 3. Phase 1: Directions from Training Data
    print("\n--- Phase 1: Computing Directions from Training Data ---")
    train_latents, train_params = extract_from_loader(model, train_loader, device, name="Train")
    concepts = compute_directions(train_latents, train_params, numeric_bounds)
    
    # Save Directions
    save_path = os.path.join(args.output_dir, 'concept_directions.pt')
    torch.save(concepts, save_path)
    print(f"Saved {len(concepts)} concept directions to {save_path}")

    # 4. Phase 2: Extract Test Data for Analysis
    print("\n--- Phase 2: Extracting Unseen Test Data for Analysis ---")
    # Using TEST set for analysis to ensure no leakage
    test_latents, test_params = extract_from_loader(model, test_loader, device, name="Test")
    
    # Save Data for Analysis (Test Split)
    data_save_path = os.path.join(args.output_dir, 'concept_data.pt')
    torch.save({
        'latents': test_latents,
        'params': test_params,
        'concepts': concepts
    }, data_save_path)
    print(f"Saved TEST split latents and params to {data_save_path}")

if __name__ == "__main__":
    main()
