
import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm
import yaml
from pathlib import Path
import pandas as pd

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
    parser.add_argument("--info_file", type=str, default='/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/info_full.csv', help="Path to info_full.csv for Age/Mass data")
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
    Extracts latents, parameters, and OBSIDs from a dataloader.
    Returns clean tensors (NaNs removed).
    """
    print(f"Extracting latents from {name}...")
    all_latents = []
    all_params = [] 
    all_obsids = []
    
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
            
            # Extract ObsIDs
            if 'obsids' in batch:
                all_obsids.extend(batch['obsids'])

    if not all_latents:
         return torch.empty(0), torch.empty(0), []

    latents = torch.cat(all_latents, dim=0)
    params = torch.cat(all_params, dim=0)
    
    # Remove NaNs
    valid_mask = ~torch.isnan(params).any(dim=1)
    latents = latents[valid_mask]
    params = params[valid_mask]
    
    # Filter obsids
    valid_indices = torch.nonzero(valid_mask).squeeze().tolist()
    # Handle single element or empty case
    if isinstance(valid_indices, int): valid_indices = [valid_indices]
    
    final_obsids = [all_obsids[i] for i in valid_indices]
    
    print(f"  [{name}] Extracted {latents.shape[0]} valid samples.")
    return latents, params, final_obsids

def compute_mass_concept(latents, obsids, info_df):
    """
    Computes Mass concept direction controlling for Age and FeH.
    """
    print("Computing Mass Concept (Stratified)...")
    
    # Match latents with info_df using obsids
    # Create DataFrame for current latents
    latent_df = pd.DataFrame({'obsid': obsids})
    # Ensure obsid types match (int vs float vs str)
    # info_df usually has int or float obsid. latent_df has whatever was in metadata.
    # Let's try to align types.
    try:
        latent_df['obsid'] = latent_df['obsid'].astype(int)
        info_df['obsid'] = info_df['obsid'].astype(int) 
    except:
        pass
    
    # Drop duplicates in info_df to prevent expanding rows
    info_df = info_df.drop_duplicates(subset='obsid')
        
    merged = latent_df.merge(info_df, on='obsid', how='left')
    
    # Check for missing data
    # We need Age, Mstar, FeH
    # Mstar might be named differently
    cols = {'Age': 'Age', 'Mstar': 'Mstar', 'FeH': 'FeH'}
    for k, v in cols.items():
        if v not in merged.columns and k.lower() in merged.columns:
            cols[k] = k.lower() # fallback to lowercase
            
    # Filter valid
    valid_idx = merged.dropna(subset=cols.values()).index
    
    if len(valid_idx) < 100:
        print("  Not enough data with matching Age/Mass/FeH for Mass Concept.")
        return None
        
    df_clean = merged.loc[valid_idx].reset_index(drop=True)
    # Filter latents
    clean_latents = latents[torch.tensor(valid_idx.values)]
    
    # Binning
    age_col = cols['Age']
    mass_col = cols['Mstar']
    feh_col = cols['FeH']
    
    # Filter bad values
    mask_good = (df_clean[age_col] > 0) & (df_clean[mass_col] > 0)
    df_clean = df_clean[mask_good]
    # Update latents again
    # We need to be careful with indexing.
    # Let's align boolean mask
    clean_latents = clean_latents[torch.tensor(mask_good.values)]
    
    age_min, age_max = df_clean[age_col].min(), df_clean[age_col].max()
    feh_min, feh_max = df_clean[feh_col].min(), df_clean[feh_col].max()
    
    # Age bins: 1 Gyr width
    age_bins = np.arange(np.floor(age_min), np.ceil(age_max) + 1.0, 1.0)
    # FeH bins: 0.2 dex width
    feh_bins = np.arange(np.floor(feh_min*5)/5, np.ceil(feh_max*5)/5 + 0.2, 0.2)
    
    concept_vectors = []
    total_weight = 0
    stats = []
    MIN_SAMPLES = 5
    
    for i in range(len(age_bins)-1):
        for j in range(len(feh_bins)-1):
            a_low, a_high = age_bins[i], age_bins[i+1]
            f_low, f_high = feh_bins[j], feh_bins[j+1]
            
            mask_bin = (
                (df_clean[age_col] >= a_low) & (df_clean[age_col] < a_high) &
                (df_clean[feh_col] >= f_low) & (df_clean[feh_col] < f_high)
            )
            
            if mask_bin.sum() < MIN_SAMPLES * 2:
                continue
            
            bin_latents = clean_latents[torch.tensor(mask_bin.values)]
            bin_masses = df_clean.loc[mask_bin, mass_col].values
            
            q25 = np.percentile(bin_masses, 25)
            q75 = np.percentile(bin_masses, 75)
            
            mask_low = bin_masses <= q25
            mask_high = bin_masses >= q75
            
            n_low = mask_low.sum()
            n_high = mask_high.sum()
            
            if n_low >= MIN_SAMPLES and n_high >= MIN_SAMPLES:
                vec_low = bin_latents[mask_low].mean(dim=0)
                vec_high = bin_latents[mask_high].mean(dim=0)
                
                # High Mass - Low Mass
                diff = vec_high - vec_low
                weight = n_low + n_high
                
                concept_vectors.append(diff * weight)
                total_weight += weight
                
                stats.append({'n_samples': weight})
                
    if total_weight == 0:
        print("  Failed to compute Mass concept: No sufficient bins.")
        return None
        
    final_concept = torch.stack(concept_vectors).sum(dim=0) / total_weight
    print(f"  Computed Mass Concept from {len(stats)} bins (N={total_weight})")
    return final_concept

def compute_age_concept(latents, obsids, info_df):
    """
    Computes Age concept direction controlling for Mass and FeH.
    """
    print("Computing Age Concept (Stratified)...")
    
    # Match latents with info_df using obsids
    latent_df = pd.DataFrame({'obsid': obsids})
    try:
        latent_df['obsid'] = latent_df['obsid'].astype(int)
        info_df['obsid'] = info_df['obsid'].astype(int) 
    except:
        pass
    
    info_df = info_df.drop_duplicates(subset='obsid')
    merged = latent_df.merge(info_df, on='obsid', how='left')
    
    cols = {'Age': 'Age', 'Mstar': 'Mstar', 'FeH': 'FeH'}
    for k, v in cols.items():
        if v not in merged.columns and k.lower() in merged.columns:
            cols[k] = k.lower()
            
    valid_idx = merged.dropna(subset=cols.values()).index
    
    if len(valid_idx) < 100:
        print("  Not enough data with matching Age/Mass/FeH for Age Concept.")
        return None
        
    df_clean = merged.loc[valid_idx].reset_index(drop=True)
    clean_latents = latents[torch.tensor(valid_idx.values)]
    
    age_col = cols['Age']
    mass_col = cols['Mstar']
    feh_col = cols['FeH']
    
    mask_good = (df_clean[age_col] > 0) & (df_clean[mass_col] > 0)
    df_clean = df_clean[mask_good]
    clean_latents = clean_latents[torch.tensor(mask_good.values)]
    
    mass_min, mass_max = df_clean[mass_col].min(), df_clean[mass_col].max()
    feh_min, feh_max = df_clean[feh_col].min(), df_clean[feh_col].max()
    
    # Mass bins: 0.1 Msun width
    mass_bins = np.arange(np.floor(mass_min*10)/10, np.ceil(mass_max*10)/10 + 0.1, 0.1)
    # FeH bins: 0.2 dex width
    feh_bins = np.arange(np.floor(feh_min*5)/5, np.ceil(feh_max*5)/5 + 0.2, 0.2)
    
    concept_vectors = []
    total_weight = 0
    stats = []
    MIN_SAMPLES = 5
    
    for i in range(len(mass_bins)-1):
        for j in range(len(feh_bins)-1):
            m_low, m_high = mass_bins[i], mass_bins[i+1]
            f_low, f_high = feh_bins[j], feh_bins[j+1]
            
            mask_bin = (
                (df_clean[mass_col] >= m_low) & (df_clean[mass_col] < m_high) &
                (df_clean[feh_col] >= f_low) & (df_clean[feh_col] < f_high)
            )
            
            if mask_bin.sum() < MIN_SAMPLES * 2:
                continue
            
            bin_latents = clean_latents[torch.tensor(mask_bin.values)]
            bin_ages = df_clean.loc[mask_bin, age_col].values
            
            q25 = np.percentile(bin_ages, 25)
            q75 = np.percentile(bin_ages, 75)
            
            mask_low = bin_ages <= q25  # Young
            mask_high = bin_ages >= q75 # Old
            
            n_low = mask_low.sum()
            n_high = mask_high.sum()
            
            if n_low >= MIN_SAMPLES and n_high >= MIN_SAMPLES:
                vec_low = bin_latents[mask_low].mean(dim=0)
                vec_high = bin_latents[mask_high].mean(dim=0)
                
                # Old - Young
                diff = vec_high - vec_low
                weight = n_low + n_high
                
                concept_vectors.append(diff * weight)
                total_weight += weight
                stats.append({'n_samples': weight})
                
    if total_weight == 0:
        print("  Failed to compute Age concept: No sufficient bins.")
        return None
        
    final_concept = torch.stack(concept_vectors).sum(dim=0) / total_weight
    print(f"  Computed Age Concept from {len(stats)} bins (N={total_weight})")
    return final_concept


def compute_directions(latents, params, numeric_bounds, obsids=None, info_df=None):
    """
    Computes concept directions from latents and params.
    """
    print("Computing numeric concept directions...")
    
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
        
    # --- Mass (Stratified) ---
    if obsids is not None and info_df is not None:
        mass_vec = compute_mass_concept(latents, obsids, info_df)
        if mass_vec is not None:
            concepts['mass_high_minus_low'] = mass_vec
            
        age_vec = compute_age_concept(latents, obsids, info_df)
        if age_vec is not None:
            concepts['age_old_minus_young'] = age_vec

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
    
    # Load Info Dataframe for Mass Concept
    info_df = None
    if os.path.exists(args.info_file):
        print(f"Loading metadata from {args.info_file}...")
        try:
            info_df = pd.read_csv(args.info_file)
        except Exception as e:
            print(f"Warning: Failed to load info_file: {e}")
    else:
        print(f"Warning: info_file not found at {args.info_file}")

    # 3. Phase 1: Directions from Training Data
    print("\n--- Phase 1: Computing Directions from Training Data ---")
    
    # We used Train split for basic calc, but for Mass Stratified we might want more data?
    # extract_from_loader respects split. 
    # BUT, if we use info_df and obsids, we match what is in the split.
    # So Mass concept will be computed from Training data only. This is correct to prevent leakage.
    
    train_latents, train_params, train_obsids = extract_from_loader(model, train_loader, device, name="Train")
    concepts = compute_directions(train_latents, train_params, numeric_bounds, train_obsids, info_df)
    
    # Save Directions
    save_path = os.path.join(args.output_dir, 'concept_directions.pt')
    torch.save(concepts, save_path)
    print(f"Saved {len(concepts)} concept directions to {save_path}")

    # 4. Phase 2: Extract Test Data for Analysis
    print("\n--- Phase 2: Extracting Unseen Test Data for Analysis ---")
    # Using TEST set for analysis to ensure no leakage
    test_latents, test_params, test_obsids = extract_from_loader(model, test_loader, device, name="Test")
    
    # Save Data for Analysis (Test Split)
    data_save_path = os.path.join(args.output_dir, 'concept_data.pt')
    torch.save({
        'latents': test_latents,
        'params': test_params,
        'obsids': test_obsids,
        'concepts': concepts
    }, data_save_path)
    print(f"Saved TEST split latents and params to {data_save_path}")

if __name__ == "__main__":
    main()
