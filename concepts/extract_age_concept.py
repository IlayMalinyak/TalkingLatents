
import os
import sys
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

# Add root directory to path to allow imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

def get_args():
    parser = argparse.ArgumentParser(description="Extract Age concept direction")
    parser.add_argument("--info_file", type=str, default='/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/info_full.csv')
    parser.add_argument("--features_file", type=str, default='/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/features.npy')
    parser.add_argument("--output_dir", type=str, default='/home/ilay.kamai/work/TalkingLatents/concepts')
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading metadata from {args.info_file}...")
    df = pd.read_csv(args.info_file)
    
    print(f"Loading features from {args.features_file}...")
    features = np.load(args.features_file)
    features = torch.tensor(features)
    
    # 2. Filter and Match
    # Assuming the order in features.npy matches the original dataset order.
    # We need to ensure we map rows correctly. 
    # Usually dataset follows the JSON or CSV order. 
    # Let's assume features.npy aligns with the dataset used to generate it.
    # Ideally we'd have an index mapping. If info_full.csv was generated ALONG WITH features, they might align.
    # However, info_full.csv has 18999 rows (from index 0 to 18998 seen in head).
    # features.npy shape needs to be checked.
    # If they don't align perfectly, we might have a problem unless we have obsids for features.
    # Standard practice here: assume 1-to-1 correspondence if lengths match, or use 'obsid' if available.
    
    if len(df) != len(features):
        print(f"Warning: Length mismatch! DF: {len(df)}, Features: {len(features)}")
        # Assuming features correspond to the first N items or similar is risky.
        # But we don't have a map file.
        # Let's proceed assuming they align and we just filter valid rows.
        if len(features) < len(df):
            df = df.iloc[:len(features)]
        else:
            features = features[:len(df)]
    
    print(f"Working with {len(df)} samples.")
    
    # 3. Clean Data
    # We need Age, Mass (Mstar), Metallicity (FeH)
    req_cols = ['Age', 'Mstar', 'FeH']
    for col in req_cols:
        if col not in df.columns:
             # Check for different names
            if col == 'Mstar' and 'mass' in df.columns: req_cols[req_cols.index(col)] = 'mass'
            if col == 'Age' and 'age' in df.columns: req_cols[req_cols.index(col)] = 'age' 
            # etc. From head output: 'Mstar', 'Age', 'FeH' exist.
            
    df = df.dropna(subset=req_cols)
    # Also filter bad values
    df = df[df['Age'] > 0]
    df = df[df['Mstar'] > 0]
    
    # Indices of valid rows
    valid_indices = df.index
    
    latents = features[valid_indices]
    df_clean = df.reset_index(drop=True)
    
    # 4. Binning Strategy
    # Mass bins: 0.1 Msun
    # FeH bins: 0.2 dex
    
    mass_min, mass_max = df_clean['Mstar'].min(), df_clean['Mstar'].max()
    feh_min, feh_max = df_clean['FeH'].min(), df_clean['FeH'].max()
    
    mass_bins = np.arange(np.floor(mass_min*10)/10, np.ceil(mass_max*10)/10 + 0.1, 0.1)
    feh_bins = np.arange(np.floor(feh_min*5)/5, np.ceil(feh_max*5)/5 + 0.2, 0.2)
    
    concept_vectors = []
    total_weight = 0
    
    print(f"Binning stars: Mass {mass_min:.2f}-{mass_max:.2f}, FeH {feh_min:.2f}-{feh_max:.2f}")
    
    # Definitions
    YOUNG_THRESH = 2.0
    OLD_THRESH = 7.0
    MIN_SAMPLES = 5
    
    stats = []

    for i in range(len(mass_bins)-1):
        for j in range(len(feh_bins)-1):
            m_low, m_high = mass_bins[i], mass_bins[i+1]
            f_low, f_high = feh_bins[j], feh_bins[j+1]
            
            mask_bin = (
                (df_clean['Mstar'] >= m_low) & (df_clean['Mstar'] < m_high) &
                (df_clean['FeH'] >= f_low) & (df_clean['FeH'] < f_high)
            )
            
            if mask_bin.sum() < MIN_SAMPLES * 2:
                continue
                
            bin_latents = latents[torch.tensor(mask_bin.values)]
            bin_ages = df_clean.loc[mask_bin, 'Age'].values
            
            mask_young = bin_ages < YOUNG_THRESH
            mask_old = bin_ages > OLD_THRESH
            
            n_young = mask_young.sum()
            n_old = mask_old.sum()
            
            if n_young >= MIN_SAMPLES and n_old >= MIN_SAMPLES:
                # Calculate vector for this bin
                # Convert boolean numpy masks to torch indices relative to the bin
                # Note: mask_young is boolean array of length bin_latents
                
                vec_young = bin_latents[mask_young].mean(dim=0)
                vec_old = bin_latents[mask_old].mean(dim=0)
                
                # Direction: Old - Young
                diff = vec_old - vec_young
                
                # Weight by number of samples involved
                weight = n_young + n_old
                
                concept_vectors.append(diff * weight)
                total_weight += weight
                
                stats.append({
                    'mass_range': (m_low, m_high),
                    'feh_range': (f_low, f_high),
                    'n_young': n_young,
                    'n_old': n_old,
                    'diff_norm': diff.norm().item()
                })

    if total_weight == 0:
        print("Error: No bins had sufficient data for both young and old populations.")
        return

    # Weighted Average
    final_concept = torch.stack(concept_vectors).sum(dim=0) / total_weight
    
    print(f"Computed Age concept from {len(stats)} bins (Total stars: {total_weight}).")
    print(f"Concept Norm: {final_concept.norm().item():.4f}")
    
    # Save
    save_path = os.path.join(args.output_dir, 'age_concept.pt')
    torch.save(final_concept, save_path)
    print(f"Saved concept to {save_path}")
    
    # Also save stats for review
    # print sample of stats
    print("Sample bins used:")
    for s in stats[:5]:
        print(s)

if __name__ == "__main__":
    main()
