
import os
import sys
import torch
import numpy as np
import pandas as pd
import argparse

# Add root directory to path to allow imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

def get_args():
    parser = argparse.ArgumentParser(description="Extract Mass concept direction")
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
    
    # Match lengths (assuming alignment as per previous scripts)
    if len(df) != len(features):
        if len(features) < len(df):
            df = df.iloc[:len(features)]
        else:
            features = features[:len(df)]
    
    # 2. Clean Data
    # We need Mass, Age, FeH
    req_cols = ['Age', 'Mstar', 'FeH']
    for col in req_cols:
        if col not in df.columns:
            if col == 'Mstar' and 'mass' in df.columns: req_cols[req_cols.index(col)] = 'mass'
            if col == 'Age' and 'age' in df.columns: req_cols[req_cols.index(col)] = 'age' 
            
    df = df.dropna(subset=req_cols)
    df = df[(df['Age'] > 0) & (df['Mstar'] > 0)]
    
    valid_indices = df.index
    latents = features[valid_indices]
    df_clean = df.reset_index(drop=True)
    
    # 3. Binning Strategy
    # To extract a "Mass" concept that moves along an isochrone, we must control for Age and Metallicity.
    # If we don't control for FeH, we might capture metallicity effects (isochrone shifts) mixed with mass effects.
    
    age_min, age_max = df_clean['Age'].min(), df_clean['Age'].max()
    feh_min, feh_max = df_clean['FeH'].min(), df_clean['FeH'].max()
    
    # Age bins: 1 Gyr width
    age_bins = np.arange(np.floor(age_min), np.ceil(age_max) + 1.0, 1.0)
    # FeH bins: 0.2 dex width
    feh_bins = np.arange(np.floor(feh_min*5)/5, np.ceil(feh_max*5)/5 + 0.2, 0.2)
    
    concept_vectors = []
    total_weight = 0
    
    print(f"Binning stars: Age {age_min:.2f}-{age_max:.2f} Gyr, FeH {feh_min:.2f}-{feh_max:.2f}")
    
    stats = []
    
    MIN_SAMPLES = 5
    
    for i in range(len(age_bins)-1):
        for j in range(len(feh_bins)-1):
            a_low, a_high = age_bins[i], age_bins[i+1]
            f_low, f_high = feh_bins[j], feh_bins[j+1]
            
            mask_bin = (
                (df_clean['Age'] >= a_low) & (df_clean['Age'] < a_high) &
                (df_clean['FeH'] >= f_low) & (df_clean['FeH'] < f_high)
            )
            
            if mask_bin.sum() < MIN_SAMPLES * 2:
                continue
            
            bin_latents = latents[torch.tensor(mask_bin.values)]
            bin_masses = df_clean.loc[mask_bin, 'Mstar'].values
            
            # Divide into High Mass and Low Mass within this Age/FeH bin
            # We use quantiles: Bottom 25% vs Top 25%
            q25 = np.percentile(bin_masses, 25)
            q75 = np.percentile(bin_masses, 75)
            
            mask_low = bin_masses <= q25
            mask_high = bin_masses >= q75
            
            n_low = mask_low.sum()
            n_high = mask_high.sum()
            
            if n_low >= MIN_SAMPLES and n_high >= MIN_SAMPLES:
                vec_low = bin_latents[mask_low].mean(dim=0)
                vec_high = bin_latents[mask_high].mean(dim=0)
                
                # Direction: High Mass - Low Mass
                diff = vec_high - vec_low
                
                weight = n_low + n_high
                concept_vectors.append(diff * weight)
                total_weight += weight
                
                stats.append({
                    'age_range': (a_low, a_high),
                    'feh_range': (f_low, f_high),
                    'n_low': n_low,
                    'n_high': n_high,
                    'mass_low_mean': bin_masses[mask_low].mean(),
                    'mass_high_mean': bin_masses[mask_high].mean(),
                    'diff_norm': diff.norm().item()
                })
                
    if total_weight == 0:
        print("Error: No bins had sufficient data.")
        return
        
    final_concept = torch.stack(concept_vectors).sum(dim=0) / total_weight
    
    print(f"Computed Mass concept from {len(stats)} bins (Total stars: {total_weight}).")
    print(f"Concept Norm: {final_concept.norm().item():.4f}")
    
    # Save
    save_path = os.path.join(args.output_dir, 'mass_concept.pt')
    torch.save(final_concept, save_path)
    print(f"Saved concept to {save_path}")
    
    print("Sample bins used:")
    for s in stats[:5]:
        print(s)

if __name__ == "__main__":
    main()
