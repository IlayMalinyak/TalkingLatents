
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Add root directory to path to allow imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

def get_args():
    parser = argparse.ArgumentParser(description="Analyze Age concept")
    parser.add_argument("--info_file", type=str, default='/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/info_full.csv')
    parser.add_argument("--features_file", type=str, default='/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/features.npy')
    parser.add_argument("--concept_file", type=str, default='/home/ilay.kamai/work/TalkingLatents/concepts/age_concept.pt')
    parser.add_argument("--output_dir", type=str, default='/home/ilay.kamai/work/TalkingLatents/concepts/plots_age')
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv(args.info_file)
    features = np.load(args.features_file)
    features = torch.tensor(features)
    
    # Match lengths
    if len(df) != len(features):
        if len(features) < len(df):
            df = df.iloc[:len(features)]
        else:
            features = features[:len(df)]
            
    # Clean
    req_cols = ['Age', 'Mstar', 'FeH']
    for col in req_cols:
         if col not in df.columns:
            if col == 'Mstar' and 'mass' in df.columns: req_cols[req_cols.index(col)] = 'mass'
            if col == 'Age' and 'age' in df.columns: req_cols[req_cols.index(col)] = 'age'
            
    df = df.dropna(subset=req_cols)
    df = df[(df['Age'] > 0) & (df['Mstar'] > 0)]
    
    valid_indices = df.index
    latents = features[valid_indices]
    df = df.reset_index(drop=True)
    
    # 2. Load Concept
    print(f"Loading concept from {args.concept_file}...")
    if not os.path.exists(args.concept_file):
        print("Concept file not found. Run extract_age_concept.py first.")
        return
        
    concept = torch.load(args.concept_file)
    # Normalize concept
    concept = concept / concept.norm()
    
    # 3. Project
    print("Projecting latents...")
    # latents: [N, D], concept: [D]
    projections = torch.matmul(latents.float(), concept.float()).numpy()
    
    df['Projection'] = projections
    
    # 4. Plots
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Age vs Projection (Overall)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Age', y='Projection', alpha=0.1, s=10)
    plt.title("Latent Projection vs Stellar Age")
    plt.xlabel("Age (Gyr)")
    plt.ylabel("Projection onto Age Concept")
    plt.savefig(os.path.join(args.output_dir, 'age_vs_projection.png'))
    plt.close()
    
    # Plot 2: Age vs Projection (Heatmap/Hexbin) for density
    plt.figure(figsize=(10, 6))
    plt.hexbin(df['Age'], df['Projection'], gridsize=50, cmap='viridis', mincnt=1)
    plt.colorbar(label='Count')
    plt.title("Latent Projection vs Stellar Age (Density)")
    plt.xlabel("Age (Gyr)")
    plt.ylabel("Projection onto Age Concept")
    plt.savefig(os.path.join(args.output_dir, 'age_vs_projection_hex.png'))
    plt.close()
    
    # Plot 3: Stratified by Mass
    # Take a few mass slices
    mass_slices = [(0.9, 1.1), (1.1, 1.3), (1.3, 1.5)]
    
    plt.figure(figsize=(12, 4))
    for i, (m_low, m_high) in enumerate(mass_slices):
        plt.subplot(1, 3, i+1)
        subset = df[(df['Mstar'] >= m_low) & (df['Mstar'] < m_high)]
        if len(subset) > 0:
            sns.scatterplot(data=subset, x='Age', y='Projection', alpha=0.3, s=15)
            correlation = subset['Age'].corr(subset['Projection'])
            plt.title(f"Mass {m_low}-{m_high} $M_\odot$\nCorr: {correlation:.2f}")
            plt.xlabel("Age (Gyr)")
            plt.ylabel("Projection")
            
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'age_vs_projection_by_mass.png'))
    plt.close()
    
    # Plot 4: Correlation with other params (Check for leakage)
    # Plot Projection vs Mass and Projection vs FeH
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.scatterplot(data=df, x='Mstar', y='Projection', ax=axes[0], alpha=0.1, s=5)
    axes[0].set_title(f"Projection vs Mass (Corr: {df['Mstar'].corr(df['Projection']):.2f})")
    
    sns.scatterplot(data=df, x='FeH', y='Projection', ax=axes[1], alpha=0.1, s=5)
    axes[1].set_title(f"Projection vs FeH (Corr: {df['FeH'].corr(df['Projection']):.2f})")
    
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'leakage_check.png'))
    plt.close()

    # Plot 5: Histograms (Old vs Young)
    plt.figure(figsize=(10, 6))
    
    q25 = df['Age'].quantile(0.25)
    q75 = df['Age'].quantile(0.75)
    
    mask_young = df['Age'] <= q25
    mask_old = df['Age'] >= q75
    
    sns.histplot(data=df[mask_old], x='Projection', color="red", label="Old (>= 75%)", kde=True, stat="density", alpha=0.5)
    sns.histplot(data=df[mask_young], x='Projection', color="blue", label="Young (<= 25%)", kde=True, stat="density", alpha=0.5)
    
    plt.title(f"Projection Distribution: Old vs Young")
    plt.xlabel("Projection Score (High ~ Old)")
    plt.legend()
    
    plt.savefig(os.path.join(args.output_dir, 'hist_age_old_minus_young.png'))
    plt.close()
    
    print(f"Analysis complete. Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
