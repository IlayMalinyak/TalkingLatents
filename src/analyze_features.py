import numpy as np
import matplotlib.pyplot as plt
import os
import umap
import pandas as pd
import math

def plot_features(features, save_dir, sample_size=10):
    B, N = features.shape
    sample_size = min(sample_size, B)
    sample_indices = np.random.choice(B, size=sample_size, replace=False)
    sample_features = features[sample_indices]
    
    # Create a comprehensive figure with multiple subplots
    fig = plt.figure()
    
    
    # Stacked Line Plot - Shows feature profiles overlaid
    ax = plt.subplot(1, 1, 1)
    for i, sample_idx in enumerate(sample_indices):
        alpha = 0.7 - (i * 0.05)  # Gradually decrease opacity
        ax.plot(features[sample_idx], alpha=max(alpha, 0.2), linewidth=1.5)
    
    # Highlight the mean
    mean_features = np.mean(features, axis=0)
    ax.plot(mean_features, color='red', linewidth=2, label='Mean', alpha=0.9)
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Feature Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/feature_analysis.png')
    plt.close()

    # histogram of features values
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.hist(np.log10(np.abs(features.flatten()[np.abs(features.flatten()) >1e-40])), bins=50, density=True,
     color='salmon', edgecolor='black')
    ax.set_title(f'Features Distribution', fontweight='bold')
    ax.set_xlabel(r'$\log_{10}$ (Feature Value)')
    ax.set_ylabel('PDF')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_histogram.png')
    plt.close()

def hist_multidir(dirs, names, save_dir):
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    for dir, name in zip(dirs, names):
        features = np.load(f'{dir}/features.npy')
        flat_features = features.flatten()
        flat_features = flat_features[np.abs(flat_features) > 1e-40]
        ax.hist(np.log10(np.abs(flat_features)), bins=50, alpha=0.8, density=True,
        label=name, histtype='step')
    ax.set_title(f'Features Distribution', fontweight='bold')
    ax.set_xlabel(r'$\log_{10}$ (Feature Value)')
    ax.set_ylabel('PDF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/feature_histogram_compare.png')
    plt.close()

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    for dir, name in zip(dirs, names):
        features = np.load(f'{dir}/features.npy')
        features = features[np.abs(features) > 1e-8]
        ax.hist(np.log10(np.abs(features.flatten())), bins=50, alpha=0.8, density=True,
        label=name,  histtype='step')
    ax.set_title(f'Features Distribution', fontweight='bold')
    ax.set_xlabel(r'$\log_{10}$ (Feature Value)')
    ax.set_ylabel('PDF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_histogram_compare_zoom.png')
    plt.close()

def plot_umap(features, df, cols, save_dir):
    # Calculate grid size dynamically to ensure all cols fit
    n_plots = len(cols)
    n_rows = 1
    n_cols_grid = math.ceil(n_plots / n_rows) # Uses math.ceil to handle odd numbers
    
    # Create subplots
    fig, axis = plt.subplots(n_rows, n_cols_grid, figsize=(25, 12 * n_rows))
    axis = axis.flatten()
    
    # --- UMAP Calculation Logic (Same as before) ---
    if 'info_umap.csv' in os.listdir(save_dir):
        df = pd.read_csv(f'{save_dir}/info_umap.csv')
    else:
        print("calculating umap...")
        reducer_standard = umap.UMAP(
            n_components=2,
            random_state=1234,
        )
    
        umap_coords = reducer_standard.fit_transform(features)
        df['umap_x'] = umap_coords[:, 0]
        df['umap_y'] = umap_coords[:, 1]
        df.to_csv(f'{save_dir}/info_umap.csv', index=False)
        
    print("plotting umap...")
    
    # --- Plotting Loop ---
    for i, col in enumerate(cols):
        ax = axis[i]

        if 'teff' in col.lower():
            unit = 'K'
        else:
            unit = 'dex'
        
        # 1. Capture the scatter object (sc) to pass to colorbar
        sc = ax.scatter(df['umap_x'], df['umap_y'], c=df[col], cmap='viridis', s=15, alpha=0.8)
        
        # 2. Add Colorbar with font size adjustments
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(unit, fontsize=24)
        cbar.ax.tick_params(labelsize=24) 
        
        # 3. Set Titles and Labels with bigger fonts
        ax.set_title(col, fontsize=24, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=18, fontweight='bold')
        ax.set_ylabel('UMAP 2', fontsize=18, fontweight='bold')
        
        # Increase tick label size
        ax.tick_params(axis='both', which='major', labelsize=14)

    # Hide any unused subplots (if your grid is bigger than the number of cols)
    for j in range(i + 1, len(axis)):
        axis[j].axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/umaps.png')
    plt.close()
    
    

if __name__ == '__main__':
    
    log_dir_v2 = '/home/ilay.kamai/work/TalkingLatents/logs/features_v2/'
    df_v2 = pd.read_csv('/home/ilay.kamai/work/TalkingLatents/logs/features_v2/info.csv')
    df_v2.rename(columns={'feh': 'FeH', 'teff': 'Teff'}, inplace=True)
    # df_v2['Teff'] = df_v2['Teff'] * (7500 - 3000) + 3000 
    log_dir = '/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/'
    df = pd.read_csv('/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/info.csv')
    # df['Teff'] = df['Teff'] * 5778
    log_dir_multimodal = '/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29-multimodal/'
    df_multimodal = pd.read_csv('/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29-multimodal/info_full_multimodal.csv')
    hist_multidir([log_dir_v2, log_dir, log_dir_multimodal], ['Dense', 'Sparse', 'Multimodal'], 'figs')
    
    for i, (log_d, df) in enumerate(zip([log_dir, log_dir_v2], [df, df_v2])):
        if i == 0:
            features = np.load(f'{log_dir}/features.npy')
            fig_dir = f'{log_dir}/figs' 
        else:
            features = np.load(f'{log_d}/features_cls.npy')
            fig_dir = f'{log_d}/figs_cls'
        print(df.shape, features.shape)
        plot_features(features, fig_dir)
        plot_umap(features, df, ['FeH', 'logg', 'Teff'], fig_dir)