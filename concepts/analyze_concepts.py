
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['font.size'] = 12

def cosine_similarity(v1, v2):
    return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))

def plot_cosine_matrix(concepts, save_path):
    keys = list(concepts.keys())
    n = len(keys)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            sim = cosine_similarity(concepts[keys[i]], concepts[keys[j]]).item()
            matrix[i, j] = sim
            
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=keys, yticklabels=keys, vmin=-1, vmax=1)
    plt.title("Cosine Similarity between Concept Directions")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved cosine matrix to {save_path}")



def project_latents(latents, direction):
    # Normalize direction
    direction = direction / torch.norm(direction)
    # Project: x . d
    return torch.matmul(latents, direction)

def plot_histograms(latents, params, concepts, numeric_bounds, save_dir):
    # params: [Teff, logg, FeH] (normalized)
    
    # Define ground truth masks again for validation visualization
    teff_norm = params[:, 0]
    logg_norm = params[:, 1]
    feh_norm = params[:, 2]
    
    def denorm(val, key):
        low, high = numeric_bounds[key]
        return val * (high - low) + low

    teff = denorm(teff_norm, 'Teff')
    logg = denorm(logg_norm, 'logg')
    feh = denorm(feh_norm, 'FeH')
    
    # Helper for Evolution logic (same as extraction)
    def is_dwarf_func(t, l):
        thresh = torch.zeros_like(t)
        mask_hot = t >= 6000
        mask_cool = t <= 4250
        mask_mid = ~(mask_hot | mask_cool)
        thresh[mask_hot] = 3.5
        thresh[mask_cool] = 4.0
        thresh[mask_mid] = 5.2 - (2.8e-4 * t[mask_mid])
        return l >= thresh

    # Define groups for coloring
    groups = {
        'evolution_giant_minus_dwarf': ('Evolution', is_dwarf_func(teff, logg), ['Giant', 'Dwarf']), # False=Giant
        'teff_hot_minus_cool': ('Teff', teff > 5250, ['Hotter', 'Cooler']), 
        'logg_high_minus_low': ('logg', logg > 3.5, ['High logg', 'Low logg']),
        'feh_rich_minus_poor': ('FeH', feh > 0.0, ['Metal-Rich', 'Metal-Poor'])
    }

    for name, direction in concepts.items():
        print(name, direction.shape)
        projections = project_latents(latents, direction).numpy()
        
        # Determine labels for plotting
        if name in groups:
            concept_type, mask, labels = groups[name]
            # mask is logical vector. 
            # For Evolution: mask is True for Dwarf. But direction is Giant - Dwarf.
            # So Dwarf (True) should be negative side ideally, Giant (False) positive.
            # Let's plot distribution.
            
            plt.figure(figsize=(10, 6))
            
            # Split data
            if name == 'evolution_giant_minus_dwarf':
                # mask is is_dwarf. 
                data_neg = projections[mask] # Dwarf
                data_pos = projections[~mask] # Giant
                label_neg = 'Dwarf'
                label_pos = 'Giant'
            else:
                # Generic split based on simple threshold for viz
                data_pos = projections[mask]
                data_neg = projections[~mask]
                label_pos = labels[0]
                label_neg = labels[1]

            sns.histplot(data=data_pos, color="red", label=label_pos, kde=True, stat="density", alpha=0.5)
            sns.histplot(data=data_neg, color="blue", label=label_neg, kde=True, stat="density", alpha=0.5)
            
            plt.title(f"Projection onto {name}")
            plt.xlabel("Projection Score")
            plt.legend()
            
            fname = os.path.join(save_dir, f"hist_{name}.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved histogram to {fname}")

def plot_2d_projections(latents, concepts, save_dir):
    # Plot Teff vs logg directions
    if 'teff_hot_minus_cool' in concepts and 'logg_high_minus_low' in concepts:
        x_dir = concepts['teff_hot_minus_cool']
        y_dir = concepts['logg_high_minus_low']
        
        x_proj = project_latents(latents, x_dir).numpy()
        y_proj = project_latents(latents, y_dir).numpy()
        
        plt.figure(figsize=(10, 8))
        plt.scatter(x_proj, y_proj, alpha=0.5, s=1)
        plt.xlabel("Teff Direction (Hot - Cool)")
        plt.ylabel("logg Direction (High - Low)")
        plt.title("Latents projected onto Teff and logg concept directions")
        
        fname = os.path.join(save_dir, "scatter_teff_vs_logg.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved scatter plot to {fname}")
        
    # Plot HR Diagram equivalent (Teff vs Giant-Dwarf?)
    if 'teff_hot_minus_cool' in concepts and 'evolution_giant_minus_dwarf' in concepts:
        x_dir = concepts['teff_hot_minus_cool']
        y_dir = concepts['evolution_giant_minus_dwarf']
        
        x_proj = project_latents(latents, x_dir).numpy()
        y_proj = project_latents(latents, y_dir).numpy()
        
        plt.figure(figsize=(10, 8))
        plt.scatter(x_proj, y_proj, alpha=0.5, s=1)
        plt.xlabel("Teff Direction (Hot - Cool)")
        plt.ylabel("Evolution Direction (Giant - Dwarf)")
        plt.title("Latents projected onto Teff and Evolution directions")
        if 'teff_hot_minus_cool' in concepts: # Flip x axis like HR diagram
             plt.gca().invert_xaxis()
        
        fname = os.path.join(save_dir, "scatter_hr_diagram_proxy.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved HR-proxy scatter plot to {fname}")


def main():
    data_path = '/home/ilay.kamai/work/TalkingLatents/concepts/concept_data.pt'
    output_dir = '/home/ilay.kamai/work/TalkingLatents/concepts/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run extract_concepts.py first.")
        return

    data = torch.load(data_path)
    latents = data['latents']
    params = data['params']
    concepts = data['concepts']
    
    # Hardcoded bounds from dataset.py
    numeric_bounds = {
            'Teff': (3000.0, 7500.0),
            'logg': (0.0, 5.0),
            'FeH': (-3.0, 0.5),
        }

    print("Generating Cosine Similarity Matrix...")
    plot_cosine_matrix(concepts, os.path.join(output_dir, 'cosine_similarity.png'))
    
    print("Generating Histograms...")
    plot_histograms(latents, params, concepts, numeric_bounds, output_dir)
    
    print("Generating 2D Projections...")
    plot_2d_projections(latents, concepts, output_dir)
    
    print("Analysis complete.")

if __name__ == "__main__":
    main()
