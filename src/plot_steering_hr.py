
import argparse
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path
import pandas as pd
os.system("pip install isochrones")
from isochrones.mist import MIST_Isochrone

# Add src to path to import extraction logic
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from src.analyze_followup_stellar_type import extract_stellar_params_from_text
except ImportError:
    # Fallback to local definition if import fails (or copy-paste regex)
    print("Warning: Could not import extract_stellar_params_from_text. Using local fallback.")
    import re
    def extract_stellar_params_from_text(text: str):
        params = {'Teff': None, 'logg': None}
        if not text: return params
        # Teff
        teff_match = re.search(r'(?:teff|temperature).*?(\d{3,5})\s*k', text.lower())
        if teff_match: params['Teff'] = float(teff_match.group(1))
        # logg
        logg_match = re.search(r'(?:logg|log\s*g).*?([0-5]\s*\.\s*\d{1,2})', text.lower())
        if logg_match: params['logg'] = float(logg_match.group(1).replace(' ', ''))
        return params

def dwarf_giant_thresh_ciardi(teff):
    """
    Condition for dwarfs (Main Sequence) vs Giants.
    Returns True if Dwarf, False if Giant.
    """
    thresh = np.zeros_like(teff)
    
    # Vectorized implementation
    mask_hot = teff >= 6000
    mask_cool = teff <= 4250
    mask_mid = ~(mask_hot | mask_cool)
    
    thresh[mask_hot] = 3.5
    thresh[mask_cool] = 4.0
    thresh[mask_mid] = 5.2 - (2.8e-4 * teff[mask_mid])
    
    return thresh

# Readable Titles
CONCEPT_TITLES = {
    'evolution_giant_minus_dwarf': 'Dwarf $\\to$ Giant',
    'age_old_minus_young': 'Young $\\to$ Old',
    'feh_rich_minus_poor': 'Metal Poor $\\to$ Rich',
    'mass_high_minus_low': 'Low Mass $\\to$ High Mass',
    'teff_hot_minus_cool': 'Cool $\\to$ Hot',
    'logg_high_minus_low': 'Low logg $\\to$ High logg',
}

def plot_hr_diagrams(json_file, output_dir, tiny_plot=False):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Handle list or dict format
    if isinstance(data, dict) and 'samples' in data:
        samples = data['samples']
    elif isinstance(data, list):
        # Flatten if it's a list of experiment results, or just take the list of samples
        if data and 'samples' in data[0]:
             samples = []
             for d in data: samples.extend(d['samples'])
        else:
             samples = data
    else:
        raise ValueError("Unknown JSON format")

    # Group by obsid
    grouped = defaultdict(list)
    for sample in samples:
        obsid = sample.get('obsid')
        if obsid is None: continue
        grouped[obsid].append(sample)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(grouped)} unique stars.")

    # Collect background (all samples)
    bg_csv = pd.read_csv('/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/info_full.csv')
    bg_teff = bg_csv['Teff'].values * 5778  
    bg_logg = bg_csv['logg'].values
    bg_feh = bg_csv['FeH'].values
    
    # Create lookup for star properties (Age, FeH)
    # Ensure obsid is same type as in samples (usually string or int)
    # samples data loaded from json usually has keys as strings or ints depending.
    # We'll try to match types.
    bg_csv_unique = bg_csv.drop_duplicates(subset='obsid')
    bg_info = bg_csv_unique.set_index('obsid').to_dict(orient='index')
    
    # Initialize MIST
    print("Initializing MIST Isochrones...")
    mist = MIST_Isochrone()

    sorted_teff = np.sort(bg_teff)
    thresh_logg = dwarf_giant_thresh_ciardi(sorted_teff)
            
    print(f"Background distribution has {len(bg_teff)} points.")

    for obsid, entries in grouped.items():
        print(f"Processing obsid {obsid} ({len(entries)} variations)...")
        
        # Organize by Concept -> dictionary of {alpha: params}
        concept_data = defaultdict(lambda: {'alphas': [], 'teffs': [], 'loggs': [], 'true_teff': None, 'true_logg': None})
        
        dataset_target_str = None

        for entry in entries:
            concept = entry.get('steering_concept')
            alpha = entry.get('steering_alpha', 0.0)
            
            # If concept is None (e.g. baseline), treat it as 'Baseline' or merge into all?
            # Typically logic: baseline (alpha=0) is universal or belongs to "None".
            # But in our generation loop, alpha=0 is generated under a concept loop too (concept, 0).
            # If concept is explicitly labeled:
            if concept is None: 
                # Should not happen with new logic, but if so call it 'Baseline'
                concept = 'Baseline'
            
            # Extract params
            text = entry.get('model_answer', '')
            
            # If model_answer is empty, look into follow_up_answers
            if not text or not text.strip():
                follow_ups = entry.get('follow_up_answers', [])
                # Look for the stellar type follow-up which contains Teff/logg info
                for fa in follow_ups:
                     if fa.get('type') == 'star_type':
                         text = fa.get('answer', '')
                         break
            
            extracted = extract_stellar_params_from_text(text)
            
            # Try getting True Params
            true_params = entry.get('stellar_params', {})
            t_teff = true_params.get('Teff')
            t_logg = true_params.get('logg')
            
            if t_teff is not None and t_logg is not None:
                concept_data[concept]['true_teff'] = float(t_teff)
                concept_data[concept]['true_logg'] = float(t_logg)
            
            # If extraction successful
            if extracted['Teff'] is not None and extracted['logg'] is not None:
                concept_data[concept]['alphas'].append(alpha)
                concept_data[concept]['teffs'].append(extracted['Teff'])
                concept_data[concept]['loggs'].append(extracted['logg'])
            
            if dataset_target_str is None:
                dataset_target_str = entry.get('dataset_target_answer', 'N/A')

        # Plot
        print(concept_data.keys())
        print(concept_data['age_old_minus_young'].keys())
        print(concept_data['age_old_minus_young']['alphas'])
        unique_concepts = [c for c in concept_data.keys() if concept_data[c]['alphas']]
        
        if tiny_plot:
            allowed = ['evolution_giant_minus_dwarf', 'age_old_minus_young']
        else:
            allowed = ['teff_hot_minus_cool', 'logg_high_minus_low',
                       'feh_rich_minus_poor', 'mass_high_minus_low']
        unique_concepts = [c for c in unique_concepts if c in allowed]
        if not unique_concepts:
            print(f"  No valid extracted parameters for obsid {obsid}. Skipping.")
            continue

        n_concepts = len(unique_concepts)
        cols = min(n_concepts, 2)
        rows = (n_concepts + cols - 1) // cols
        
        # Increase figure size
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
        # fig.suptitle(f"Steering Effects for Star {obsid}", fontsize=16)
        
        # Flatten axes for easy iteration
        axes_flat = axes.flatten()

        for idx, concept in enumerate(unique_concepts):
            print(idx, concept)
            ax = axes_flat[idx]
            data = concept_data[concept]
            
            alphas = np.array(data['alphas'])
            teffs = np.array(data['teffs'])
            loggs = np.array(data['loggs'])

            # bg_color = bg_feh if not tiny_plot else 'gray'
            bg_color = 'gray'
            
            # Background
            if len(bg_teff) > 0:
                ax.scatter(bg_teff, bg_logg, cmap='viridis', c=bg_color, alpha=0.3, s=10, label='All Samples')
                
                # Dwarf/Giant Threshold
                ax.plot(sorted_teff, thresh_logg, 'k--', alpha=0.8, zorder=1)
                
            # Plot Isochrone if Age/FeH available
            # Try finding info for this obsid
            star_props = None
            try:
                # Try int
                star_props = bg_info.get(int(obsid))
                if not star_props:
                    star_props = bg_info.get(str(obsid))
            except:
                pass
                
            if star_props and 'Age' in star_props and 'FeH' in star_props:
                age_gyr = star_props['Age']
                feh_star = star_props['FeH']
                if age_gyr > 0:
                    # Generate continuous age factors (e.g. 0.1x to 5x, or fixed range around true age)
                    # Let's go from e.g. 0.2 * age to 5 * age, or clamped to reasonable Gyr
                    min_age = max(0.1, age_gyr * 0.2)
                    max_age = min(13.5, age_gyr * 5.0)
                    
                    if min_age < max_age:
                        n_iso = 30
                        # Log spacing
                        ages_to_plot = np.logspace(np.log10(min_age), np.log10(max_age), n_iso)
                        
                        # Add the true age if not close
                        if not np.any(np.isclose(ages_to_plot, age_gyr, rtol=0.05)):
                             ages_to_plot = np.sort(np.append(ages_to_plot, age_gyr))
                        
                        # Colormap setup
                        log_ages = np.log10(ages_to_plot * 1e9)
                        norm = plt.Normalize(vmin=log_ages.min(), vmax=log_ages.max())
                        cmap = plt.get_cmap('autumn_r') # Yellow to Red? Or viridis? 'autumn_r' puts yellow (young) to red (old) which makes sense? 
                                                        # Actually 'jet' or 'turbo' gives good contrast. Let's use 'viridis' or 'plasma'.
                                                        # User asked for continuous colors.
                        cmap = plt.get_cmap('plasma') 
                        
                        youngest_to = None
                        oldest_to = None
                        youngest_age_val = float('inf')
                        oldest_age_val = float('-inf')

                        for current_age in ages_to_plot:
                            if current_age < 0.1: continue
                            if current_age > 14: continue
                            
                            try:
                                log_age = np.log10(current_age * 1e9)
                                iso = mist.isochrone(log_age, feh_star)
                                
                                # Isochrone gives logTeff, logg
                                iso_teff = 10**iso['logTeff']
                                valid_teff = iso_teff < 8000
                                iso_logg = iso['logg'][valid_teff]
                                iso_teff = iso_teff[valid_teff]
                                valid_logg = iso_logg > 1
                                iso_logg = iso_logg[valid_logg]
                                iso_teff = iso_teff[valid_logg]

                                is_true_age = np.isclose(current_age, age_gyr, rtol=0.01)
                                
                                # Color
                                color = cmap(norm(log_age))
                                lw = 2.5 if is_true_age else 1.0
                                alpha_iso = 0.9 if is_true_age else 0.4
                                zorder = 4 if is_true_age else 2
                                
                                # label = f'Isochrone ({current_age:.1f} Gyr)' if is_true_age else None
                                
                                ax.plot(iso_teff, iso_logg, linewidth=lw, color=color, linestyle='-', alpha=alpha_iso, zorder=zorder)
                                
                                # Find Turn-Off (max Teff)
                                if len(iso_teff) > 0:
                                    max_teff_idx = np.argmax(iso_teff)
                                    to_point = (iso_teff[max_teff_idx], iso_logg[max_teff_idx])
                                    
                                    if current_age < youngest_age_val:
                                        youngest_age_val = current_age
                                        youngest_to = to_point
                                    if current_age > oldest_age_val:
                                        oldest_age_val = current_age
                                        oldest_to = to_point
                                
                            except Exception as e:
                                # print(f"  Failed to generate isochrone for obsid {obsid} at age {current_age}: {e}")
                                pass
                        
                        # # Draw Age Arrow
                        # if youngest_to and oldest_to:
                        #     ax.annotate(
                        #         "Age", 
                        #         xy=oldest_to, xycoords='data',
                        #         xytext=youngest_to, textcoords='data',
                        #         arrowprops=dict(arrowstyle="->", color='black', lw=2, connectionstyle="arc3,rad=-0.1"),
                        #         fontsize=12, fontweight='bold', ha='center', va='center',
                        #         zorder=20
                        #     )

            
            # Plot True Value
            if data['true_teff'] is not None:
                ax.scatter(data['true_teff'], data['true_logg'], c='black', marker='*', s=200, label='True Params', zorder=10)
            
            # Plot Steered Values
            # Use 'coolwarm' colormap centered at 0? 
            # Alpha typically -5 to 5.
            sc = ax.scatter(teffs, loggs, c=alphas, cmap='coolwarm', s=100, edgecolor='k', zorder=5)
            
            # Add trajectory line (sort by alpha)
            sort_idx = np.argsort(alphas)
            ax.plot(teffs[sort_idx], loggs[sort_idx], 'k--', alpha=0.3, zorder=1)

            # Labels and Titles
            # Use readable title if available
            title_text = CONCEPT_TITLES.get(concept, concept)
            ax.set_title(title_text, fontsize=14)
            # ax.set_title(f"Concept: {concept}")
            
            ax.set_xlabel("Teff (K)")
            ax.set_ylabel("log(g)")
            
            # Invert axes for HR diagram convention
            ax.invert_xaxis() # High Teff on left
            ax.invert_yaxis() # Low logg (Bright/Giant) on top
            
            # Colorbar
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label(r"$\alpha$ (Steering Strength)")
            
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide unused subplots
        for j in range(len(unique_concepts), len(axes_flat)):
            axes_flat[j].axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
        save_file = output_path / f"sample_{obsid}_steering.png"
        plt.savefig(save_file)
        plt.close(fig)
        print(f"  Saved plot to {save_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot HR diagrams for steering results.")
    parser.add_argument('json_file', type=str, help="Path to steering results JSON file")
    parser.add_argument('--output_dir', type=str, default="plots/steering_analysis", help="Directory to save plots")
    parser.add_argument('--tiny_plot', action='store_true', help="Only plot Giant-Dwarf and Age concepts")
    
    args = parser.parse_args()
    
    plot_hr_diagrams(args.json_file, args.output_dir, args.tiny_plot)
