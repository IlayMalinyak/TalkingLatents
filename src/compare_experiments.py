"""
Compare multiple experimental results for stellar type and parameter prediction.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from src.snr_lookup import load_snr_lookup

# Standard spectral types list for validation/ordering
from typing import Dict, List, Optional, Tuple, Any
import re
import textwrap
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

# ==============================================================================
# Extraction Logic (Copied from analyze_followup_stellar_type.py for standalone usage)
# ==============================================================================

# ==============================================================================
# Extraction Logic
# ==============================================================================

from src.analyze_followup_stellar_type import (
    extract_stellar_type_from_text,
    extract_spectral_type_from_text,
    extract_full_subclass_from_text,
    extract_stellar_params_from_text,
    is_valid_text,
    extract_value_from_text,
    SPECTRAL_TYPES,
)
# ==============================================================================
# Analysis Logic
# ==============================================================================

def process_experiment(json_path: Path, exp_name: str, snr_map: Dict[str, float]) -> Tuple[Dict, Dict]:
    """Process a single experiment file and return stats and param data."""
    print(f"Processing {exp_name} from {json_path}")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return {}, {}

    if isinstance(data, list):
        if len(data) > 0 and 'samples' in data[0]:
             samples = data[0]['samples']
        elif len(data) > 0 and ('followup_qa' in data[0] or 'follow_up_answers' in data[0]):
             samples = data
        else:
             print(f"Unknown list format in {json_path}")
             return {}, {}
    elif isinstance(data, dict) and 'samples' in data:
        samples = data['samples']
    else:
        print(f"Unexpected JSON format in {json_path}")
        return {}, {}

    stats = {
        'total_questions': 0,
        'total_stellar_type_questions': 0,
        'valid_responses': 0,
        'evol_class_correct': 0,
        'spectral_letter_correct': 0,
        'full_subclass_correct': 0,
        'correct_matches': 0,
    }
    
    param_data = {
        'true_Teff': [], 'pred_Teff': [],
        'true_logg': [], 'pred_logg': [],
        'true_FeH': [], 'pred_FeH': [],
        'true_Lstar': [], 'pred_Lstar': [],
        'true_Mstar': [], 'pred_Mstar': [],
        'true_Age': [], 'pred_Age': [],
        'snr_Teff': [],
        'snr_logg': [],
        'snr_FeH': [],
        'snr_Lstar': [],
        'snr_Mstar': [],
        'snr_Age': [],
    }

    for sample in samples:
        followup_qa = sample.get('followup_qa', sample.get('follow_up_answers', []))
        stellar_params = sample.get('stellar_params', {})
        
        # Get subclass
        subclass = None
        stellar_data = sample.get('stellar_data')
        if stellar_data and isinstance(stellar_data, dict):
            subclass = stellar_data.get('subclass')
        if not subclass:
            subclass = sample.get('subclass', None)

        for qa in followup_qa:
            question = qa.get('question', '')
            q_type = qa.get('type', '')
            
            # Identify question type
            # Use 'type' field from JSON if available (matches analyze_followup_stellar_type.py)
            if q_type:
                is_stellar_type = 'type' in q_type.lower() and 'luminosity' not in q_type.lower() and 'mass' not in q_type.lower()
                # Correction: analyze_followup_stellar_type.py uses 'type' in q_type.lower(). 
                # JSON has "star_type", "luminosity_followup", "mass_followup".
                # So "type" matches "star_type", but strictly it assumes others don't have "type".
                # Actually "star_type" has "type". "luminosity_followup" has no "type".
                
                # Let's match analyze_followup EXACTLY:
                # is_stellar_type_q = 'type' in q_type.lower()
                is_stellar_type = 'type' in q_type.lower() and 'star' in q_type.lower() 
                # Wait, analyze_followup says: is_stellar_type_q = 'type' in q_type.lower()
                # But if q_type is "star_type", it works.
                # If q_type is "luminosity_followup", "type" is NOT in it.
                # So `is_stellar_type = 'type' in q_type.lower()` is correct for "star_type".
                
                is_stellar_type = 'type' in q_type.lower() or 'classification' in q_type.lower()
                is_lum = 'luminosity' in q_type.lower()
                is_mass = 'mass' in q_type.lower()
                is_age = 'age' in q_type.lower()
            else:
                # Fallback to question text if 'type' key is missing
                is_lum = 'luminosity' in question.lower()
                is_mass = 'mass' in question.lower()
                is_age = 'age' in question.lower()
                
                # Filter for stellar type questions
                is_stellar_type = any(k in question.lower() for k in ['stellar classification', 'stellar type', 'what is the stellar', 'what stellar classification', 'what is the type'])

            is_relevant = is_stellar_type or is_lum or is_mass or is_age
            
            if not is_relevant:
                continue
                
            stats['total_questions'] += 1
            generated = qa.get('generated_response', qa.get('answer', ''))
            
            if not is_valid_text(generated):
                continue
                
            stats['valid_responses'] += 1
            
            # Expected values
            expected_evol_class = None
            if stellar_params:
                teff = stellar_params.get('Teff', 0)
                logg = stellar_params.get('logg', 0)
                if logg < 3.5:
                    expected_evol_class = "supergiant" if logg < 1.0 else "giant"
                elif logg > 4.5:
                    expected_evol_class = "white dwarf"
                else:
                    expected_evol_class = "dwarf" if 3.5 <= logg <= 4.5 else "subgiant"
            
            expected_spectral = None
            if subclass and isinstance(subclass, str) and len(subclass) > 0:
                first_char = subclass[0].upper()
                if first_char in SPECTRAL_TYPES:
                    expected_spectral = first_char
            
            expected_subclass = subclass.upper() if subclass and isinstance(subclass, str) else None

            # Extracted values
            generated_evol_type = extract_stellar_type_from_text(generated)
            generated_subclass = extract_full_subclass_from_text(generated)
            generated_spectral_candidates = set()
            if generated_subclass and generated_subclass[0] in SPECTRAL_TYPES:
                generated_spectral_candidates.add(generated_subclass[0])
            
            # Extract text-based mentions (now a list)
            text_spectrals = extract_spectral_type_from_text(generated)
            if text_spectrals:
                generated_spectral_candidates.update(text_spectrals)
            
            # Convert to comma-separated string
            generated_spectral = ",".join(sorted(list(generated_spectral_candidates))) if generated_spectral_candidates else None
            
            extracted_params = extract_stellar_params_from_text(generated)

            # Matches
            # NOTE: analyze_followup_stellar_type.py counts matches across ALL questions (even mass/lum)
            # if they contain the correct spectral/evol/subclass info. 
            # It then divides by total_stellar_type_questions (200), which can yield > 100% accuracy.
            # We replicate this logic here to match the analysis script results exactly.
            evol_match = (expected_evol_class and generated_evol_type and expected_evol_class == generated_evol_type)
            spectral_match = False
            if expected_spectral is not None and generated_spectral is not None:
                candidates = generated_spectral.split(',')
                spectral_match = expected_spectral in candidates
            subclass_match = (expected_subclass and generated_subclass and expected_subclass == generated_subclass)

            if evol_match: stats['evol_class_correct'] += 1
            if spectral_match: stats['spectral_letter_correct'] += 1
            if subclass_match: stats['full_subclass_correct'] += 1
            if evol_match or spectral_match or subclass_match: stats['correct_matches'] += 1
            
            if is_stellar_type:
                stats['total_stellar_type_questions'] += 1
            # Params
            # Get SNR (try JSON first, then lookup)
            snr = qa.get('snr')
            if snr is None:
                # Fallback lookup
                obsid = sample.get('obsid')
                if obsid:
                    snr = snr_map.get(str(obsid))
            
            if snr is None:
                snr = 0.0
            

            for p in ['Teff', 'logg', 'FeH']:
                if stellar_params.get(p) is not None and extracted_params[p] is not None:
                    if p == 'Teff' and extracted_params[p] < 1000:
                        continue
                    param_data[f'true_{p}'].append(stellar_params[p])
                    param_data[f'pred_{p}'].append(extracted_params[p])
                    param_data[f'snr_{p}'].append(snr)
                    
            if is_lum:
                pred_L = extract_value_from_text(generated, unit='Lsun')
                true_L = qa.get('true_value')
                if true_L is not None and pred_L is not None:
                    if pred_L < 1000:
                        # Linear vs Linear Comparison (as requested)
                        param_data['true_Lstar'].append(float(true_L))
                        param_data['pred_Lstar'].append(pred_L)
                        param_data['snr_Lstar'].append(snr)
            
            if is_mass:
                pred_M = extract_value_from_text(generated, unit='Msun')
                true_M = qa.get('true_value')
                if true_M is not None and pred_M is not None:
                    param_data['true_Mstar'].append(float(true_M))
                    param_data['pred_Mstar'].append(pred_M)
                    param_data['snr_Mstar'].append(snr)
            
            if is_age:
                pred_Age = extract_value_from_text(generated, unit='Gyr')
                true_Age = qa.get('true_value')
                if true_Age is not None and pred_Age is not None:
                    param_data['true_Age'].append(float(true_Age))
                    param_data['pred_Age'].append(pred_Age)
                    param_data['snr_Age'].append(snr)
    print("total questions: ", stats['total_questions'])
    print("valid responses: ", stats['valid_responses'])
    print("correct matches: ", stats['correct_matches'])
    print("correct subclass: ", stats['full_subclass_correct'])
    print("correct spectral letter: ", stats['spectral_letter_correct'])
    print("correct evolution class: ", stats['evol_class_correct'])

    # Calculate accuracies
    total = stats['total_questions']
    total_st = stats['total_stellar_type_questions']
    print("total_st: ", total_st)
    if total > 0:
        if total_st > 0:
            stats['accuracy'] = (stats['correct_matches'] / total_st) * 100
            stats['evol_acc'] = (stats['evol_class_correct'] / total_st) * 100
            stats['spectral_acc'] = (stats['spectral_letter_correct'] / total_st) * 100
            stats['subclass_acc'] = (stats['full_subclass_correct'] / total_st) * 100
        else:
            stats['accuracy'] = 0
            stats['evol_acc'] = 0
            stats['spectral_acc'] = 0
            stats['subclass_acc'] = 0
    else:
        stats['accuracy'] = stats['evol_acc'] = stats['spectral_acc'] = stats['subclass_acc'] = 0
        
    return stats, param_data

def find_json_file(directory: Path) -> Optional[Path]:
    """Find the result JSON file in the directory."""
    candidates = ['followup_comparison.json', 'follow_up_answers.json', 'late_fusion_followups.json']
    for cand in candidates:
        p = directory / cand
        if p.exists():
            return p
    
    # Try globbing
    jsons = list(directory.glob('*followup*.json'))
    if jsons:
        return jsons[0]
        
    return None

def main():
    parser = argparse.ArgumentParser(description="Compare multiple stellar analysis experiments.")
    parser.add_argument('--dirs', type=str, nargs='+', required=True, help='Paths to experiment directories')
    parser.add_argument('--names', type=str, nargs='+', default=None, help='Names for experiments (optional)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for plots')
    
    args = parser.parse_args()
    
    dirs = [Path(d) for d in args.dirs]
    names = args.names if args.names else [d.name for d in dirs]
    
    if len(names) != len(dirs):
        print("Warning: Number of names does not match number of directories. Using directory names for missing ones.")
        names = names[:len(dirs)] + [d.name for d in dirs[len(names):]]

    out_dir = Path(args.output_dir) if args.output_dir else dirs[0].parent / 'comparison_results'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_stats = []
    all_params = []
    valid_names = []
    
    # Load SNR map once
    snr_map = load_snr_lookup()
    
    for d, name in zip(dirs, names):
        json_file = find_json_file(d)
        if not json_file:
            print(f"Could not find result JSON in {d}, skipping.")
            continue
            
        stats, params = process_experiment(json_file, name, snr_map)
        all_stats.append(stats)
        all_params.append(params)
        valid_names.append(name)
        
    if not all_stats:
        print("No valid experiments found.")
        return

    # Plotting
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12
    
    # 1. Spectral Type Accuracy Comparison (Only Spectral Type)
    fig, ax = plt.subplots(figsize=(max(8, len(valid_names)*1.5), 6))
    
    x = np.arange(len(valid_names))
    width = 0.6
    
    spectral_accs = [s.get('spectral_acc', 0) for s in all_stats]
    
    bars = ax.bar(x, spectral_accs, width, color='steelblue', alpha=0.8)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Spectral Type Classification Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_names, rotation=45, ha='right')
    ax.set_ylim(0, 105)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()
    plt.savefig(out_dir / 'spectral_accuracy_comparison.png', dpi=300)
    print(f"Saved spectral accuracy plot to {out_dir / 'spectral_accuracy_comparison.png'}")
    # subclass bar plot
    fig, ax = plt.subplots(figsize=(max(8, len(valid_names)*1.5), 6))

    subclass_accs = [s.get('subclass_acc', 0) for s in all_stats]
    
    bars = ax.bar(x, subclass_accs, width, color='steelblue', alpha=0.8)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Subclass Classification Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_names, rotation=45, ha='right')
    ax.set_ylim(0, 105)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()
    plt.savefig(out_dir / 'subclass_accuracy_comparison.png', dpi=300)
    print(f"Saved subclass accuracy plot to {out_dir / 'subclass_accuracy_comparison.png'}")
    
    # 2. Parameter MAE Comparison
    param_names = ['Teff', 'logg', 'FeH', 'Lstar', 'Mstar', 'Age']
    mae_data = {p: [] for p in param_names}
    medae_data = {p: [] for p in param_names}
    
    for params in all_params:
        for p in param_names:
            true_vals = params.get(f'true_{p}', [])
            pred_vals = params.get(f'pred_{p}', [])
            if true_vals and pred_vals:
                mae = mean_absolute_error(true_vals, pred_vals)
                medae = median_absolute_error(true_vals, pred_vals)
                mae_data[p].append(mae)
                medae_data[p].append(medae)
            else:
                mae_data[p].append(0) # Or NaN? 0 for now to avoid plotting errors
                medae_data[p].append(0)

    fig, axes = plt.subplots(1, len(param_names), figsize=(max(20, len(valid_names)*4), 5))
    fig.suptitle('Parameter Prediction MAE (Lower is Better)', fontsize=14)
    
    for idx, p in enumerate(param_names):
        ax = axes[idx]
        vals = mae_data[p]
        
        bars = ax.bar(x, vals, width, color='coral', alpha=0.8)
        
        ax.set_title(f'{p} MAE')
        ax.set_xticks(x)
        ax.set_xticklabels(valid_names, rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            else:
                 ax.text(bar.get_x() + bar.get_width()/2., 0,
                        'N/A', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / 'parameter_mae_comparison.png', dpi=300)
    print(f"Saved parameter MAE plot to {out_dir / 'parameter_mae_comparison.png'}")

    # 3. Parameter MedAE Comparison
    fig, axes = plt.subplots(1, len(param_names), figsize=(max(20, len(valid_names)*4), 5))
    fig.suptitle('Parameter Prediction MedAE (Lower is Better)', fontsize=14)
    
    for idx, p in enumerate(param_names):
        ax = axes[idx]
        vals = medae_data[p]
        
        bars = ax.bar(x, vals, width, color='mediumpurple', alpha=0.8)
        
        ax.set_title(f'{p} MedAE')
        ax.set_xticks(x)
        ax.set_xticklabels(valid_names, rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            else:
                 ax.text(bar.get_x() + bar.get_width()/2., 0,
                        'N/A', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / 'parameter_medae_comparison.png', dpi=300)
    print(f"Saved parameter MedAE plot to {out_dir / 'parameter_medae_comparison.png'}")

    # 3. Parameter Scatter Plots
    def plot_scatter_grid(param_subset, filename_suffix, title_suffix):
        n_params = len(param_subset)
        if n_params == 0:
            return

        cols = n_params
        # If we have only 1 parameter, we might just want 1 column. 
        # But generally we have N cols = N params.
        
        n_exps = len(valid_names)
        
        # Adjust figure size based on number of params and experiments
        # width ~ 5 per param, height ~ 4 per experiment
        fig, axes = plt.subplots(n_exps, cols, figsize=(5 * cols, 4 * n_exps))
        
        # Ensure axes is always 2D array [exp, param]
        if n_exps == 1 and cols == 1:
            axes = np.array([[axes]])
        elif n_exps == 1:
            axes = np.array([axes]) # Shape (1, cols)
        elif cols == 1:
            axes = np.array([[ax] for ax in axes]) # Shape (n_exps, 1) -- wait, subplots returns 1D array if cols=1? 
                                                  # Actually: subplots(2,1) -> array([ax1, ax2]) if default squeeze=True
                                                  # subplots(1,2) -> array([ax1, ax2])
                                                  # So let's just be careful with squeezing.
        
        # Easier: Just iterate flat if we can, but we need grid structure.
        # Let's fix squeeze=False so it's always 2D
        plt.close(fig) # close the one we just made to restart with squeeze=False
        fig, axes = plt.subplots(n_exps, cols, figsize=(5 * cols, 4 * n_exps), squeeze=False)

        fig.suptitle(f'Parameter Prediction Scatter Plots - {title_suffix}', fontsize=16)

        for i, name in enumerate(valid_names):
            params = all_params[i]
            for j, p in enumerate(param_subset):
                ax = axes[i, j]
                true_vals = params.get(f'true_{p}', [])
                pred_vals = params.get(f'pred_{p}', [])

                if true_vals and pred_vals:
                    # Scatter plot with SNR coloring (placeholder color for now as per original code)
                    sc = ax.scatter(true_vals, pred_vals, c='blue', alpha=0.6)
                    
                    min_v = min(min(true_vals), min(pred_vals))
                    max_v = max(max(true_vals), max(pred_vals))
                    
                    # Ensure decent padding
                    range_v = max_v - min_v
                    if range_v == 0: range_v = 1
                    
                    # Plot diagonal
                    ax.plot([min_v, max_v], [min_v, max_v], 'r--', alpha=0.8)

                    rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
                    mae = mean_absolute_error(true_vals, pred_vals)
                    medae = median_absolute_error(true_vals, pred_vals)
                    ax.set_title(f'{name} - {p}\nRMSE={rmse:.2f}, MAE={mae:.2f}, MedAE={medae:.2f}')

                    # Set Log-Log scale for Luminosity and Mass
                    if p in ['Lstar', 'Mstar']:
                        ax.set_xscale('log')
                        ax.set_yscale('log')
                        ax.grid(True, which="both", alpha=0.3)
                    else:
                        ax.grid(True, alpha=0.3)

                else:
                    ax.set_title(f'{name} - {p} (No Data)')

                if i == n_exps - 1:
                    ax.set_xlabel(f'True {p}')
                if j == 0:
                    ax.set_ylabel(f'Predicted {p}')

        plt.tight_layout()
        save_path = out_dir / f'parameter_scatter_{filename_suffix}.png'
        plt.savefig(save_path, dpi=300)
        print(f"Saved {filename_suffix} scatter plot to {save_path}")

    # Split parameters
    # Split parameters
    basic_params = ['Teff', 'logg', 'FeH']
    phys_params = ['Lstar', 'Mstar', 'Age']
    
    plot_scatter_grid(basic_params, 'basic', 'Basic Parameters')
    plot_scatter_grid(phys_params, 'physical', 'Physical Parameters')

if __name__ == "__main__":
    main()
