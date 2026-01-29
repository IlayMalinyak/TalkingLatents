import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# Copied from analyze_concepts.py
CONCEPT_TITLES = {
    'evolution_giant_minus_dwarf': 'Dwarf $\\to$ Giant',
    'age_old_minus_young': 'Young $\\to$ Old',
    'feh_rich_minus_poor': 'Metal Poor $\\to$ Rich',
    'mass_high_minus_low': 'Low Mass $\\to$ High Mass',
    'teff_hot_minus_cool': 'Cool $\\to$ Hot',
    'logg_high_minus_low': 'Low logg $\\to$ High logg',
}

def load_metrics(path, experiment_name):
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None
    df = pd.read_csv(path)
    df['Experiment'] = experiment_name
    return df

def main():
    base_dir = '/home/ilay.kamai/work/TalkingLatents'
    path_v1 = os.path.join(base_dir, 'concepts/plots/concept_metrics.csv')
    path_v2 = os.path.join(base_dir, 'concepts_v2/plots/concept_metrics.csv')
    output_dir = os.path.join(base_dir, 'concepts/plots') 

    df_v1 = load_metrics(path_v1, 'Sparse Features')
    df_v2 = load_metrics(path_v2, 'Dense Features')

    if df_v1 is None and df_v2 is None:
        print("No data found.")
        return

    dfs = [d for d in [df_v1, df_v2] if d is not None]
    if not dfs:
        return
        
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Map to pretty names
    combined_df['pretty_concept'] = combined_df['concept'].map(lambda x: CONCEPT_TITLES.get(x, x))

    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['font.size'] = 16

    # Plot Difference of Averages
    plt.figure(figsize=(12, 8))
    # Using hue for color distinction and style for fill/edge if possible, 
    # but seaborn barplot doesn't support 'fill=False' natively with hue easily in one go.
    # We can use edgecolor and facecolor=(0,0,0,0) with matplotlib, but seaborn is easier for grouping.
    # To achieve "maybe without filling (only edge color)", we can iterate or set params.
    # Let's try standard barplot first with specific palette, and then modify patches if needed, 
    # OR just use alpha.
    
    ax = sns.barplot(
        data=combined_df, 
        x='pretty_concept', 
        y='diff_averages', 
        hue='Experiment',
        palette={'Sparse Features': 'blue', 'Dense Features': 'red'},
        alpha=0.6 # Make them slightly transparent to see overlap? Or just side-by-side (default)
    )
    
    # User asked for "without filling (only edge color)". 
    # Let's modify the patches to be empty with colored edges.
    colors = {'Sparse Features': 'blue', 'Dense Features': 'red'}
    for i, bar in enumerate(ax.patches):
        # Calculate which experiment this bar belongs to
        # Seaborn plots all bars for first hue, then all for second hue.
        # But safely, we can just use the facecolor to reset edge.
        pass
        
    # A better approach for "only edge color" in seaborn is manual adjustment:
    # However, standard filled bars with different colors is often clearer than edge-only if side-by-side.
    # If the user meant "overlayed" (on top of each other), barplot does side-by-side by default.
    # If they want true overlay, we shouldn't use hue in one call, but two barplot calls?
    # But side-by-side (dodged) is standard for comparison. 
    # "overlayed" usually implies occupying same x-space.
    # "make them in different colors ... and maybe without filling" suggesting they might overlap.
    # Let's assume standard side-by-side comparison (dodge=True) but with the style requested.
    
    # Let's just do standard side-by-side bars for clarity first, but apply the style request:
    # "maybe without filling (only edge color)"
    
    # Clear current axis to restart with custom style
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # We use FactorPlot or similar logic? No, let's Stick to barplot but modify afterwards.
    sns.barplot(
        data=combined_df, 
        x='pretty_concept', 
        y='diff_averages', 
        hue='Experiment',
        palette={'Sparse Features': 'none', 'Dense Features': 'none'}, # Transparent face
        edgecolor='gray', # Placeholder, will be overwritten
        ax=ax
    )
    
    # Now color the edges based on experiment manually or just use proper palette with facecolor='none'
    # Actually, seaborn doesn't make facecolor='none' easy with hue.
    
    # SIMPLER APPROACH: Standard side-by-side bars with low alpha or hatching?
    # User said "maybe without filling".
    # Let's try to set facecolor to none and edgecolor to the specific color.
    
    plt.clf()
    plt.figure(figsize=(12, 8))
    
    # Custom loop to control everything
    concepts = combined_df['pretty_concept'].unique()
    x = np.arange(len(concepts))
    width = 0.35
    
    # Filter data
    df1 = combined_df[combined_df['Experiment'] == 'Sparse Features']
    df2 = combined_df[combined_df['Experiment'] == 'Dense Features']
    
    # Reindex to ensure order matches 'concepts'
    # Create mapping
    val1 = []
    val2 = []
    
    for c in concepts:
        v1 = df1[df1['pretty_concept'] == c]['diff_averages'].values
        v2 = df2[df2['pretty_concept'] == c]['diff_averages'].values
        val1.append(v1[0] if len(v1) > 0 else 0)
        val2.append(v2[0] if len(v2) > 0 else 0)
        
    plt.bar(x - width/2, val1, width, label='Sparse Features', color='none', edgecolor='blue', linewidth=2)
    plt.bar(x + width/2, val2, width, label='Dense Features', color='none', edgecolor='red', linewidth=2)
    
    plt.ylabel('Difference of Averages')
    plt.title('Difference of Averages (Pos - Neg)')
    plt.xticks(x, concepts, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_diff_avg.png'))
    print(f"Saved comparison_diff_avg.png to {output_dir}")
    plt.close()

    # KS P-Values
    plt.figure(figsize=(12, 8))
    
    val1_ks = []
    val2_ks = []
    
    for c in concepts:
        v1 = df1[df1['pretty_concept'] == c]['ks_pvalue'].values
        v2 = df2[df2['pretty_concept'] == c]['ks_pvalue'].values
        val1_ks.append(v1[0] if len(v1) > 0 else 0)
        val2_ks.append(v2[0] if len(v2) > 0 else 0)

    plt.bar(x - width/2, val1_ks, width, label='Sparse Features', color='salmon', edgecolor='black', linewidth=2)
    plt.bar(x + width/2, val2_ks, width, label='Dense Features', color='darkgreen', edgecolor='black', linewidth=2)

    plt.ylabel('KS-test log(P-value)')
    plt.title('KS Test log(P-value)')
    plt.xticks(x, concepts, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_ks_pvalue.png'))
    print(f"Saved comparison_ks_pvalue.png to {output_dir}")
    plt.close()

if __name__ == "__main__":
    main()
