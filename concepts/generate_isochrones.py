
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from isochrones.mist import MIST_Isochrone

def generate_and_plot_cmd():
    output_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing MIST Isochrone...")
    mist = MIST_Isochrone()
    
    # Ages to plot in Gyr
    ages_gyr = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0]
    feh = 0.0
    
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    
    # Color mapping
    norm = plt.Normalize(min(ages_gyr), max(ages_gyr))
    cmap = plt.cm.viridis
    
    print("Generating isochrones...")
    
    for age_gyr in ages_gyr:
        # Convert Gyr to log10(years)
        # 1 Gyr = 1e9 years
        log_age = np.log10(age_gyr * 1e9)
        
        try:
            # Generate isochrone
            iso = mist.isochrone(log_age, feh)
            
            # Extract Photometry
            # Gaia bands: 'G_mag', 'BP_mag', 'RP_mag'
            # Check availability (based on user output they should be there)
            if 'BP_mag' in iso.columns and 'RP_mag' in iso.columns and 'G_mag' in iso.columns:
                bp = iso['BP_mag']
                rp = iso['RP_mag']
                g = iso['G_mag']
                
                color_bp_rp = bp - rp
                mag_g = g
                
                # Plot
                c = cmap(norm(age_gyr))
                plt.plot(color_bp_rp, mag_g, label=f'{age_gyr} Gyr', color=c, linewidth=2)
            else:
                print(f"Warning: Gaia columns not found for age {age_gyr}")
                
        except Exception as e:
            print(f"Error generating isochrone for age {age_gyr}: {e}")

    plt.gca().invert_yaxis() # Magnitudes decrease upwards
    plt.xlabel('$G_{BP} - G_{RP}$')
    plt.ylabel('$M_G$')
    plt.title('MIST Isochrones (Gaia CMD)')
    plt.legend(title="Age")
    
    # Set reasonable limits if needed, but auto-scaling usually works
    # Common CMD limits: Color -0.5 to 3.5, Mag 20 to -5
    
    save_path = os.path.join(output_dir, 'isochrones_gaia_cmd.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved CMD plot to {save_path}")

if __name__ == "__main__":
    generate_and_plot_cmd()
