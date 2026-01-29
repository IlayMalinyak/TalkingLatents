
import os
import sys
import numpy as np
from isochrones.mist import MIST_Isochrone

def test_isochrones():
    print("Initializing MIST Isochrone...")
    mist = MIST_Isochrone()
    
    print("Generating isochrones...")
    # age in log10(years). 1 Gyr = 1e9 yr -> log10 = 9.0
    log_age = 9.0 
    feh = 0.0
    
    # isochrone method returns a dataframe
    iso = mist.isochrone(log_age, feh)
    
    print("Columns available:")
    print(iso.columns)
    
    print("\nFirst few rows:")
    print(iso.head())
    
    # Check for Gaia bands (G, BP, RP) usually denoted as G_mag, BP_mag, RP_mag or similar
    gaia_cols = [c for c in iso.columns if 'G' in c or 'BP' in c or 'RP' in c]
    print("\nPotential Gaia columns:", gaia_cols)

if __name__ == "__main__":
    test_isochrones()
