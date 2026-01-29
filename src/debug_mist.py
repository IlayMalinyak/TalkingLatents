
import os
import sys
import numpy as np

try:
    from isochrones.mist import MIST_Isochrone
    import isochrones
    print("MIST imported successfully.")
    print(f"User Home: {os.path.expanduser('~')}")
    print(f"Isochrones Root: {os.path.join(os.path.expanduser('~'), '.isochrones')}")
    # Check if dir exists
    root = os.path.join(os.path.expanduser('~'), '.isochrones')
    if os.path.exists(root):
        print(f"Root dir exists: Yes")
        print(f"Contents: {os.listdir(root)}")
    else:
        print(f"Root dir exists: NO")

    # Check REAL user home
    real_home = "/home/ilay.kamai/.isochrones"
    print(f"Checking {real_home}...")
    if os.path.exists(real_home):
        print(f"Real Home exists: Yes")
        if os.path.exists(os.path.join(real_home, 'mist')):
            print(f"MIST dir contents: {os.listdir(os.path.join(real_home, 'mist'))[:5]}")
    else:
        print(f"Real Home exists: NO")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_mist():
    print("Initializing MIST...")
    try:
        mist = MIST_Isochrone()
        print("MIST initialized.")
    except Exception as e:
        print(f"MIST init failed: {e}")
        return

    # Test 1: Solar
    print("\n--- Test 1: Solar values ---")
    log_age = np.log10(4.6e9)
    feh = 0.0
    print(f"Querying: log_age={log_age:.2f}, feh={feh}")
    try:
        iso = mist.isochrone(log_age, feh)
        print(f"Result length: {len(iso)}")
        if len(iso) > 0:
             print(f"Stats: Teff {10**iso['logTeff'].min():.0f}-{10**iso['logTeff'].max():.0f}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Problematic Star
    print("\n--- Test 2: Problematic Star ---")
    # Age=2.92, FeH=-0.021
    log_age = np.log10(2.92e9)
    feh = -0.021
    print(f"Querying: log_age={log_age:.2f}, feh={feh}")
    try:
        iso = mist.isochrone(log_age, feh)
        print(f"Result length: {len(iso)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_mist()
