
import os
import sys
import kiauhoku as kh

# Add root directory to path to allow imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

def check_mist_columns():
    print("Loading MIST grid to check columns...")
    try:
        # Assuming 'mist' is the name used in stellar_evolution.py
        # You might need to verify the exact string if it's different in kh.load_grid
        # logic in stellar_evolution.py says: df = kh.load_grid(grid_name)
        df = kh.load_grid('mist')
        print("\nColumns in MIST grid:")
        for col in df.columns:
            print(f"  - {col}")
            
    except Exception as e:
        print(f"Error loading grid: {e}")

if __name__ == "__main__":
    check_mist_columns()
