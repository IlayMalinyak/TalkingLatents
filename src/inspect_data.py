import pandas as pd
import numpy as np
import glob
import os

DESA_FEATURES_PATH = '/home/ilay.kamai/work/TalkingLatents/logs/desa_features'
LAMOST_KEPLER_PATH = '/home/ilay.kamai/work/lamost/lamost_dr8_gaia_dr3_kepler_ids.csv'

def check_duplicates():
    print("Loading target CSV...")
    target_csv = pd.read_csv(LAMOST_KEPLER_PATH)
    print(f"Target CSV shape: {target_csv.shape}")
    print(f"Target CSV columns: {target_csv.columns}")
    if 'KID' in target_csv.columns:
        dups = target_csv['KID'].duplicated().sum()
        print(f"Duplicates in Target CSV 'KID': {dups}")
        if dups == 0:
            print("Target CSV 'KID' is unique.")
        else:
            print("Target CSV 'KID' is NOT unique.")

    print("\nLoading feature CSVs...")
    all_dfs = []
    splits = ['train', 'val', 'test']
    for split in splits:
        npy_pattern = os.path.join(DESA_FEATURES_PATH, f"*{split}.npy")
        npy_files = glob.glob(npy_pattern)
        npy_files.sort()
        for npy_file in npy_files:
            base_name = os.path.splitext(npy_file)[0].replace('embedding_projections', 'preds')
            csv_file = base_name + ".csv"
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                all_dfs.append(df)
    
    if not all_dfs:
        print("No feature CSVs found.")
        return

    full_csv = pd.concat(all_dfs, axis=0, ignore_index=True)
    print(f"Full CSV shape: {full_csv.shape}")
    print(f"Full CSV columns: {full_csv.columns}")
    
    if 'kid' in full_csv.columns:
        dups = full_csv['kid'].duplicated().sum()
        print(f"Duplicates in Feature CSV 'kid': {dups}")
    else:
        print("'kid' column not found in Feature CSV.")
        if 'KID' in full_csv.columns:
             print(f"Found 'KID' instead. Duplicates: {full_csv['KID'].duplicated().sum()}")
    
    if 'ObsID' in target_csv.columns and 'ObsID' in full_csv.columns:
        # Check if we should merge on ObsID
        print("\nChecking matching logic...")
        # Check overlap
        common_kids = set(target_csv['KID']).intersection(set(full_csv['kid']))
        print(f"Common KIDs: {len(common_kids)}")

if __name__ == '__main__':
    check_duplicates()
