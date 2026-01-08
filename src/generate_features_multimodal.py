import yaml
import pandas as pd
import datetime
import argparse
import torch
import os
import glob
from collections import OrderedDict
import numpy as np


DESA_FEATURES_PATH = '/home/ilay.kamai/work/TalkingLatents/logs/desa_features'
LAMOST_KEPLER_PATH = '/home/ilay.kamai/work/lamost/lamost_dr8_gaia_dr3_kepler_ids.csv'
TARGET_CSV = '/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/info_full.csv'
TARGET_DIR = '/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29'
NSS_PATH = '/home/ilay.kamai/work/lightPred/tables/nss_dataset.csv'
AGES_PATH = '/home/ilay.kamai/work/lightPred/tables/ages_dataset.csv'


def load_split_features(dir_path):
    # Define the splits we look for
    splits = ['train', 'val', 'test']
    
    all_features = []
    all_dfs = []
    
    for split in splits:
        # Find all npy files for this split
        npy_pattern = os.path.join(dir_path, f"*{split}.npy")
        npy_files = glob.glob(npy_pattern)
        
        # Sort to ensure consistent order if multiple matches (though typically one per split expected)
        npy_files.sort()
        print(f"Found {len(npy_files)} files for split {split}")
        
        for npy_file in npy_files:

            # Look for corresponding csv
            base_name = os.path.splitext(npy_file)[0].replace('embedding_projections', 'preds')
            csv_file = base_name + ".csv"
            
            if not os.path.exists(csv_file):
                print(f"Warning: Corresponding CSV not found for {npy_file}, skipping.")
                continue
                
            print(f"Loading {npy_file} and {csv_file}")
            features = np.load(npy_file)
            df = pd.read_csv(csv_file)
            
            if len(features) != len(df):
                raise ValueError(f"Length mismatch between {npy_file} ({len(features)}) and {csv_file} ({len(df)})")
            
            all_features.append(features)
            all_dfs.append(df)
            
    if not all_features:
        raise ValueError("No matching feature files found.")

    full_features = np.concatenate(all_features, axis=0)
    full_csv = pd.concat(all_dfs, axis=0, ignore_index=True)
    
    return full_features, full_csv

def sort_features(raw_features, raw_csv, target_csv):
    # Ensure raw_csv has an index column to track original positions
    raw_csv = raw_csv.copy()
    raw_csv['_original_index'] = np.arange(len(raw_csv))
    
    print(f"[sort_features] Raw CSV 'kid' dtype: {raw_csv['kid'].dtype}")
    print(f"[sort_features] Target CSV 'KID' dtype: {target_csv['KID'].dtype}")
    
    # Ensure raw_csv 'kid' is int
    if not pd.api.types.is_integer_dtype(raw_csv['kid']):
        print("Coercing raw_csv 'kid' to int...")
        raw_csv['kid'] = raw_csv['kid'].fillna(-1).astype(int)

    # Ensure target_csv 'KID' is int (handling floats/NaNs)
    target_csv = target_csv.copy()
    if not pd.api.types.is_integer_dtype(target_csv['KID']):
        print("Coercing target_csv 'KID' to int...")
        target_csv['KID'] = target_csv['KID'].fillna(-1).astype(int)

    # Handle duplicates by creating a rank for each occurrence of a kid
    # This allows 1-to-1 matching of the n-th occurrence of a KID in target 
    # to the n-th occurrence of separate kid in raw features.
    raw_csv['kid_rank'] = raw_csv.groupby('kid').cumcount()
    target_csv['KID_rank'] = target_csv.groupby('KID').cumcount()
    
    # We want the output to match target_csv exactly.
    # Merge on rank + ID
    print("[sort_features] Merging target and raw CSVs...")
    merged = target_csv.merge(
        raw_csv[['kid', 'kid_rank', '_original_index']], 
        left_on=['KID', 'KID_rank'], 
        right_on=['kid', 'kid_rank'], 
        how='left',
        suffixes=('', '_to_drop')
    )
    
    # Initialize full features array with zeros (dummy features)
    # Shape: (len(target_csv), feature_dim)
    # Get feature dimension from raw_features
    feature_shape = raw_features.shape[1:]
    sorted_features = np.zeros((len(target_csv), *feature_shape), dtype=raw_features.dtype)
    
    # Identify valid matches
    valid_mask = ~merged['_original_index'].isna()
    valid_count = valid_mask.sum()
    print(f"[sort_features] Found {valid_count} valid matches out of {len(target_csv)} target rows.")
    
    if not valid_mask.all():
        missing_count = (~valid_mask).sum()
        print(f"Warning: {missing_count} rows in target CSV did not match any features. Filled with zeros.")
    
    if valid_count > 0:
        # Get the indices in raw_features for the valid matches
        valid_indices = merged.loc[valid_mask, '_original_index'].astype(int).values
        
        # Get the indices in the target (sorted) array where we should place them
        # sorted_features is aligned with target_csv, so we use valid_mask directly
        sorted_features[valid_mask] = raw_features[valid_indices]
    else:
        print("CRITICAL WARNING: No matches found during merge! Feature array is all zeros.")
    
    # The requirement is "final csv should be the target csv"
    # We strip the temporary rank column from the copy we made
    if 'KID_rank' in target_csv.columns:
        target_csv = target_csv.drop(columns=['KID_rank'])
        
    aligned_target_csv = target_csv.copy()
    sorted_csv = target_csv.copy()
    
    return sorted_features, sorted_csv, aligned_target_csv

if __name__ == '__main__':
    full_features, full_csv = load_split_features(DESA_FEATURES_PATH)
    target_csv = pd.read_csv(TARGET_CSV)
    
    print(f"Loaded features shape: {full_features.shape}")
    print(f"Loaded CSV shape: {full_csv.shape}")
    print(f"Target CSV shape: {target_csv.shape}")

    sorted_features, sorted_csv, aligned_target_csv = sort_features(full_features, full_csv, target_csv)
    
    nss_csv = pd.read_csv(NSS_PATH)

    astero_age_df = pd.read_csv(AGES_PATH)
   
    aligned_target_csv = aligned_target_csv.merge(astero_age_df[['final_age', 'KID', 'age_ref']], on='KID', how='left')
    
    aligned_target_csv = aligned_target_csv.merge(nss_csv[['KID', 'binarity_class']], on='KID', how='left')
    aligned_target_csv['binarity_class_hard'] = aligned_target_csv['binarity_class'].apply(lambda x: 1 if x > 0 else 0)
    aligned_target_csv.loc[aligned_target_csv['binarity_class'].isna(), 'binarity_class_hard'] = np.nan
     
    print(f"Sorted features shape: {sorted_features.shape}")
    print(f"Sorted CSV shape: {sorted_csv.shape}")
    print(f"Aligned target CSV shape: {aligned_target_csv.shape}")
    print(aligned_target_csv.head())
    
    # --- NEW: Align with JSON ---
    import json
    
    JSON_PATH = '/home/ilay.kamai/work/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json'
    print(f"Loading JSON from {JSON_PATH} for final alignment...")
    
    with open(JSON_PATH, 'r') as f:
        json_data = json.load(f)
        
    json_obsids = [item['obsid'] for item in json_data]
    print(f"JSON contains {len(json_obsids)} entries.")
    
    # Ensure obsid in CSV is correct type (int)
    if 'obsid' not in aligned_target_csv.columns:
        raise ValueError("merged CSV does not have 'obsid' column, cannot align to JSON.")
        
    aligned_target_csv['obsid'] = aligned_target_csv['obsid'].astype(int)
    
    # Create a mapping from obsid to current row index in aligned_target_csv
    # We use a dictionary for O(1) lookup
    # Note: aligned_target_csv comes from info_full.csv which should have unique obsids per row ??
    # If info_full has duplicates, this might ambiguity. 
    # Usually info_full describes observations.
    
    # Check if all JSON obsids are in the CSV
    csv_obsids = set(aligned_target_csv['obsid'])
    missing_obsids = [oid for oid in json_obsids if oid not in csv_obsids]
    if missing_obsids:
        raise ValueError(f"Found {len(missing_obsids)} obsids in JSON that are missing from the features CSV. First few: {missing_obsids[:5]}")
        
    # Build list of indices to reorder features/CSV
    # We want: for each obsid in JSON, find the index in aligned_target_csv
    obsid_to_idx = {obsid: idx for idx, obsid in enumerate(aligned_target_csv['obsid'])}
    
    reorder_indices = [obsid_to_idx[oid] for oid in json_obsids]
    reorder_indices = np.array(reorder_indices)
    
    # Reorder Features
    final_features = sorted_features[reorder_indices]
    
    # Reorder CSV
    final_csv = aligned_target_csv.iloc[reorder_indices].reset_index(drop=True)
    
    print(f"Final aligned features shape: {final_features.shape}")
    print(f"Final aligned CSV shape: {final_csv.shape}")
    
    # Verify alignment
    for i in range(min(50, len(final_csv))):
        json_oid = json_obsids[i]
        csv_oid = final_csv.iloc[i]['obsid']
        assert json_oid == csv_oid, f"Mismatch at index {i}: JSON {json_oid} != CSV {csv_oid}"
    print("Alignment verification passed (first 50 samples).")

    np.save(f'{TARGET_DIR}/multimodal_features.npy', final_features)
    final_csv.to_csv(f'{TARGET_DIR}/info_full_multimodal.csv', index=False)



    
    