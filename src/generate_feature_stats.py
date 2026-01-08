import argparse
import numpy as np
import os
import sys
import torch

# Add root dir to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Install dependencies if needed (though environment should be set)
# os.system('pip install tiktoken fairscale fire blobfile')

from data.dataset_interpert import StellarQuestionsDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Generate feature statistics from features file')
    parser.add_argument('--features_file', type=str, required=True, help='Path to spectral features numpy file')
    parser.add_argument('--json_file', type=str, 
                        default='/home/ilay.kamai/work/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json',
                        help='Path to stellar descriptions JSON file')
    parser.add_argument('--output_file', type=str, default='feature_stats.npz', help='Output path for stats')
    
    # Split arguments matching main script defaults
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--random_seed', type=int, default=42)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading features from {args.features_file}...")
    if not os.path.exists(args.features_file):
        print(f"Error: Usage file {args.features_file} not found.")
        sys.exit(1)
        
    try:
        features = np.load(args.features_file)
        print(f"Features shape: {features.shape}")
    except Exception as e:
        print(f"Error loading features file: {e}")
        sys.exit(1)
    
    print(f"Initializing dataset to calculate splits (seed={args.random_seed})...")
    
    # Instantiate dataset with 'train' split to get training indices
    # We pass tokenizer=None to avoid loading LLM tokenizer
    dataset = StellarQuestionsDataset(
        json_file=args.json_file,
        features_array=features,
        split='train',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed,
        tokenizer=None, 
        tokenizer_path=None
    )
    
    train_indices = dataset.split_indices
    print(f"Training set size: {len(train_indices)}")
    
    print("Collecting feature indices for training set...")
    # This maps sample indices to feature indices
    feature_indices = dataset._collect_feature_indices()
    
    print(f"Found {len(feature_indices)} valid feature vectors for training set.")
    
    if len(feature_indices) == 0:
        print("Error: No features found for training set. Check alignment between JSON and features file.")
        sys.exit(1)

    # Extract features
    train_features = features[feature_indices]
    
    # Compute stats
    print("Computing mean and std...")
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)
    
    print(f"Mean shape: {mean.shape}")
    print(f"Std shape: {std.shape}")
    print(f"Mean example: {mean[:5]}")
    print(f"Std example: {std[:5]}")
    
    print(f"Saving to {args.output_file}...")
    np.savez(args.output_file, mean=mean, std=std)
    print("Done.")

if __name__ == "__main__":
    main()
