#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np

# Add root directory to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# Import the baseline function
from src.inference import run_baseline_mlp_interpolation
from data.dataset_mixed import MixedDataset

def main():
    # Set up paths
    baseline_checkpoint = os.path.join(ROOT_DIR, 'logs', 'baseline_mlp', 'baseline_mlp.pt')
    print(f"Baseline checkpoint path: {baseline_checkpoint}")
    print(f"Exists: {os.path.isfile(baseline_checkpoint)}")
    
    # Create a minimal dataset for testing
    dataset = MixedDataset(
        stellar_questions_file='data/stellar_descriptions_questions_short.jsonl',
        comparative_file='data/comparative_dataset.jsonl',
        cache_dir='logs/debug',
        max_samples_stellar=100,
        max_samples_comparative=100,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    # Test with a few pairs
    pairs = [(0, 1), (2, 3)]
    alphas = [0.0, 0.5, 1.0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Testing baseline interpolation with {len(pairs)} pairs...")
    
    try:
        results = run_baseline_mlp_interpolation(
            checkpoint_path=baseline_checkpoint,
            dataset=dataset,
            pairs=pairs,
            alphas=alphas,
            device=device
        )
        
        print(f"Results length: {len(results)}")
        if results:
            print(f"First result keys: {list(results[0].keys())}")
            print(f"First result: {results[0]}")
        else:
            print("No results returned!")
            
    except Exception as e:
        print(f"Error running baseline interpolation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()