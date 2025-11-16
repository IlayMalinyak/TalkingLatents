#!/usr/bin/env python
"""Test script for StellarFeaturePredictionDataset"""

import sys
import os
import numpy as np

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from data.dataset_feature_pred import StellarFeaturePredictionDataset, create_feature_prediction_dataloaders


def test_dataset():
    """Test the feature prediction dataset"""

    # Paths
    json_file = "/data/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json"
    features_file = "/data/TalkingLatents/logs/2025-07-29/features.npy"
    tokenizer_path = "/data/.llama/Llama3.1-8B/tokenizer.model"

    # Check if files exist
    if not os.path.exists(json_file):
        print(f"ERROR: JSON file not found: {json_file}")
        return
    if not os.path.exists(features_file):
        print(f"ERROR: Features file not found: {features_file}")
        return

    print("="*80)
    print("Testing StellarFeaturePredictionDataset")
    print("="*80)

    # Load features
    print(f"\nLoading features from {features_file}")
    features_array = np.load(features_file)
    print(f"Features shape: {features_array.shape}")

    # Create dataset
    print(f"\nCreating dataset from {json_file}")
    dataset = StellarFeaturePredictionDataset(
        json_file=json_file,
        features_array=features_array,
        split='train',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        tokenizer_path=tokenizer_path,
        max_length=512,
        feature_precision=4,
        feature_separator=' '
    )

    print(f"\nDataset created successfully!")
    print(f"Number of samples: {len(dataset)}")

    # Get a sample
    print("\n" + "="*80)
    print("Sample 0:")
    print("="*80)
    sample = dataset[0]

    print(f"\nKeys in sample: {list(sample.keys())}")
    print(f"\nInput IDs shape: {sample['input_ids'].shape}")
    print(f"Target IDs shape: {sample['target_ids'].shape}")
    print(f"Features shape: {sample['features'].shape}")

    print(f"\nQuestion length: {sample['input_length']}")
    print(f"Answer length: {sample['target_length']}")

    print(f"\nStellar parameters:")
    for param, value in sample['stellar_params'].items():
        print(f"  {param}: {value:.4f}")

    print(f"\nInput text (question):")
    print(f"  {sample['input_text']}")

    print(f"\nTarget text (answer, first 200 chars):")
    answer_preview = sample['target_text'][:200] + "..." if len(sample['target_text']) > 200 else sample['target_text']
    print(f"  {answer_preview}")

    # Test multiple samples
    print("\n" + "="*80)
    print("Testing multiple samples:")
    print("="*80)
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Teff={sample['stellar_params']['Teff']:.1f}, "
              f"logg={sample['stellar_params']['logg']:.2f}, "
              f"FeH={sample['stellar_params']['FeH']:.2f}")
        print(f"  Features shape: {sample['features'].shape}")
        print(f"  Question length: {sample['input_length']}, Answer length: {sample['target_length']}")

    # Test dataloaders
    print("\n" + "="*80)
    print("Testing dataloaders:")
    print("="*80)

    train_loader, val_loader, test_loader = create_feature_prediction_dataloaders(
        json_file=json_file,
        features_array=features_array,
        batch_size=4,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        num_workers=0,
        cache_dir='./cache_test',
        tokenizer_path=tokenizer_path,
        max_length=512
    )

    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")

    # Get a batch
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {list(batch.keys())}")
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch target_ids shape: {batch['target_ids'].shape}")
    print(f"Batch features shape: {batch['features'].shape}")

    print("\n" + "="*80)
    print("All tests passed successfully!")
    print("="*80)


if __name__ == "__main__":
    test_dataset()
