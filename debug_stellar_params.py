#!/usr/bin/env python3
"""
Debug script to understand the stellar parameter tensor structure.
"""

import os
import sys
import torch
import numpy as np

# Add the project root to Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

def debug_mixed_dataset():
    """Debug the mixed dataset to see what stellar parameters are being provided"""
    print("Debugging mixed dataset stellar parameters...")
    
    from data.dataset_mixed import MixedStellarQADataset, collate_mixed_fn
    from torch.utils.data import DataLoader
    
    # Create mock datasets
    class MockSingleDataset:
        def __init__(self):
            self.length = 10
            
        def __len__(self):
            return self.length
            
        def __getitem__(self, idx):
            # Mock single-star sample
            return {
                'input_ids': torch.randint(0, 1000, (50,)),
                'target_ids': torch.randint(0, 1000, (50,)),
                'answer_start_idx': 20,
                'target_length': 10,
                'masked_spectra': torch.randn(2048),
                'stellar_data': {
                    'Teff': 5000.0 + idx * 100,
                    'logg': 4.0 + idx * 0.1,
                    'FeH': -0.2 + idx * 0.05,
                    'extra_param': 42.0  # This might cause issues
                },
                'input_text': f'What is the temperature of star {idx}?',
                'target_text': f'The temperature is {5000 + idx * 100} K',
                'feature_start_idx': 5,
                'obsid': f'single_{idx}'
            }
    
    class MockComparativeDataset:
        def __init__(self):
            self.length = 10
            
        def __len__(self):
            return self.length
            
        def __getitem__(self, idx):
            # Mock comparative sample
            return {
                'input_ids': torch.randint(0, 1000, (50,)),
                'target_ids': torch.randint(0, 1000, (50,)),
                'answer_start_idx': 25,
                'target_length': 8,
                'masked_spectra_a': torch.randn(2048),
                'masked_spectra_b': torch.randn(2048),
                'star_a_params': {
                    'Teff': 4500.0 + idx * 50,
                    'logg': 4.2 + idx * 0.05,
                    # Missing FeH to test partial data
                },
                'star_b_params': {
                    'Teff': 5500.0 + idx * 75,
                    'logg': 3.8 + idx * 0.08,
                    'FeH': 0.1 + idx * 0.02,
                },
                'input_text': f'Which star is hotter, A or B? (sample {idx})',
                'target_text': 'Star B is hotter',
                'star_a_feature_indices': list(range(8)),
                'star_b_feature_indices': list(range(16, 24)),
                'obsid': f'comp_{idx}'
            }
    
    # Create mixed dataset
    single_dataset = MockSingleDataset()
    comparative_dataset = MockComparativeDataset()
    
    mixed_dataset = MixedStellarQADataset(
        single_dataset=single_dataset,
        comparative_dataset=comparative_dataset,
        single_sample_prob=0.5,
        seed=42,
        length_strategy="max",
        numeric_keys=('Teff', 'logg', 'FeH')
    )
    
    # Create dataloader
    dataloader = DataLoader(
        mixed_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_mixed_fn
    )
    
    # Get first batch and examine it
    batch = next(iter(dataloader))
    
    print("\nBatch structure:")
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"  {key}: {value.shape} {value.dtype}")
            if 'stellar_params' in key:
                print(f"    Values: {value}")
        elif isinstance(value, list):
            print(f"  {key}: list of length {len(value)}")
            if len(value) > 0:
                print(f"    First item: {value[0]}")
        else:
            print(f"  {key}: {type(value)}")
    
    # Focus on stellar parameter tensors
    print("\nDetailed stellar parameter analysis:")
    
    stellar_keys = [k for k in batch.keys() if 'stellar_params' in k]
    for key in stellar_keys:
        if key in batch and batch[key] is not None:
            tensor = batch[key]
            print(f"\n{key}:")
            print(f"  Shape: {tensor.shape}")
            print(f"  Dtype: {tensor.dtype}")
            print(f"  Values:\n{tensor}")
            
            # Check if this is a mask
            if 'present' in key:
                print(f"  Valid samples: {tensor.sum().item()}/{len(tensor)}")
    
    # Test the stellar parameter loss function manually
    print("\nTesting stellar parameter loss computation...")
    
    # Mock stellar predictions
    stellar_predictions = {
        'Teff': torch.tensor([4990.0, 5450.0]),
        'logg': torch.tensor([4.05, 4.15]),
        'FeH': torch.tensor([-0.18, 0.08])
    }
    
    # Import the loss function
    from nn.train import LLMTrainer
    
    # Create a mock trainer to use the loss function
    class MockTrainer:
        def compute_stellar_parameter_loss(self, stellar_predictions, batch):
            import torch.nn.functional as F
            
            device = next(iter(stellar_predictions.values())).device
            total_loss = torch.tensor(0.0, device=device)
            loss_count = 0
            
            print(f"  Computing loss with predictions: {stellar_predictions}")
            
            # For single-star mode, use stellar_params_gt
            if 'stellar_params_gt' in batch and batch['stellar_params_gt'] is not None:
                gt_params = batch['stellar_params_gt'].to(device)
                gt_mask = batch['stellar_params_gt_present'].to(device)
                
                print(f"  Single-star GT shape: {gt_params.shape}, mask: {gt_mask}")
                
                if gt_mask.any() and gt_params.size(1) > 0:
                    param_names = ['Teff', 'logg', 'FeH']
                    for i, param_name in enumerate(param_names):
                        if param_name in stellar_predictions and i < gt_params.size(1):
                            pred = stellar_predictions[param_name][gt_mask]
                            gt = gt_params[gt_mask, i]
                            
                            print(f"    {param_name}: pred={pred}, gt={gt}")
                            
                            if len(pred) > 0 and len(gt) > 0:
                                param_loss = F.l1_loss(pred, gt)
                                total_loss += param_loss
                                loss_count += 1
                                print(f"      Loss: {param_loss.item():.4f}")
            
            # Return average loss if any parameters were processed
            if loss_count > 0:
                avg_loss = total_loss / loss_count
                print(f"  Total stellar parameter loss: {avg_loss.item():.4f}")
                return avg_loss
            else:
                print("  No stellar parameter loss computed")
                return None
    
    mock_trainer = MockTrainer()
    loss = mock_trainer.compute_stellar_parameter_loss(stellar_predictions, batch)
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("DEBUGGING STELLAR PARAMETER TENSORS")
    print("="*60)
    
    try:
        debug_mixed_dataset()
        print("\n✅ Debug completed successfully")
    except Exception as e:
        print(f"\n❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()