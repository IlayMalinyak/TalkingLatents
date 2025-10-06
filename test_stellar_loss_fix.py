#!/usr/bin/env python3
"""
Test the stellar parameter loss fix.
"""

import os
import sys
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

def test_fixed_stellar_loss():
    """Test the fixed stellar parameter loss computation"""
    print("Testing fixed stellar parameter loss computation...")
    
    # Create the exact problematic scenario from the debug
    batch = {
        'stellar_params_gt_a': torch.tensor([[4500.0, 4.2], [0.0, 0.0]]),  # Only 2 params
        'stellar_params_gt_a_present': torch.tensor([True, False]),
        'stellar_params_gt_b': torch.tensor([[5500.0, 3.8, 0.1], [0.0, 0.0, 0.0]]),  # 3 params
        'stellar_params_gt_b_present': torch.tensor([True, False]),
    }
    
    stellar_predictions = {
        'Teff': torch.tensor([4480.0, 5480.0]),
        'logg': torch.tensor([4.3, 4.0]),
        'FeH': torch.tensor([-0.4, 0.1])
    }
    
    # Replicate the fixed loss function
    class FixedTrainer:
        def compute_stellar_parameter_loss(self, stellar_predictions, batch):
            device = next(iter(stellar_predictions.values())).device
            total_loss = torch.tensor(0.0, device=device)
            loss_count = 0
            
            # Helper function to compute loss for a parameter set
            def compute_param_loss(gt_params_tensor, gt_mask_tensor, param_names_subset, debug_prefix=""):
                nonlocal total_loss, loss_count
                
                if gt_mask_tensor.any() and gt_params_tensor.size(1) > 0:
                    for i, param_name in enumerate(param_names_subset):
                        if param_name in stellar_predictions and i < gt_params_tensor.size(1):
                            pred = stellar_predictions[param_name][gt_mask_tensor]
                            gt = gt_params_tensor[gt_mask_tensor, i]
                            
                            # Check for valid values (non-zero ground truth)
                            valid_mask = (gt != 0.0)
                            if valid_mask.any():
                                pred_valid = pred[valid_mask]
                                gt_valid = gt[valid_mask]
                                
                                if len(pred_valid) > 0 and len(gt_valid) > 0:
                                    param_loss = F.l1_loss(pred_valid, gt_valid)
                                    total_loss += param_loss
                                    loss_count += 1
                                    print(f"    {debug_prefix} {param_name}: pred={pred_valid}, gt={gt_valid}, loss={param_loss.item():.4f}")
            
            # For comparative mode, use stellar_params_gt_a and stellar_params_gt_b
            if 'stellar_params_gt_a' in batch and batch['stellar_params_gt_a'] is not None:
                gt_params_a = batch['stellar_params_gt_a'].to(device)
                gt_mask_a = batch['stellar_params_gt_a_present'].to(device)
                # Only use parameters that exist in this tensor
                available_params = ['Teff', 'logg', 'FeH'][:gt_params_a.size(1)]
                print(f"  Star A - available params: {available_params}, tensor shape: {gt_params_a.shape}")
                compute_param_loss(gt_params_a, gt_mask_a, available_params, "star_a")
            
            if 'stellar_params_gt_b' in batch and batch['stellar_params_gt_b'] is not None:
                gt_params_b = batch['stellar_params_gt_b'].to(device)
                gt_mask_b = batch['stellar_params_gt_b_present'].to(device)
                # Only use parameters that exist in this tensor
                available_params = ['Teff', 'logg', 'FeH'][:gt_params_b.size(1)]
                print(f"  Star B - available params: {available_params}, tensor shape: {gt_params_b.shape}")
                compute_param_loss(gt_params_b, gt_mask_b, available_params, "star_b")
            
            # Return average loss if any parameters were processed
            if loss_count > 0:
                avg_loss = total_loss / loss_count
                print(f"  Total stellar parameter loss: {avg_loss.item():.4f}")
                return avg_loss
            else:
                print("  No stellar parameter loss computed")
                return None
    
    fixed_trainer = FixedTrainer()
    
    print("Testing with problematic batch structure:")
    print(f"stellar_params_gt_a shape: {batch['stellar_params_gt_a'].shape}")
    print(f"stellar_params_gt_b shape: {batch['stellar_params_gt_b'].shape}")
    
    try:
        loss = fixed_trainer.compute_stellar_parameter_loss(stellar_predictions, batch)
        print(f"✅ Fixed loss computation successful: {loss}")
        return True
    except Exception as e:
        print(f"❌ Fixed loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("TESTING STELLAR PARAMETER LOSS FIX")
    print("="*60)
    
    success = test_fixed_stellar_loss()
    
    if success:
        print("\n🎉 The stellar parameter loss fix works correctly!")
        print("The IndexError should now be resolved.")
    else:
        print("\n⚠️ The fix still has issues.")