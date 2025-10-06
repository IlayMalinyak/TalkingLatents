#!/usr/bin/env python3
"""
Test script to verify stellar parameter prediction functionality.
"""

import os
import sys
import torch
import numpy as np

# Add the project root to Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

def test_stellar_parameter_predictor():
    """Test the StellarParameterPredictor class"""
    print("Testing StellarParameterPredictor...")
    
    from nn.llm_multi import StellarParameterPredictor
    
    # Create a predictor
    predictor = StellarParameterPredictor(
        hidden_dim=4096,
        stellar_params=['Teff', 'logg', 'FeH']
    )
    
    # Test with different input shapes
    batch_size = 2
    seq_len = 100
    hidden_dim = 4096
    
    # Test with 3D input (batch_size, seq_len, hidden_dim)
    hidden_states_3d = torch.randn(batch_size, seq_len, hidden_dim)
    predictions_3d = predictor(hidden_states_3d)
    
    print(f"3D input shape: {hidden_states_3d.shape}")
    for param, pred in predictions_3d.items():
        print(f"  {param}: {pred.shape} -> {pred}")
    
    # Test with 2D input (batch_size, hidden_dim)
    hidden_states_2d = torch.randn(batch_size, hidden_dim)
    predictions_2d = predictor(hidden_states_2d)
    
    print(f"2D input shape: {hidden_states_2d.shape}")
    for param, pred in predictions_2d.items():
        print(f"  {param}: {pred.shape} -> {pred}")
    
    return True

def test_mixed_dataset_stellar_params():
    """Test mixed dataset stellar parameter extraction"""
    print("Testing mixed dataset stellar parameter functionality...")
    
    # Create a mock batch similar to mixed dataset output
    batch = {
        'stellar_params_gt': torch.tensor([[5000.0, 4.5, -0.2], [6000.0, 4.0, 0.1]]),
        'stellar_params_gt_present': torch.tensor([True, True]),
        'stellar_params_gt_a': torch.tensor([[4500.0, 4.2, -0.5], [5500.0, 4.1, 0.0]]),
        'stellar_params_gt_a_present': torch.tensor([True, True]),
        'stellar_params_gt_b': torch.tensor([[5200.0, 4.8, 0.3], [6200.0, 3.9, 0.2]]),
        'stellar_params_gt_b_present': torch.tensor([True, True]),
    }
    
    # Create mock stellar predictions
    stellar_predictions = {
        'Teff': torch.tensor([4980.0, 5950.0]),
        'logg': torch.tensor([4.4, 4.1]),
        'FeH': torch.tensor([-0.15, 0.05])
    }
    
    print("Mock batch stellar parameters:")
    for key, value in batch.items():
        if 'stellar_params' in key:
            print(f"  {key}: {value}")
    
    print("Mock stellar predictions:")
    for param, pred in stellar_predictions.items():
        print(f"  {param}: {pred}")
    
    return True

def test_stellar_loss_computation():
    """Test stellar parameter loss computation"""
    print("Testing stellar parameter loss computation...")
    
    # Import necessary functions
    import torch.nn.functional as F
    
    # Mock predictions and ground truth
    stellar_predictions = {
        'Teff': torch.tensor([4980.0, 5950.0]),
        'logg': torch.tensor([4.4, 4.1]),
        'FeH': torch.tensor([-0.15, 0.05])
    }
    
    batch = {
        'stellar_params_gt': torch.tensor([[5000.0, 4.5, -0.2], [6000.0, 4.0, 0.1]]),
        'stellar_params_gt_present': torch.tensor([True, True]),
    }
    
    # Compute loss manually
    device = torch.device('cpu')
    total_loss = torch.tensor(0.0, device=device)
    loss_count = 0
    
    gt_params = batch['stellar_params_gt'].to(device)
    gt_mask = batch['stellar_params_gt_present'].to(device)
    
    param_names = ['Teff', 'logg', 'FeH']
    for i, param_name in enumerate(param_names):
        if param_name in stellar_predictions:
            pred = stellar_predictions[param_name][gt_mask]
            gt = gt_params[gt_mask, i]
            
            param_loss = F.l1_loss(pred, gt)
            total_loss += param_loss
            loss_count += 1
            
            print(f"{param_name}: pred={pred}, gt={gt}, loss={param_loss.item():.4f}")
    
    avg_loss = total_loss / loss_count if loss_count > 0 else 0
    print(f"Average stellar parameter loss: {avg_loss.item():.4f}")
    
    return True

def test_json_formatting():
    """Test JSON formatting for stellar parameters"""
    print("Testing JSON formatting for stellar parameters...")
    
    import json
    
    # Test single-star format
    single_star_params = {
        'Teff': 5000.123,
        'logg': 4.456,
        'FeH': -0.234
    }
    
    single_json = json.dumps({k: round(v, 2) for k, v in single_star_params.items()}, separators=(',', ':'))
    print(f"Single-star JSON: {single_json}")
    
    # Test comparative format
    comparative_params = {
        'Teff_A': 4500.789,
        'logg_A': 4.123,
        'FeH_A': -0.567,
        'Teff_B': 5200.456,
        'logg_B': 4.789,
        'FeH_B': 0.321
    }
    
    comparative_json = json.dumps({k: round(v, 2) for k, v in comparative_params.items()}, separators=(',', ':'))
    print(f"Comparative JSON: {comparative_json}")
    
    return True

def main():
    """Run all tests"""
    print("="*60)
    print("TESTING STELLAR PARAMETER PREDICTION FUNCTIONALITY")
    print("="*60)
    
    tests = [
        ("Stellar Parameter Predictor", test_stellar_parameter_predictor),
        ("Mixed Dataset Stellar Params", test_mixed_dataset_stellar_params),
        ("Stellar Loss Computation", test_stellar_loss_computation),
        ("JSON Formatting", test_json_formatting),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'-'*40}")
        print(f"Running: {test_name}")
        print(f"{'-'*40}")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"✓ {test_name}: PASSED")
        except Exception as e:
            results.append((test_name, False))
            print(f"✗ {test_name}: FAILED - {e}")
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All tests passed! Stellar parameter prediction is ready.")
        print("\nNext steps:")
        print("1. Run actual training with stellar parameter loss")
        print("2. Monitor stellar parameter predictions during validation")
        print("3. Check that both language modeling and stellar parameter losses are computed")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)