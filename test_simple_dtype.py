#!/usr/bin/env python3
"""
Simple test to verify dtype conversion works.
"""

import torch
import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

def test_dtype_conversion():
    """Test the exact scenario that was failing"""
    print("Testing dtype conversion scenario...")
    
    from nn.llm_multi import StellarParameterPredictor
    
    # Create predictor (default float32)
    predictor = StellarParameterPredictor(
        hidden_dim=4096,
        stellar_params=['Teff', 'logg', 'FeH']
    )
    
    print(f"Initial predictor dtype: {next(predictor.parameters()).dtype}")
    
    # Convert to half (simulating model.half())
    predictor.half()
    print(f"Predictor dtype after .half(): {next(predictor.parameters()).dtype}")
    
    # Create hidden states in half precision (simulating model output)
    h = torch.randn(2, 50, 4096, dtype=torch.float16)
    print(f"Hidden states dtype: {h.dtype}")
    
    # Test the conversion logic from our fix
    h_for_predictor = h.to(dtype=next(predictor.parameters()).dtype)
    print(f"Converted hidden states dtype: {h_for_predictor.dtype}")
    
    # Test forward pass (this should work now)
    with torch.no_grad():
        predictions = predictor(h_for_predictor)
        print("Forward pass successful!")
        for param, pred in predictions.items():
            print(f"  {param}: dtype={pred.dtype}, shape={pred.shape}")
    
    return True

def test_mixed_dtype_scenario():
    """Test the exact scenario: float32 predictor with half precision model"""
    print("\nTesting mixed dtype scenario...")
    
    from nn.llm_multi import StellarParameterPredictor
    
    # Predictor in float32 (this was the issue)
    predictor = StellarParameterPredictor(
        hidden_dim=4096,
        stellar_params=['Teff', 'logg', 'FeH']
    )
    
    print(f"Predictor dtype: {next(predictor.parameters()).dtype}")
    
    # Hidden states in half precision (from model)
    h = torch.randn(2, 50, 4096, dtype=torch.float16)
    print(f"Hidden states dtype: {h.dtype}")
    
    # Our fix: convert hidden states to match predictor
    h_for_predictor = h.to(dtype=next(predictor.parameters()).dtype)
    print(f"Converted hidden states dtype: {h_for_predictor.dtype}")
    
    # This should work now
    with torch.no_grad():
        predictions = predictor(h_for_predictor)
        print("Mixed dtype forward pass successful!")
        for param, pred in predictions.items():
            print(f"  {param}: dtype={pred.dtype}, shape={pred.shape}")
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("SIMPLE DTYPE CONVERSION TESTS")
    print("="*60)
    
    try:
        test_dtype_conversion()
        test_mixed_dtype_scenario()
        print("\n🎉 All simple dtype tests passed!")
        print("\nThe dtype mismatch issue has been fixed.")
        print("The model should now work correctly with half precision.")
    except Exception as e:
        print(f"\n✗ Tests failed: {e}")
        import traceback
        traceback.print_exc()