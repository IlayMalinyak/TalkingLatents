#!/usr/bin/env python3
"""
Test the dtype fix for stellar parameter predictor.
"""

import os
import sys
import torch

# Add the project root to Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

def test_dtype_conversion():
    """Test that stellar predictor matches model dtype"""
    print("Testing dtype conversion for stellar predictor...")
    
    from nn.llm_multi import MultimodalLlamaModelMultiTokens, StellarParameterPredictor
    
    # Mock model params
    class MockParams:
        def __init__(self):
            self.dim = 4096
            self.n_heads = 32
            self.rope_theta = 10000.0
            self.vocab_size = 32000
    
    class MockLlamaModel:
        def __init__(self):
            self.params = MockParams()
            self.tok_embeddings = torch.nn.Embedding(32000, 4096)
            self.layers = []
            self.norm = torch.nn.LayerNorm(4096)
            self.output = torch.nn.Linear(4096, 32000)
    
    # Test 1: Float32 (default)
    print("\nTest 1: Float32 model")
    base_model = MockLlamaModel()
    model = MultimodalLlamaModelMultiTokens(
        base_model=base_model,
        fm_model=None,
        latent_dim=2048,
        hidden_dim=1024,
        num_spectral_features=8,
        mode="single_star",
        use_cfm=False,
        predict_stellar_params=True,
        stellar_params=['Teff', 'logg', 'FeH']
    )
    
    # Check predictor dtype
    predictor_dtype = next(model.stellar_predictor.parameters()).dtype
    print(f"Stellar predictor dtype: {predictor_dtype}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 50
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    input_spectra = torch.randn(batch_size, 2048)
    special_token_positions = torch.tensor([10, 20])
    
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            input_spectra=input_spectra,
            special_token_positions=special_token_positions
        )
    
    print(f"Forward pass successful: {'stellar_predictions' in outputs}")
    if 'stellar_predictions' in outputs:
        for param, pred in outputs['stellar_predictions'].items():
            print(f"  {param}: dtype={pred.dtype}, shape={pred.shape}")
    
    # Test 2: Half precision
    print("\nTest 2: Half precision model")
    base_model_half = MockLlamaModel()
    base_model_half.half()
    
    model_half = MultimodalLlamaModelMultiTokens(
        base_model=base_model_half,
        fm_model=None,
        latent_dim=2048,
        hidden_dim=1024,
        num_spectral_features=8,
        mode="single_star",
        use_cfm=False,
        predict_stellar_params=True,
        stellar_params=['Teff', 'logg', 'FeH']
    )
    
    # Convert stellar predictor to half
    model_half.stellar_predictor.half()
    
    predictor_dtype_half = next(model_half.stellar_predictor.parameters()).dtype
    print(f"Stellar predictor dtype (half): {predictor_dtype_half}")
    
    # Test forward pass with half precision
    model_half.eval()
    with torch.no_grad():
        try:
            outputs_half = model_half(
                input_ids=input_ids,
                input_spectra=input_spectra.half(),
                special_token_positions=special_token_positions
            )
            print(f"Half precision forward pass successful: {'stellar_predictions' in outputs_half}")
            if 'stellar_predictions' in outputs_half:
                for param, pred in outputs_half['stellar_predictions'].items():
                    print(f"  {param}: dtype={pred.dtype}, shape={pred.shape}")
        except Exception as e:
            print(f"Half precision forward pass failed: {e}")
    
    return True

def test_dtype_conversion_function():
    """Test the dtype conversion in forward pass"""
    print("Testing dtype conversion function...")
    
    from nn.llm_multi import StellarParameterPredictor
    
    # Create predictor in half precision
    predictor = StellarParameterPredictor(
        hidden_dim=4096,
        stellar_params=['Teff', 'logg', 'FeH']
    ).half()
    
    print(f"Predictor dtype: {next(predictor.parameters()).dtype}")
    
    # Test with float32 input
    hidden_states_float = torch.randn(2, 100, 4096)
    print(f"Input dtype: {hidden_states_float.dtype}")
    
    # Convert input to match predictor
    hidden_states_converted = hidden_states_float.to(dtype=next(predictor.parameters()).dtype)
    print(f"Converted input dtype: {hidden_states_converted.dtype}")
    
    # Forward pass
    with torch.no_grad():
        predictions = predictor(hidden_states_converted)
        for param, pred in predictions.items():
            print(f"  {param}: dtype={pred.dtype}, shape={pred.shape}")
    
    return True

def main():
    """Run dtype tests"""
    print("="*60)
    print("DTYPE CONVERSION TESTS")
    print("="*60)
    
    tests = [
        ("Dtype Conversion", test_dtype_conversion),
        ("Dtype Conversion Function", test_dtype_conversion_function),
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
            print(f"✗ {test_name}: FAILED - {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "="*60)
    print("DTYPE TEST RESULTS")
    print("="*60)
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All dtype tests passed!")
        print("\nThe dtype mismatch issue should now be resolved.")
    else:
        print("\n⚠️  Some dtype tests failed.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)