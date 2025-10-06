#!/usr/bin/env python3
"""
Integration test for stellar parameter prediction in the full model.
"""

import os
import sys
import torch
import numpy as np

# Add the project root to Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

def test_model_integration():
    """Test stellar parameter prediction integration with the full model"""
    print("Testing stellar parameter prediction integration...")
    
    from nn.llm_multi import MultimodalLlamaModelMultiTokens, StellarParameterPredictor
    
    # Mock model params (simplified)
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
            self.layers = []  # No layers for this test
            self.norm = torch.nn.LayerNorm(4096)
            self.output = torch.nn.Linear(4096, 32000)
    
    # Create the multimodal model
    base_model = MockLlamaModel()
    fm_model = None  # No FM model for this test
    
    model = MultimodalLlamaModelMultiTokens(
        base_model=base_model,
        fm_model=fm_model,
        latent_dim=2048,
        hidden_dim=1024,
        num_spectral_features=8,
        mode="single_star",
        use_cfm=False,
        predict_stellar_params=True,
        stellar_params=['Teff', 'logg', 'FeH']
    )
    
    print(f"Model created with stellar parameter prediction: {model.predict_stellar_params}")
    print(f"Stellar predictor available: {hasattr(model, 'stellar_predictor')}")
    
    # Test with mock input
    batch_size = 2
    seq_len = 100
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    input_spectra = torch.randn(batch_size, 2048)
    special_token_positions = torch.tensor([10, 20])  # Position for spectral features
    
    # Run forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            input_spectra=input_spectra,
            special_token_positions=special_token_positions
        )
    
    print(f"Forward pass outputs: {list(outputs.keys())}")
    
    # Check if stellar predictions are present
    if 'stellar_predictions' in outputs:
        stellar_preds = outputs['stellar_predictions']
        print(f"Stellar predictions found: {list(stellar_preds.keys())}")
        
        for param, pred in stellar_preds.items():
            print(f"  {param}: shape={pred.shape}, values={pred}")
        
        # Verify shapes
        for param, pred in stellar_preds.items():
            assert pred.shape == (batch_size,), f"Expected shape ({batch_size},), got {pred.shape}"
        
        print("✓ Stellar parameter prediction integration successful")
        return True
    else:
        print("✗ No stellar predictions found in outputs")
        return False

def test_mixed_mode_integration():
    """Test stellar parameter prediction with mixed mode"""
    print("Testing stellar parameter prediction with mixed mode...")
    
    from nn.llm_multi import MultimodalLlamaModelMultiTokens
    
    # Mock model params (simplified)
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
            self.layers = []  # No layers for this test
            self.norm = torch.nn.LayerNorm(4096)
            self.output = torch.nn.Linear(4096, 32000)
    
    # Create the multimodal model in combined mode
    base_model = MockLlamaModel()
    fm_model = None
    
    model = MultimodalLlamaModelMultiTokens(
        base_model=base_model,
        fm_model=fm_model,
        latent_dim=2048,
        hidden_dim=1024,
        num_spectral_features=8,
        mode="combined",
        use_cfm=False,
        predict_stellar_params=True,
        stellar_params=['Teff', 'logg', 'FeH']
    )
    
    # Create mock mixed batch
    batch_size = 2
    seq_len = 100
    
    mixed_batch = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
        'mode_mask_single': torch.tensor([True, False]),
        'mode_mask_comparative': torch.tensor([False, True]),
        'masked_spectra': torch.randn(batch_size, 2048),
        'masked_spectra_a': torch.randn(batch_size, 2048),
        'masked_spectra_b': torch.randn(batch_size, 2048),
        'feature_start_indices': torch.tensor([10, 20]),
        'star_a_feature_indices': torch.zeros(batch_size, 8, dtype=torch.long),
        'star_b_feature_indices': torch.zeros(batch_size, 8, dtype=torch.long),
    }
    
    # Run forward pass in mixed mode
    model.eval()
    with torch.no_grad():
        outputs = model.forward_mixed(mixed_batch)
    
    print(f"Mixed mode forward pass outputs: {list(outputs.keys())}")
    
    # Check if stellar predictions are present
    if 'stellar_predictions' in outputs:
        stellar_preds = outputs['stellar_predictions']
        print(f"Mixed mode stellar predictions found: {list(stellar_preds.keys())}")
        
        for param, pred in stellar_preds.items():
            print(f"  {param}: shape={pred.shape}, values={pred}")
        
        print("✓ Mixed mode stellar parameter prediction successful")
        return True
    else:
        print("✗ No stellar predictions found in mixed mode outputs")
        return False

def main():
    """Run integration tests"""
    print("="*60)
    print("STELLAR PARAMETER PREDICTION INTEGRATION TESTS")
    print("="*60)
    
    tests = [
        ("Single Star Mode Integration", test_model_integration),
        ("Mixed Mode Integration", test_mixed_mode_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'-'*40}")
        print(f"Running: {test_name}")
        print(f"{'-'*40}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name}: FAILED - {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "="*60)
    print("INTEGRATION TEST RESULTS")
    print("="*60)
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All integration tests passed!")
        print("\nThe stellar parameter prediction functionality is fully integrated and ready for training.")
        print("\nExpected behavior during training:")
        print("1. Model will predict stellar parameters alongside text generation")
        print("2. Stellar parameter loss will be computed and added to the total loss")
        print("3. During evaluation, predicted and true stellar parameters will be displayed")
        print("4. JSON format: {\"Teff\":5000.12,\"logg\":4.46,\"FeH\":-0.23}")
    else:
        print("\n⚠️  Some integration tests failed. Please check the implementation.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)