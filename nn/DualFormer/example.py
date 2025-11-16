"""
Example usage of the DualFormer model with light curves and spectra.
"""

import torch
from dual_attention import DualFormer, DualFormerForJointEmbedding, DualFormerForRegression

def example_dualformer_usage():
    # Example parameters
    batch_size = 8
    seq_len_lc = 128  # Length of light curve sequence
    seq_len_spec = 64  # Length of spectra sequence
    embed_dim = 256   # Embedding dimension
    
    # Create random embeddings for light curves and spectra
    light_curve_embeddings = torch.randn(batch_size, seq_len_lc, embed_dim)
    spectra_embeddings = torch.randn(batch_size, seq_len_spec, embed_dim)
    
    # Create random padding masks
    # 1 = valid token, 0 = padding token
    lc_padding_mask = torch.ones(batch_size, seq_len_lc)
    spec_padding_mask = torch.ones(batch_size, seq_len_spec)
    
    # Add some random padding
    for i in range(batch_size):
        pad_len_lc = torch.randint(0, 20, (1,)).item()
        pad_len_spec = torch.randint(0, 10, (1,)).item()
        if pad_len_lc > 0:
            lc_padding_mask[i, -pad_len_lc:] = 0
        if pad_len_spec > 0:
            spec_padding_mask[i, -pad_len_spec:] = 0
    
    print("===== Basic DualFormer Example =====")
    # Create a DualFormer model
    dual_former = DualFormer(
        embed_dim=embed_dim,
        num_layers=4,
        num_heads=8,
        ffn_dim=1024,
        dropout=0.1,
        pooling="mean"  # Get sequence-level representations
    )
    
    # Process through DualFormer
    lc_output, spec_output = dual_former(
        light_curve_embeddings, 
        spectra_embeddings,
        lc_padding_mask,
        spec_padding_mask
    )
    
    print(f"Light curve output shape: {lc_output.shape}")
    print(f"Spectra output shape: {spec_output.shape}")
    
    print("\n===== Joint Embedding Example =====")
    # Create a DualFormer for joint embedding
    joint_model = DualFormerForJointEmbedding(
        embed_dim=embed_dim,
        projection_dim=128,
        num_layers=4,
        num_heads=8,
        pooling="mean"
    )
    
    # Get joint embeddings
    lc_embedding, spec_embedding = joint_model(
        light_curve_embeddings, 
        spectra_embeddings,
        lc_padding_mask,
        spec_padding_mask
    )
    
    print(f"Light curve embedding shape: {lc_embedding.shape}")
    print(f"Spectra embedding shape: {spec_embedding.shape}")
    
    # Calculate cosine similarity
    similarity = torch.cosine_similarity(lc_embedding, spec_embedding, dim=1)
    print(f"Cosine similarity: {similarity}")
    
    print("\n===== Regression Example =====")
    # Create a DualFormer for regression
    regression_model = DualFormerForRegression(
        embed_dim=embed_dim,
        num_outputs=3,  # Predict 3 values (e.g., Teff, logg, FeH)
        num_layers=4,
        num_heads=8,
        pooling="mean",
        fusion="concat"  # Concatenate embeddings from both modalities
    )
    
    # Get regression predictions
    predictions = regression_model(
        light_curve_embeddings, 
        spectra_embeddings,
        lc_padding_mask,
        spec_padding_mask
    )
    
    print(f"Regression predictions shape: {predictions.shape}")
    print(f"Example predictions:\n{predictions[:2]}")

if __name__ == "__main__":
    example_dualformer_usage()