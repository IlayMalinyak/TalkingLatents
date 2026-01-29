import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
from torch.utils.data import DataLoader

# Add root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from data.dataset_interpert import StellarQuestionsDataset, collate_fn
from data.transforms import Compose, GeneralSpectrumPreprocessor, ToTensor
from src.simple_questions_multitok import ensure_backend_config
from src.simple_questions import _load_spectra_model
from nn.llm_multi import MultimodalLlamaModelMultiTokens

class MockLLM(nn.Module):
    def __init__(self, hidden_size=4096, vocab_size=128256):
        super().__init__()
        self.config = argparse.Namespace(hidden_size=hidden_size, vocab_size=vocab_size)
        self.params = argparse.Namespace(dim=hidden_size, vocab_size=vocab_size, n_heads=32, rope_theta=500000)
        self.layers = nn.ModuleList([])
    
    def forward(self, *args, **kwargs):
        return None

    def tok_embeddings(self, x):
        return torch.randn(x.shape[0], x.shape[1], self.config.hidden_size)

def display_array_stats(arr, name):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    print(f"[{name}] Shape: {arr.shape}")
    print(f"  Min: {arr.min():.6e}")
    print(f"  Max: {arr.max():.6e}")
    print(f"  Mean: {arr.mean():.6e}")
    print(f"  Std: {arr.std():.6e}")
    if arr.size > 5:
        print(f"  First 5: {arr.flatten()[:5]}")

def parse_args():
    parser = argparse.ArgumentParser()
    # Paths matches simple_questions_multitok.py defaults
    parser.add_argument('--json_file', type=str, 
                        default='/home/ilay.kamai/work/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json')
    parser.add_argument('--features_file', type=str, 
                        default='/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/features.npy')
    
    parser.add_argument('--spectral_embedding_dim', type=int, default=2048)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_spectral_features', type=int, default=8)
    parser.add_argument('--v2', action='store_true', default=False)
    
    parser.add_argument('--max_seq_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # =========================================================================
    # 1. Pipeline 1: Pre-computed Features Dataset (Raw from file)
    # =========================================================================
    print(f"Loading features from {args.features_file}...")
    features_array = np.load(args.features_file)
    print(f"Features shape: {features_array.shape}")
    display_array_stats(features_array, "Raw Features Array")

    print("\n--- Initializing Pipeline 1 (Pre-computed) ---")
    transf = Compose([GeneralSpectrumPreprocessor(rv_norm=True), ToTensor()])
    
    # We use index_df if available to ensure alignment
    try:
        index_df = pd.read_csv('/home/ilay.kamai/work/TalkingLatents/data/dataset/index.csv')
    except:
        index_df = None

    dataset_1 = StellarQuestionsDataset(
        json_file=args.json_file,
        features_array=features_array,
        split='train',
        train_ratio=0.99, # Use most data
        val_ratio=0.005,
        test_ratio=0.005,
        random_state=42,
        spectral_transforms=transf,
        num_spectral_features=args.num_spectral_features,
        normalize_features=False, # <--- RAW FEATURES
        index_df=index_df
    )
    
    # Compute stats just for info, but don't use them for normalization
    stats = dataset_1.get_feature_normalization_stats()
    if stats:
        display_array_stats(stats['mean'], "Dataset Stats Mean")

    # =========================================================================
    # 2. Pipeline 2: On-the-fly Generation Dataset + Model (Raw Output)
    # =========================================================================
    print("\n--- Initializing Pipeline 2 (On-the-fly) ---")
    dataset_2 = StellarQuestionsDataset(
        json_file=args.json_file,
        features_array=None, # Loads raw spectra
        split='train',
        train_ratio=0.99,
        val_ratio=0.005,
        test_ratio=0.005,
        random_state=42, # Same seed
        spectral_transforms=transf,
        num_spectral_features=args.num_spectral_features,
        normalize_features=False, # <--- RAW SPECTRA Output
        index_df=index_df,
        feature_stats=None
    )

    print("Loading spectral model (fm_model)...")
    fm_model = _load_spectra_model(args)
    fm_model = fm_model.to(device)
    fm_model.eval()

    print("Creating Multimodal Wrapper (Mock LLM)...")
    mock_llm = MockLLM()
    # feature_stats=None forces Raw output from _encode_latent_features
    model_wrapper = MultimodalLlamaModelMultiTokens(
        base_model=mock_llm,
        fm_model=fm_model,
        latent_dim=args.spectral_embedding_dim,
        hidden_dim=args.hidden_dim,
        num_spectral_features=args.num_spectral_features,
        feature_stats=None 
    )
    model_wrapper = model_wrapper.to(device)
    model_wrapper.eval()

    # =========================================================================
    # 3. Comparison
    # =========================================================================
    print("\n--- Starting Comparison (Raw vs Raw) ---")
    
    # Verify alignment
    if len(dataset_1) != len(dataset_2):
        print(f"Warning: Dataset lengths differ ({len(dataset_1)} vs {len(dataset_2)}). Truncating to min.")
        min_len = min(len(dataset_1), len(dataset_2))
        indices = dataset_1.split_indices[:min_len]
        dataset_1.split_indices = indices
        dataset_2.split_indices = indices
    else:
        # Check indices match
        if not np.array_equal(dataset_1.split_indices, dataset_2.split_indices):
            print("Warning: Split indices mismatch even with same seed. Forcing match.")
            dataset_2.split_indices = dataset_1.split_indices

    loader_1 = DataLoader(dataset_1, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    loader_2 = DataLoader(dataset_2, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    
    num_batches = 5
    for i, (batch_1, batch_2) in enumerate(zip(loader_1, loader_2)):
        if i >= num_batches: break
        
        obsids_1 = batch_1['obsids']
        obsids_2 = batch_2['obsids']
        
        # Check obsid match (convert to list for comparison)
        if isinstance(obsids_1, torch.Tensor): obsids_1 = obsids_1.tolist()
        if isinstance(obsids_2, torch.Tensor): obsids_2 = obsids_2.tolist()
        
        # Approximate check if they are strings or ints
        # Just check first element
        # print(f"Batch {i} IDs: {obsids_1[0]} vs {obsids_2[0]}")

        # Features 1: From File (Raw because normalize_features=False)
        feat_1 = batch_1['features'].to(device)
        if feat_1.dim() == 3 and feat_1.shape[1] == 1:
            feat_1 = feat_1.squeeze(1)

        # Features 2: Raw Spectra -> Model -> Latents (Raw)
        raw_spectra = batch_2['features'].to(device)
        
        with torch.no_grad():
            feat_2 = model_wrapper._encode_latent_features(raw_spectra)
        
        # Compare
        if feat_1.shape == feat_2.shape:
             diff = (feat_1 - feat_2).abs()
             print(f"\nBatch {i}:")
             print(f"  Max Diff: {diff.max().item():.6f}")
             print(f"  Mean Diff: {diff.mean().item():.6f}")
             
             display_array_stats(feat_1, "Feat 1 (File)")
             display_array_stats(feat_2, "Feat 2 (Fly)")
             
             # Correlation
             f1_flat = feat_1.view(-1).cpu().numpy()
             f2_flat = feat_2.view(-1).cpu().numpy()
             if len(f1_flat) > 0:
                 corr = np.corrcoef(f1_flat, f2_flat)[0, 1]
                 print(f"  Correlation: {corr:.6f}")
        else:
             print(f"Batch {i}: Shape mismatch {feat_1.shape} vs {feat_2.shape}")

    print("Done.")
if __name__ == "__main__":
    main()
