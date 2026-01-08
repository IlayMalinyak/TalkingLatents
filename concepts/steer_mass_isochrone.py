
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
from isochrones.mist import MIST_Isochrone

# Add root directory to path to allow imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.simple_questions_multitok import build_model_multitok, ensure_backend_config
from src.tokenizer_adapter import load_tokenizer_adapter
from data.transforms import Compose, GeneralSpectrumPreprocessor, ToTensor

def get_args():
    parser = argparse.ArgumentParser(description="Steer Mass and verify with Isochrones")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--info_file", type=str, default='/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/info_full.csv')
    parser.add_argument("--features_file", type=str, default='/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/features.npy')
    parser.add_argument("--concept_file", type=str, default='/home/ilay.kamai/work/TalkingLatents/concepts/mass_concept.pt')
    parser.add_argument("--output_dir", type=str, default='/home/ilay.kamai/work/TalkingLatents/concepts/plots_mass')
    parser.add_argument("--num_samples", type=int, default=5, help="Number of stars to steer")
    
    # Model defaults
    parser.add_argument('--llm_backend', type=str, default='llama')
    parser.add_argument('--llm_root', type=str, default='/home/ilay.kamai/work/.llama')
    parser.add_argument('--llm_model', type=str, default='Llama3.1-8B')
    parser.add_argument('--spectral_embedding_dim', type=int, default=2048) 
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_spectral_features', type=int, default=8)
    parser.add_argument('--use_cfm', action='store_true', default=False)
    parser.add_argument('--predict_stellar_params', action='store_true', default=False)
    parser.add_argument('--enable_classification', action='store_true', default=True)
    parser.add_argument('--predict_features', action='store_true', default=False)
    parser.add_argument('--llm_precision', type=str, default='fp16')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False)
    parser.add_argument('--hf_quantization', type=str, default='none')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.159, 0.5, 0.841])

    return parser.parse_args()

def extract_params_from_text(text):
    # Simple regex extraction for Teff, logg, FeH
    import re
    params = {}
    
    # Teff
    m = re.search(r"Teff\s*=\s*([0-9.]+)", text, re.IGNORECASE)
    if m: params['Teff'] = float(m.group(1))
    
    # logg
    m = re.search(r"logg\s*=\s*([0-9.]+)", text, re.IGNORECASE)
    if m: params['logg'] = float(m.group(1))
    
    # L? If model outputs L. "Luminosity = x Lsun"
    m = re.search(r"L\s*=\s*([0-9.]+)\s*Lsun", text, re.IGNORECASE)
    if m: params['L'] = float(m.group(1))
    
    return params

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv(args.info_file)
    features = np.load(args.features_file)
    features = torch.tensor(features)
    
    # Match
    if len(df) != len(features):
        if len(features) < len(df): df = df.iloc[:len(features)]
        else: features = features[:len(df)]
            
    df = df.dropna(subset=['Age', 'Mstar', 'FeH', 'Teff', 'logg'])
    
    # Select sample stars: ~1 solar mass, solar metallicity, young/old
    # Let's pick a few distinct stars
    samples = df[
        (df['Mstar'] > 0.9) & (df['Mstar'] < 1.1) & 
        (df['FeH'] > -0.1) & (df['FeH'] < 0.1)
    ].sample(args.num_samples, random_state=42)
    
    # 2. Load Model
    print("Loading model...")
    backend_config = ensure_backend_config(args)
    model = build_model_multitok(args, device, backend_config=backend_config)
    
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    cleaned_sdict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_sdict, strict=False)
    model.to(device)
    model.eval()
    
    tokenizer_adapter = load_tokenizer_adapter(
       backend=backend_config.tokenizer_backend,
       tokenizer_path=backend_config.tokenizer_path, 
       hf_model_name=backend_config.model_name_or_path
    )
    
    # 3. Load Concept
    concept = torch.load(args.concept_file).to(device)
    # Get feature std for scaling
    # Assuming standard scaling was used for features in training?
    # Usually we steer in normalized space. Here features are raw?
    # extract_mass_concept used features directly. So concept is on raw scale.
    # However, model might expect normalized if dataset did so.
    # We will assume raw features input to steering logic.
    
    # Normalization from dataset? Need stats.
    # We'll use identity for now or try to estimate from batch if needed.
    # Actually, verify if model takes raw 'features' (latents) directly.
    # The 'features.npy' are usually the output of the spectrum encoder. 
    # In 'steer_inference.py', it loads spectra and encodes them.
    # Here we have Pre-computed features 'features.npy'.
    # We can inject them directly?
    # Model `generate_response_from_batch` expects 'masked_spectra' which are usually spectra.
    # BUT, if we want to steer *latents*, we should inject latents.
    # Our model class might need modification or we manually feed embeddings?
    # `model.generate` usually calls `self.encoder(spectra)`.
    # We need to bypass encoder or hook it.
    
    # Wait, 'steer_inference.py' does:
    # h_base = batch['masked_spectra']
    # h_steered = h_base + alpha * v
    # This assumes h_base IS the latent?
    # Let's check `steer_inference.py` again.
    # It does: `h_base = batch['masked_spectra'].to(device)`
    # And calls `model.generate_response_from_batch(..., batch_steered)`
    # In `simple_questions_multitok.py`, `forward` takes `spectra`.
    # If `masked_spectra` are actually Latents, then fine.
    # BUT `create_stellar_dataloaders` returns *spectra*.
    # SO `steer_inference.py` might be steering INPUT SPECTRA? 
    # NO, that would be weird. 'concept_directions.pt' were computed from *output of encoder* (latents).
    # Ah, `extract_concepts.py` computes `x_enc, _ = model.encoder(spectra)`.
    # So `features.npy` are latents.
    # But `steer_inference.py` adds vector to `masked_spectra`. 
    # This implies `masked_spectra` in `steer_inference.py` logic is TREATED as latent inputs?
    # Or `steer_inference.py` is WRONG?
    # Let's check `build_model_multitok`. If it uses `LLMWithLateFusion`, 
    # `forward` checks if input shape is compatible with encoder input.
    # If we pass embeddings to encoder, it might fail.
    
    # CRITICAL: We need to know where to inject the steered latent.
    # The `LLMWithLateFusion` usually takes `spectra`.
    # If we want to steer latents, we must ensure `generate_response_from_batch` uses the steered latent.
    # Let's look at `steer_inference.py` lines 226, 247, 265.
    # It passes `batch_steered['masked_spectra'] = h_steered`.
    # `model.generate_response_from_batch` must handle pre-computed latents if passed?
    # If not, `steer_inference.py` is flawed or assumes `masked_spectra` ARE latents and model skips encoder.
    
    # Let's assume for now that passing latents as `masked_spectra` works IF the model supports it.
    # Many methods check input dimension. Spectra: [B, 1, 4000]. Latents: [B, 2048].
    # If model handles [B, 2048], we are good.
    
    # 4. Loop samples
    mist = MIST_Isochrone()
    alphas = np.linspace(-10, 10, 5) # Steering strength
    
    for idx, row in samples.iterrows():
        print(f"Processing Star {row['obsid']} (Age={row['Age']:.2f}, FeH={row['FeH']:.2f})")
        
        # Get raw latent
        # We start from the 'features.npy' we loaded, so it matches.
        idx_in_features = valid_indices.get_loc(idx)
        latent_base = latents[idx_in_features].to(device)
        
        # Ground Truth
        teff_gt = row['Teff']
        logg_gt = row['logg']
        
        # Generate Isochrone for this star's Age/FeH
        log_age = np.log10(row['Age'] * 1e9)
        iso = mist.isochrone(log_age, row['FeH'])
        
        # Plot Isochrone
        plt.figure(figsize=(8, 8))
        plt.plot(iso['logTeff'], iso['logg'], label=f'Isochrone (Age={row["Age"]:.1f})', color='black', alpha=0.5)
        plt.scatter(np.log10(teff_gt), logg_gt, color='green', marker='*', s=200, label='True Star')
        
        predicted_points = []
        
        for alpha in alphas:
            h_steering = latent_base + alpha * concept
            
            # Prepare batch for model
            # We need to constructing a dummy batch expected by `generate_response_from_batch`
            # We'll need a prompt. "Describe this star."
            
            batch = {
                'masked_spectra': h_steering.unsqueeze(0), # [1, D]
                # Add other keys if model requires (e.g. 'y_numeric' for unused parts?)
            }
            
            # Note: We need to ensure `generate_response_from_batch` constructs the prompt correctly.
            # Usually it uses templates.
            # We'll just define a fixed input_text if possible, or let it default.
            
            try:
                # We interpret usage from steer_inference.py
                # It relies on batch['masked_spectra'] being the latent. 
                # Be careful: generate_response_from_batch might try to run encoder if input looks like spectra.
                # Latent dim 2048 vs Spectra dim 4000+
                
                # Mock generate call (Since we don't have full context of model code in this script, 
                # we assume model handles it)
                response, _, _, _ = model.generate_response_from_batch(
                    batch, 
                    batch_idx=0, 
                    tokenizer=tokenizer_adapter
                )
                
                params = extract_params_from_text(response)
                
                if 'Teff' in params and 'logg' in params:
                    predicted_points.append((params['Teff'], params['logg'], alpha))
                    
            except Exception as e:
                print(f"Inference failed for alpha={alpha}: {e}")
                
        # Plot predicted
        if predicted_points:
            teffs, loggs, als = zip(*predicted_points)
            plt.scatter(np.log10(teffs), loggs, c=als, cmap='coolwarm', label='Steered')
            
            # Connect
            plt.plot(np.log10(teffs), loggs, linestyle='--', color='gray', alpha=0.5)
            
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.xlabel('log Teff')
        plt.ylabel('log g')
        plt.title(f"Mass Steering Validation\nStar {row['obsid']}")
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, f"steering_star_{row['obsid']}.png"))
        plt.close()
        
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
