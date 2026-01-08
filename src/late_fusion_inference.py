#!/usr/bin/env python3
"""
Updated LateFusionModel inference script.
Evaluates the model on the test set, records answers, parameters, and spectral reconstructions,
and generates summary plots.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.distributed as dist
import yaml
from tqdm import tqdm
os.system('pip install tiktoken fairscale fire blobfile torchdiffeq torchcfm transformers bitsandbytes accelerate')


# Ensure project root on path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from data.dataset_diverse import create_diverse_dataloaders
from nn.late_fusion import LateFusionModel
from src.tokenizer_adapter import load_tokenizer_adapter
# src.simple_questions contains _load_llm_model and _load_spectra_model
from src.simple_questions import _load_llm_model, _load_spectra_model

# --- Helper functions copied/adapted from src/late_fusion.py to avoid script import issues ---

def resolve_path(path_like: Optional[str]) -> Optional[Path]:
    if path_like is None:
        return None
    candidate = Path(path_like)
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    return candidate

def load_yaml_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result

def load_model_config(model_section: Dict[str, Any]) -> Dict[str, Any]:
    config_data: Dict[str, Any] = {}
    cfg_path = model_section.get("config_path")
    if cfg_path:
        cfg_file = resolve_path(cfg_path)
        if cfg_file is None:
            raise FileNotFoundError(f"Model config path '{cfg_path}' could not be resolved.")
        if cfg_file.suffix.lower() in {".yaml", ".yml"}:
            config_data = load_yaml_config(cfg_file)
        else:
            with open(cfg_file, "r") as handle:
                config_data = json.load(handle)
    else:
        config_data = model_section.get("config", {})

    overrides = model_section.get("overrides", {})
    if overrides:
        config_data = deep_update(config_data, overrides)
    return config_data

def build_llm_args(llm_cfg: Dict[str, Any], data_cfg: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        llm_root=llm_cfg.get("llm_root"),
        llm_model=llm_cfg.get("llm_model"),
        llm_path=llm_cfg.get("llm_path"),
        batch_size=data_cfg.get("batch_size", 4),
        max_seq_length=data_cfg.get("max_length", 512),
        # Add backend/precision if needed by simple_questions loader
        llm_backend=llm_cfg.get("backend", "llama"),
        llm_precision=llm_cfg.get("llm_precision", "fp16"),
        hf_model_name=llm_cfg.get("llm_model") if llm_cfg.get("backend") == "hf" else None,
    )

def maybe_load_spectral_model(spec_cfg: Dict[str, Any]):
    if not spec_cfg.get("enabled", False):
        return None
    return _load_spectra_model()

def apply_precision(llm_model: torch.nn.Module, precision: str) -> None:
    precision = (precision or "fp16").lower()
    if precision == "fp16":
        llm_model.half()
    elif precision == "bf16":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            llm_model.to(dtype=torch.bfloat16)
        else:
            print("bfloat16 not supported on this device; keeping default precision.")
    elif precision == "fp32":
        llm_model.float()
    else:
        raise ValueError(f"Unsupported precision setting: {precision}")

# --- End helpers ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Late-fusion multi-task inference")
    parser.add_argument("--config", type=str, default="configs/late_fusion.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--output-dir", type=str, default="evaluation/late_fusion", help="Output directory for results and plots.")
    parser.add_argument("--max-samples", type=int, default=100, help="Maximum samples to evaluate.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu).")
    return parser.parse_args()

def load_checkpoint(model: LateFusionModel, checkpoint_path: str, device: torch.device):
    print(f"Loading checkpoint from {checkpoint_path}...")
    # Load to CPU first to avoid OOM on GPU (checkpoint can be very large)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    
    # Handle DDP-wrapped checkpoints if necessary
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    print("✓ Model weights loaded.")
    
    # Clear memory
    del ckpt
    del state_dict
    del new_state_dict
    torch.cuda.empty_cache()

def plot_stellar_params(results: List[Dict[str, Any]], output_path: Path):
    """Plot predicted vs true stellar parameters."""
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    param_names = ["Teff", "logg", "Fe/H"]
    keys = ["Teff", "logg", "Fe_H"] 
    
    # Use vibrant, harmonious colors
    colors = ["#00d4ff", "#ff007f", "#39ff14"] # Cyan, Pink, Neon Green
    
    for i, (name, key) in enumerate(zip(param_names, keys)):
        true_vals = []
        pred_vals = []
        
        for res in results:
            t = res["true_params"].get(key)
            idx = i
            p = res.get("pred_params_numeric", [None]*3)[idx]
            
            if t is not None and p is not None:
                true_vals.append(t)
                pred_vals.append(p)
        
        if not true_vals:
            axes[i].text(0.5, 0.5, f"No data for {name}", ha='center', color='gray')
            continue
            
        true_vals = np.array(true_vals)
        pred_vals = np.array(pred_vals)
        
        # Plot with subtle shadow/glow effect
        axes[i].scatter(true_vals, pred_vals, alpha=0.7, color=colors[i], edgecolors='white', linewidth=0.5, s=50, zorder=3)
        
        # Identity line
        limit_min = min(true_vals.min(), pred_vals.min())
        limit_max = max(true_vals.max(), pred_vals.max())
        axes[i].plot([limit_min, limit_max], [limit_min, limit_max], color='white', linestyle='--', alpha=0.3, zorder=2)
        
        axes[i].set_title(f"{name} Evaluation", fontsize=16, fontweight='bold', pad=15)
        axes[i].set_xlabel("Ground Truth", fontsize=12, labelpad=10)
        axes[i].set_ylabel("Model Prediction", fontsize=12, labelpad=10)
        
        # Add metrics in a sleek box
        mae = np.mean(np.abs(true_vals - pred_vals))
        corr = np.corrcoef(true_vals, pred_vals)[0, 1]
        text_box = f"MAE: {mae:.3f}\nρ: {corr:.3f}"
        axes[i].text(0.05, 0.95, text_box, 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#1e1e1e', alpha=0.8, edgecolor=colors[i]))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='#121212')
    plt.close()
    print(f"✓ Saved stellar parameters plot to {output_path}")

def plot_spectral_reconstruction(results: List[Dict[str, Any]], output_path: Path, num_examples=5):
    """Plot example spectral reconstructions with a premium look."""
    plt.style.use('dark_background')
    num_plots = min(len(results), num_examples)
    if num_plots == 0:
        return
        
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 5 * num_plots))
    if num_plots == 1:
        axes = [axes]
        
    for i in range(num_plots):
        res = results[i]
        true_spec = np.array(res["true_spectrum"]).flatten()
        pred_spec = np.array(res["pred_spectrum"]).flatten()
        
        x = np.arange(len(true_spec))
        
        # Plot with gradient-like fills
        axes[i].plot(x, true_spec, label="Target Latent", alpha=0.5, color='#aaaaaa', linewidth=1)
        axes[i].plot(x, pred_spec, label="Perceiver Reconstruction", alpha=1.0, color='#00d4ff', linewidth=2)
        
        axes[i].set_title(f"Spectral Latent Reconstruction • ObsID {res['obsid']}", fontsize=14, fontweight='bold', loc='left', pad=10)
        axes[i].legend(loc='upper right', frameon=True, facecolor='#1e1e1e')
        axes[i].set_xlabel("Latent Dim", fontsize=10)
        axes[i].set_ylabel("Value", fontsize=10)
        
        # Glow effect for prediction
        axes[i].fill_between(x, pred_spec, color='#00d4ff', alpha=0.1)
        
        # Metrics
        mse = np.mean((true_spec - pred_spec)**2)
        axes[i].text(0.01, 0.02, f"Latent MSE: {mse:.2e}", transform=axes[i].transAxes, 
                    fontsize=10, color='#00d4ff', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='#121212')
    plt.close()
    print(f"✓ Saved spectral reconstruction examples to {output_path}")

def main():
    args = parse_args()
    
    # Initialize distributed process group and fairscale for LLaMA
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "13555"
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=0, world_size=1)
        
    try:
        import fairscale.nn.model_parallel.initialize as fs_init
        if not fs_init.model_parallel_is_initialized():
            fs_init.initialize_model_parallel(1)
    except ImportError:
        print("Warning: fairscale not installed, but might be required for LLaMA model.")

    config = load_yaml_config(resolve_path(args.config))
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    seed_everything(config.get("experiment", {}).get("seed", 42))
    
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # 1. Load Data
    data_cfg = config.get("data", {})
    llm_cfg = config.get("llm", {})
    
    # Force use_followups=True for inference to get follow-up turns
    data_cfg["use_followups"] = True
    data_cfg["followup_prob"] = 1.0 # Ensure we get them if available
    
    # Load tokenizer
    llm_backend = llm_cfg.get("backend", "llama")
    tokenizer_path = resolve_path(data_cfg.get("tokenizer_path") or llm_cfg.get("tokenizer_path"))
    tokenizer = load_tokenizer_adapter(
        backend=llm_backend,
        tokenizer_path=str(tokenizer_path) if tokenizer_path else None,
        hf_model_name=llm_cfg.get("llm_model") if llm_backend != 'llama' else None,
    )
    
    # Features loading
    features_file = resolve_path(data_cfg.get("features_file"))
    features_array = None
    if features_file and features_file.exists():
        print(f"Loading spectral features from {features_file}")
        features_array = np.load(features_file)
        
    # Update dataset_kwargs
    dataset_kwargs = dict(data_cfg.get("dataset_kwargs") or {})
        
    _, _, test_loader = create_diverse_dataloaders(
        json_file=str(resolve_path(data_cfg["json_file"])),
        features_array=features_array,
        batch_size=4, 
        train_ratio=data_cfg.get("train_ratio", 0.7),
        val_ratio=data_cfg.get("val_ratio", 0.15),
        test_ratio=data_cfg.get("test_ratio", 0.15),
        random_state=data_cfg.get("random_state", 42),
        num_workers=0,
        world_size=1,
        max_length=data_cfg.get("max_length", 512),
        tokenizer=tokenizer,
        tokenizer_backend=llm_backend,
        **dataset_kwargs
    )
    
    # 2. Build Model
    llm_args = build_llm_args(llm_cfg, data_cfg)
    # Using _load_llm_model from simple_questions which handles both HF and Llama
    # but expects SimpleNamespace or args object
    llm_model = _load_llm_model(llm_args)
    apply_precision(llm_model, llm_cfg.get("llm_precision", "fp16"))
    llm_model = llm_model.to(device)
    
    spectral_model = maybe_load_spectral_model(config.get("spectral_model", {}))
    if spectral_model is not None:
        spectral_model = spectral_model.to(device)
        
    perceiver_config = load_model_config(config.get("model", {}))
    model = LateFusionModel(llm_model, spectral_model, perceiver_config).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    
    # 3. Inference Loop
    results = []
    processed = 0
    max_samples = args.max_samples
    
    # Force bfloat16 for inference to handle large spectral values (avoid FP16 overflow)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using precision: {dtype}")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            if processed >= max_samples:
                break
            
            # Move batch to device
            if isinstance(batch, dict):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
            
            # with torch.cuda.amp.autocast(dtype=dtype):
                # Run base forward to get predictions
                outputs = model(batch)
                
                # Predict text for base description
                try:
                    gen_text_base, _, target_text_base, _ = model.generate_response_from_batch(
                        batch, 
                        batch_idx=0, 
                        tokenizer=tokenizer,
                        max_new_tokens=64
                    )
                except Exception as e:
                    print(f"Error generating base response: {e}")
                    gen_text_base = "Error"
                    target_text_base = "N/A"
                    import traceback
                    traceback.print_exc()

                # Reconstructed features
                # outputs["spectral_reconstruction"] is [B, N]
                pred_spec = outputs["spectral_reconstruction"][0].float().cpu().numpy()
                true_spec = outputs["spectral_targets"][0].squeeze().float().cpu().numpy() # Use input spectra as truth
                
                # Stellar parameters
                # Denormalize as in evaluate_validation_samples
                pred_params_tensor = outputs["stellar_prediction"][0]
                pred_teff = pred_params_tensor[0].item() * 5700.0
                pred_logg = pred_params_tensor[1].item()
                pred_feh = pred_params_tensor[2].item()
                pred_params_numeric = [pred_teff, pred_logg, pred_feh]
                
                stellar_data = batch["stellar_data"][0] if "stellar_data" in batch else {}
                true_params = {
                    "Teff": stellar_data.get("Teff"),
                    "logg": stellar_data.get("logg"),
                    "Fe_H": stellar_data.get("Fe_H")
                }
                
                # Handle follow-up turns
                turns_results = []
                followup_turns_raw = batch.get("followup_turns", [[]])[0]
                
                for q_text, a_text in followup_turns_raw:
                    turn_batch = {k: v for k, v in batch.items()}
                    
                    # Encode the question
                    q_tokens = tokenizer.encode(f"\nFollow-up question: {q_text}\nAnswer:", bos=False, eos=False)
                    turn_batch["followup_input_ids"] = torch.tensor([q_tokens], device=device)
                    # Create labels - just dummy -100 as we are generating
                    turn_batch["followup_labels"] = torch.full((1, len(q_tokens)), -100, device=device, dtype=torch.long)
                    
                    try:
                        gen_f_text, _, _, _ = model.generate_response_from_batch(
                            turn_batch,
                            batch_idx=0,
                            tokenizer=tokenizer,
                            max_new_tokens=64
                        )
                    except Exception as e:
                        print(f"Error generating followup response: {e}")
                        gen_f_text = "Error"
                    
                    turns_results.append({
                        "question": q_text,
                        "true_answer": a_text,
                        "generated_answer": gen_f_text
                    })

                sample_res = {
                    "obsid": str(batch["obsids"][0]) if "obsids" in batch else str(batch.get("obsid", ["unknown"])[0]),
                    "base_question": batch["input_texts"][0] if "input_texts" in batch else "",
                    "base_true_answer": target_text_base,
                    "base_generated_answer": gen_text_base,
                    "followup_turns": turns_results,
                    "true_params": true_params,
                    "pred_params_numeric": pred_params_numeric,
                    "true_spectrum": true_spec.tolist(),
                    "pred_spectrum": pred_spec.tolist()
                }
                results.append(sample_res)
                processed += 1
            
    # 4. Save results
    json_path = output_dir / "late_fusion_followups.json"
    
    # Exclude true_spectrum from JSON to avoid large file size/mess
    results_to_save = []
    for r in results:
        r_copy = r.copy()
        if "true_spectrum" in r_copy:
            del r_copy["true_spectrum"]
        results_to_save.append(r_copy)
        
    with open(json_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"✓ Saved results to {json_path}")
    
    # 5. Plotting
    plot_stellar_params(results, output_dir / "stellar_params_comparison.png")
    plot_spectral_reconstruction(results, output_dir / "spectral_reconstruction_examples.png")

if __name__ == "__main__":
    main()
