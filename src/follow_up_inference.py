#!/usr/bin/env python3
"""
Follow-up inference script for the multimodal stellar model.

Loads the trained model and dataset the same way as src/inference.py, then asks:
1. The original dataset question
2. A follow-up question about the stellar type
3. A contrastive follow-up question about a similar-but-different star

All generated answers are written to JSON for downstream analysis.
"""

import argparse
import json
import math
import os
os.system('pip install tiktoken fairscale fire blobfile torchdiffeq torchcfm transformers bitsandbytes accelerate')
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.simple_questions import (  # noqa: E402
    get_model_path,
    setup,
)
from src.simple_questions_multitok import (  # noqa: E402
    build_model_multitok,
    create_datasets_and_loaders,
    ensure_backend_config,
)
from src.tokenizer_adapter import load_tokenizer_adapter, TokenizerAdapter
from src.follow_up_templates import (  # noqa: E402
    PHYSICAL_BOUNDS,
    PARAM_KEY_ALIASES,
    create_follow_up_specs,
)
from data.dataset_diverse import create_diverse_dataloaders
from data.dataset_twostar_inference import create_twostar_inference_dataloader


PHYSICAL_BOUNDS = {
    'Teff': (3000.0, 7500.0),
    'logg': (0.0, 5.0),
    'FeH': (-3.0, 0.5),
}

PARAM_SYNONYMS = {
    'Teff': ['temperature', 'effective temperature', 'Teff', 'stellar temperature', 'thermal profile'],
    'logg': ['surface gravity', 'logg', 'log g', 'gravity at the photosphere'],
    'FeH': ['metallicity', 'iron abundance', '[Fe/H]', 'FeH content'],
}

PARAM_DIRECTION_PHRASES = {
    'Teff': ['a hotter', 'a cooler'],
    'logg': ['higher', 'lower'],
    'FeH': ['more metal-rich', 'more metal-poor'],
}

PARAM_KEY_ALIASES = {
    'Teff': ['Teff', 'teff', 'teff_k', 'effective_temperature'],
    'logg': ['logg', 'log_g', 'log_g_surface'],
    'FeH': ['FeH', 'feh', '[Fe/H]', 'metallicity'],
}


class InferenceProfiler:
    def __init__(self) -> None:
        self.stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {'calls': 0, 'seconds': 0.0, 'tokens': 0.0})

    def record(self, label: str, seconds: float, tokens: int = 0) -> None:
        entry = self.stats[label]
        entry['calls'] += 1
        entry['seconds'] += seconds
        entry['tokens'] += tokens

    def summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for label, data in self.stats.items():
            avg = data['seconds'] / data['calls'] if data['calls'] else 0.0
            tok_rate = (data['tokens'] / data['seconds']) if data['seconds'] > 0 else 0.0
            summary[label] = {
                'calls': int(data['calls']),
                'seconds': data['seconds'],
                'avg_seconds': avg,
                'tokens': int(data['tokens']),
                'tokens_per_second': tok_rate,
            }
        return summary


ACTIVE_PROFILER: Optional[InferenceProfiler] = None


def _profile_record(label: str, start_time: float, tokens: int = 0) -> None:
    if ACTIVE_PROFILER is None:
        return
    duration = time.perf_counter() - start_time
    ACTIVE_PROFILER.record(label, duration, tokens)


def load_luminosity_mass_data(csv_path: str) -> Dict[str, Dict[str, float]]:
    """
    Load Lstar and Mstar from info_full.csv using standard csv module.
    Returns a dict mapping obsid (as string) to {'Lstar': float, 'Mstar': float}.
    """
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found at {csv_path}")
        return {}

    print(f"Loading luminosity and mass data from {csv_path}...")
    data_map = {}
    try:
        import csv
        # Increase field limit just in case, though standard typically suffices for this file
        csv.field_size_limit(1000000)
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Check headers
            if not reader.fieldnames:
                print("Error: CSV file is empty or unreadable")
                return {}
                
            # Handle potential whitespace in headers
            headers = [h.strip() for h in reader.fieldnames]
            if 'obsid' not in headers:
                print(f"Error: 'obsid' column not found in CSV. available: {headers[:5]}...")
                return {}
            
            # Map clean headers to actual headers if needed, but DictReader uses raw fieldnames
            # We'll just access by key matching the raw header if it's clean, 
            # otherwise we might need to be careful. The previous head showed clean names.
            
            count = 0
            for row in reader:
                # Basic safety check
                if 'obsid' not in row:
                    continue
                    
                obsid = str(row['obsid']).strip()
                entry = {}
                
                # Parse Rstar (for prompt context)
                if 'Rstar' in row and row['Rstar']:
                    try:
                        val = float(row['Rstar'])
                        if not(math.isnan(val) or math.isinf(val)):
                            entry['Rstar'] = val
                    except (ValueError, TypeError):
                        pass

                # Parse Lstar
                if 'Lstar' in row and row['Lstar']:
                    try:
                        val = float(row['Lstar'])
                        if not(math.isnan(val) or math.isinf(val)):
                            entry['Lstar'] = val
                    except (ValueError, TypeError):
                        pass

                # Parse Mstar
                if 'Mstar' in row and row['Mstar']:
                    try:
                        val = float(row['Mstar'])
                        if not(math.isnan(val) or math.isinf(val)):
                            entry['Mstar'] = val
                    except (ValueError, TypeError):
                        pass

                # Parse SNR (snrg)
                if 'snrg' in row and row['snrg']:
                    try:
                        val = float(row['snrg'])
                        if not(math.isnan(val) or math.isinf(val)):
                            entry['snr'] = val
                    except (ValueError, TypeError):
                        pass
                
                if entry:
                    data_map[obsid] = entry
                    count += 1
                
        print(f"[OK] Loaded Rstar/Lstar/Mstar data for {count} stars")
        return data_map
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return {}


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(
        model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask follow-up questions with the multimodal model.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config_path', type=str, default=None, help='Optional explicit training config JSON')
    parser.add_argument('--output_dir', type=str, default='follow_up_outputs', help='Directory (relative to chkpt) for JSON output')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit the number of test samples to process')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for dataloader (overrides config)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible follow-up prompts')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Max tokens to sample per generation')
    parser.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature for text generation')
    parser.add_argument('--top_p', type=float, default=0.8, help='Top-p nucleus sampling value')
    parser.add_argument('--json_filename', type=str, default='follow_up_answers.json', help='Filename for the saved JSON payload')
    parser.add_argument('--profile_timings', action='store_true', help='Enable lightweight CPU timing profiler')
    parser.add_argument('--profile_output', type=str, default=None, help='Optional JSON file to dump timing stats')
    parser.add_argument('--max_decode_group', type=int, default=2, help='Maximum number of samples to decode in parallel')
    parser.add_argument('--max_decode_tokens_per_group', type=int, default=4096,
                        help='Upper bound on prompt_length * group_size before we force sequential decoding')
    parser.add_argument('--predict_features',
                        action=argparse.BooleanOptionalAction,
                        default=None,
                        help='Use feature-prediction dataset/mode instead of QA dataset')
    parser.add_argument('--llm_backend', type=str, choices=['llama', 'hf', 'qwen'], default=None,
                        help='Override backbone family (defaults to what training used).')
    parser.add_argument('--dataset_type', type=str, default='standard',
                       choices=['standard', 'diverse', 'two_star'],
                       help='Type of dataset to load: "standard" (original), "diverse" (mixed QA), or "two_star" (comparison inference).')
    
    parser.add_argument('--two_star_test', action='store_true',
                       help='Run special 2-star comparison inference test')

    parser.add_argument('--norm_stats_file', type=str, default=None,
                       help='Path to feature normalization stats file (for diverse dataset)')
    parser.add_argument('--hf_trust_remote_code', action=argparse.BooleanOptionalAction, default=None,
                        help='Allow custom HF modeling code.')
    parser.add_argument('--hf_device_map', type=str, default=None,
                        help='Override HF device map.')
    parser.add_argument('--hf_quantization', type=str, choices=['none', '8bit', '4bit'], default=None,
                        help='Override HF quantization strategy.')
    parser.add_argument('--hf_cache_dir', type=str, default=None,
                        help='Override HF cache dir.')
    parser.add_argument('--hf_max_memory_gb', type=float, default=None,
                        help='Override HF max memory per device.')
    parser.add_argument('--hf_auth_token', type=str, default=None,
                        help='Override HF auth token.')
    parser.add_argument('--llm_precision', type=str, choices=['fp32', 'fp16', 'bf16'], default=None,
                        help='Override base precision for Meta backbone.')
    parser.add_argument('--gradient_checkpointing', action=argparse.BooleanOptionalAction, default=None,
                        help='Request gradient checkpointing (LLaMA backend).')
    
    # Steering arguments
    parser.add_argument('--steering_concept_file', type=str, default=None, help='Path to concept directions (.pt)')
    parser.add_argument('--steering_alphas', type=str, default='0', help='Comma-sep alphas (e.g. -5,0,5)')
    
    parser.add_argument('--v2', action='store_true', default=False,
                        help='Use V2 features path')
    parser.add_argument('--use_multimodal', action='store_true', default=False,
                        help='Use multimodal dataframe features')

    return parser.parse_args()


# Constants for features
FEATURES_PATH_V2 = '/home/ilay.kamai/work/TalkingLatents/logs/2025-12-16/features.npy'
MULTIMODAL_PATH = '/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/multimodal_features.npy'
MULTIMODAL_DF_PATH = '/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/info_full_multimodal.csv'


def load_config_from_checkpoint_dir(checkpoint_path: str) -> Dict[str, Any]:
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, 'training_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No training config found at {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)


def create_args_from_config(config: Dict[str, Any], inference_args: argparse.Namespace) -> argparse.Namespace:
    args = argparse.Namespace(**vars(inference_args))
    for key, value in config.items():
        setattr(args, key, value)
    # Ensure cli batch size overrides config for quicker experimentation
    if getattr(inference_args, 'batch_size', None) is not None:
        args.batch_size = inference_args.batch_size
    # Allow explicit CLI override for predict_features (BooleanOptionalAction yields None if unspecified)
    predict_features_cli = getattr(inference_args, 'predict_features', None)
    if predict_features_cli is not None:
        args.predict_features = predict_features_cli
    else:
        args.predict_features = getattr(args, 'predict_features', False)
    override_keys = [
        'llm_backend', 'hf_model_name', 'hf_revision', 'hf_trust_remote_code',
        'hf_device_map', 'hf_quantization', 'hf_cache_dir', 'hf_max_memory_gb',
        'hf_auth_token', 'llm_precision', 'gradient_checkpointing',
        'use_multimodal'
    ]
    for key in override_keys:
        cli_value = getattr(inference_args, key, None)
        if cli_value is not None:
            setattr(args, key, cli_value)
    return args


def load_model(checkpoint_path: str,
               args: argparse.Namespace,
               device: torch.device,
               model: Optional[torch.nn.Module] = None) -> torch.nn.Module:
    print(f"Loading model from {checkpoint_path}")
    if not hasattr(args, 'llm_precision'):
        args.llm_precision = 'fp16'
    if not hasattr(args, 'gradient_checkpointing'):
        args.gradient_checkpointing = False

    backend_config = ensure_backend_config(args)

    if model is None:
        # Load feature stats if provided (critical for normalization)
        feature_stats = None
        norm_stats_file = getattr(args, 'norm_stats_file', None)
        if norm_stats_file and os.path.exists(norm_stats_file):
            print(f"Loading feature stats from {norm_stats_file}")
            try:
                stats = np.load(norm_stats_file)
                feature_stats = {
                    'mean': stats['mean'],
                    'std': stats['std']
                }
                print("✓ Loaded feature stats")
            except Exception as e:
                print(f"Error loading stats file: {e}")

        model = build_model_multitok(args, device, world_size=1, backend_config=backend_config, feature_stats=feature_stats)
    else:
        model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    del checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith('module.') else key
        cleaned_state_dict[new_key] = value

    target_model = model.module if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model
    missing, unexpected = target_model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        print(f"Warning: Missing keys while loading checkpoint (showing up to 5): {missing[:5]}")
    if unexpected:
        print(f"Warning: Unexpected keys while loading checkpoint (showing up to 5): {unexpected[:5]}")

    model.eval()
    model.to(device)
    print("[OK] Model loaded successfully")
    return model


def seed_everything(seed: Optional[int]) -> random.Random:
    rng = random.Random(seed)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return rng


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> None:
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.to(device)


def sanitize_for_json(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(k): sanitize_for_json(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [sanitize_for_json(item) for item in payload]
    if torch.is_tensor(payload):
        return sanitize_for_json(payload.detach().cpu().tolist())
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    if isinstance(payload, (np.floating,)):
        value = float(payload)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(payload, float):
        return None if math.isnan(payload) or math.isinf(payload) else payload
    return payload


def build_sample_metadata(batch: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
    sample: Dict[str, Any] = {}
    stellar_collection = batch.get('stellar_data')
    if isinstance(stellar_collection, Sequence) and sample_idx < len(stellar_collection):
        sample['stellar_data'] = stellar_collection[sample_idx]

    metadata_collection = batch.get('metadata')
    if isinstance(metadata_collection, Sequence) and sample_idx < len(metadata_collection):
        sample['metadata'] = metadata_collection[sample_idx]

    y_numeric = batch.get('y_numeric')
    if torch.is_tensor(y_numeric) and y_numeric.ndim >= 2:
        sample['y_numeric'] = y_numeric[sample_idx]
    return sample


def extract_physical_param(sample_dict: Dict[str, Any],
                           param: str,
                           bounds: Dict[str, Tuple[float, float]] = PHYSICAL_BOUNDS) -> Optional[float]:
    if sample_dict is None:
        return None

    param_index = {'Teff': 0, 'logg': 1, 'FeH': 2}.get(param)

    def _convert_norm(norm_value: float) -> float:
        low, high = bounds.get(param, (0.0, 1.0))
        return norm_value * (high - low) + low

    y_numeric = sample_dict.get('y_numeric')
    if y_numeric is not None and param_index is not None:
        try:
            tensor = torch.as_tensor(y_numeric, dtype=torch.float32).view(-1)
            if tensor.numel() > param_index:
                value = float(tensor[param_index].item())
                if not math.isnan(value):
                    return _convert_norm(value)
        except Exception:
            pass

    keys = PARAM_KEY_ALIASES.get(param, (param,))
    candidate_dicts: List[Optional[Dict[str, Any]]] = [
        sample_dict.get('stellar_data'),
        sample_dict,
    ]

    meta = sample_dict.get('metadata')
    if isinstance(meta, dict):
        candidate_dicts.append(meta)
        raw_meta = meta.get('raw')
        if isinstance(raw_meta, dict):
            candidate_dicts.extend([
                raw_meta.get('stellar_data'),
                raw_meta.get('star_a'),
                raw_meta.get('star_a_params'),
                raw_meta.get('star_b'),
                raw_meta.get('star_b_params'),
                raw_meta,
            ])

    for mapping in candidate_dicts:
        if not isinstance(mapping, dict):
            continue
        for key in keys:
            value = mapping.get(key)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(numeric):
                continue
            return numeric
    return None


def get_stellar_params(batch: Dict[str, Any], sample_idx: int) -> Dict[str, Optional[float]]:
    sample_meta = build_sample_metadata(batch, sample_idx)
    params = {}
    for param in ['Teff', 'logg', 'FeH']:
        params[param] = extract_physical_param(sample_meta, param)
    return params


def create_follow_up_questions(params: Dict[str, Optional[float]], 
                                rng: random.Random,
                                followup_data: Optional[List] = None,
                                sample_idx: Optional[int] = None,
                                obsid: Optional[Any] = None,
                                l_m_data: Optional[Dict[str, Dict[str, float]]] = None) -> List[Dict[str, str]]:
    """Create followup questions: 1) stellar_type, 2) description, 3) luminosity/mass."""
    followups = []
    
    # Pre-fetch star_info/SNR if available
    snr_val = None
    star_info = None
    if l_m_data is not None and obsid is not None:
        obsid_str = str(obsid)
        star_info = l_m_data.get(obsid_str)
        if star_info:
            snr_val = star_info.get('snr')
    
    # FIRST followup: Generate stellar type question from templates
    stellar_type_specs = create_follow_up_specs(params, rng, max_pairs=1, include_answers=False)
    if stellar_type_specs:
        if snr_val is not None:
            stellar_type_specs[0]['snr'] = snr_val
        followups.append(stellar_type_specs[0])
    
    # SECOND followup: Get description from JSON file (if available)
    if followup_data is not None and sample_idx is not None and 0 <= sample_idx < len(followup_data):
        try:
            item = followup_data[sample_idx]
            description = item.get('description', '')
            if description:
                # Try to parse as JSON
                import json as json_module
                try:
                    parsed = json_module.loads(description)
                    question = parsed.get('question', '')
                    if question:
                        spec = {'type': 'description_followup', 'question': question}
                        if snr_val is not None:
                            spec['snr'] = snr_val
                        followups.append(spec)
                except json_module.JSONDecodeError:
                    pass
        except (IndexError, KeyError, TypeError):
            pass

    # THIRD followup: Lstar and Mstar questions (from info_full.csv)
    if star_info:
        # Add Luminosity question
        if 'Lstar' in star_info:
            question_text = "Estimate the luminosity of the star."
            l_calc = None
                
            if 'Rstar' in star_info:
                question_text = f"Given a radius of {star_info['Rstar']} solar units, estimate the luminosity of the star."
                     
            # Calculate Stefan-Boltzmann Luminosity if Teff is available
            # L/Lsun = (R/Rsun)^2 * (Teff/Tsun)^4
            teff = params.get('Teff')
            if teff is not None:
                try:
                    # Use Teff_sun = 5772 K
                    l_calc = (star_info['Rstar'] ** 2) * ((teff / 5772.0) ** 4)
                except Exception:
                    pass

            followups.append({
                'type': 'luminosity_followup',
                'question': question_text,
                'true_value': 10 ** star_info['Lstar'],
                'Rstar': star_info.get('Rstar'),
                'L_SB_calculated': l_calc,
                'snr': star_info.get('snr')
                })
                
            # Add Mass question
            if 'Mstar' in star_info:
                followups.append({
                    'type': 'mass_followup',
                    'question': "Estimate the mass of the star.",
                    'true_value': star_info['Mstar'],
                    'snr': star_info.get('snr')
                })
    
    return followups


def normalize_mode_label(mode_value: Any) -> str:
    if mode_value is None:
        return 'single_star'
    if isinstance(mode_value, torch.Tensor):
        if mode_value.ndim == 0:
            mode_value = mode_value.item()
        else:
            mode_value = mode_value.tolist()
    if isinstance(mode_value, (list, tuple)):
        mode_value = mode_value[0] if mode_value else None
    if isinstance(mode_value, bytes):
        mode_value = mode_value.decode('utf-8', errors='ignore')
    if isinstance(mode_value, bool):
        return 'two_star' if mode_value else 'single_star'
    if mode_value is None:
        return 'single_star'

    mode_str = str(mode_value).strip().lower()
    if mode_str in {'two_star', 'comparative', 'comparison', 'pair', 'dual', 'two', 'two-star'}:
        return 'two_star'
    if mode_str in {'single_star', 'single', 'one_star', 'singlemode', 'single-star'}:
        return 'single_star'
    return 'single_star'


def infer_batch_modes(batch: Dict[str, Any], batch_size: int) -> List[str]:
    modes = ['single_star'] * batch_size
    explicit_modes = batch.get('mode')
    if explicit_modes:
        for idx in range(min(batch_size, len(explicit_modes))):
            modes[idx] = normalize_mode_label(explicit_modes[idx])

    comp_mask = batch.get('mode_mask_comparative')
    if comp_mask is not None:
        for idx in range(min(batch_size, len(comp_mask))):
            value = comp_mask[idx]
            is_comp = bool(value.item()) if isinstance(value, torch.Tensor) else bool(value)
            if is_comp:
                modes[idx] = 'two_star'
            elif modes[idx] not in ('single_star', 'two_star'):
                modes[idx] = 'single_star'
    return modes


def build_generation_context(batch_data: Dict[str, Any],
                             batch_idx: int) -> Tuple[Dict[str, Any], str, str]:
    """Prepare per-sample tensors and base prompt for iterative follow-up generation."""
    context: Dict[str, Any] = {}
    raw_mode = None
    if 'mode' in batch_data and batch_data['mode']:
        try:
            raw_mode = batch_data['mode'][batch_idx]
        except (IndexError, KeyError, TypeError):
            raw_mode = None
    current_mode = normalize_mode_label(raw_mode)

    if current_mode is None:
        comp_mask = batch_data.get('mode_mask_comparative')
        if comp_mask is not None:
            try:
                if isinstance(comp_mask, torch.Tensor):
                    current_mode = "two_star" if bool(comp_mask[batch_idx].item()) else "single_star"
                else:
                    current_mode = "two_star" if bool(comp_mask[batch_idx]) else "single_star"
            except (IndexError, TypeError):
                current_mode = None

    if current_mode is None:
        has_two_star_features = (
            batch_data.get('masked_spectra_a') is not None and
            batch_data.get('masked_spectra_b') is not None
        )
        current_mode = "two_star" if has_two_star_features else "single_star"

    input_ids = batch_data['input_ids'][batch_idx:batch_idx+1]
    if current_mode == "two_star":
        context.update({
            'star_a_features': batch_data['masked_spectra_a'][batch_idx:batch_idx+1],
            'star_b_features': batch_data['masked_spectra_b'][batch_idx:batch_idx+1],
            'star_a_indices': batch_data['star_a_feature_indices'][batch_idx:batch_idx+1],
            'star_b_indices': batch_data['star_b_feature_indices'][batch_idx:batch_idx+1],
        })
        answer_start_source = batch_data.get('answer_start_indices', [input_ids.shape[1]])
        answer_start_idx = answer_start_source[batch_idx]
        if isinstance(answer_start_idx, torch.Tensor):
            answer_start_idx = answer_start_idx.item()
    else:
        feature_start_raw = batch_data['feature_start_indices'][batch_idx]
        if isinstance(feature_start_raw, torch.Tensor):
            feature_start_idx = feature_start_raw
        else:
            feature_start_idx = torch.tensor(int(feature_start_raw), dtype=torch.long, device=input_ids.device)
        context.update({
            'feature_start_idx': feature_start_idx,
        })
        answer_start_raw = batch_data['answer_start_indices'][batch_idx]
        answer_start_idx = answer_start_raw.item() if isinstance(answer_start_raw, torch.Tensor) else int(answer_start_raw)

    if 'input_texts' in batch_data:
        input_text = batch_data.get('input_texts', [''])[batch_idx]
        target_text = batch_data.get('target_texts', [''])[batch_idx]
    elif 'input_text' in batch_data:
        input_field = batch_data.get('input_text', '')
        target_field = batch_data.get('target_text', '')
        input_text = input_field[batch_idx] if isinstance(input_field, list) else input_field
        target_text = target_field[batch_idx] if isinstance(target_field, list) else target_field
    elif 'metadata' in batch_data and batch_data['metadata']:
        meta = batch_data['metadata'][batch_idx]
        if meta and 'raw' in meta:
            input_text = meta.get('input_text', '')
            target_text = meta.get('target_text', '')
        else:
            input_text = ''
            target_text = ''
    else:
        input_text = ''
        target_text = ''

    prompt = input_ids[:, :max(1, min(answer_start_idx, input_ids.shape[1]))].clone()
    context.update({
        'prompt': prompt,
        'mode': current_mode,
        'input_spectra': batch_data['masked_spectra'][batch_idx:batch_idx+1],
    })
    return context, input_text, target_text


def append_text_to_prompt(context: Dict[str, Any], tokenizer: TokenizerAdapter, text: str) -> None:
    """Append arbitrary text (tokenized) after the current model response."""
    if not text:
        return
    tokens = tokenizer.encode(text, bos=False, eos=False)
    if not tokens:
        return
    tensor = torch.tensor([tokens], dtype=context['prompt'].dtype, device=context['prompt'].device)
    context['prompt'] = torch.cat([context['prompt'], tensor], dim=1)


def _sample_top_p(logits: torch.Tensor, temperature: float, top_p: float, pad_id: Optional[int] = None) -> int:
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        return 0
    
    # Suppress padding
    if pad_id is not None:
        logits[pad_id] = float('-inf')

    logits = torch.clamp(logits, min=-1e4, max=1e4)
    if temperature > 0:
        logits = logits / temperature
    logits = logits - logits.max()
    probs = torch.softmax(logits, dim=-1)
    if torch.isnan(probs).any() or torch.isinf(probs).any():
        probs = torch.ones_like(probs) / probs.numel()
    if 0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        cutoff = (cdf > top_p).float().argmax().item()
        cutoff = max(1, cutoff)
        sorted_probs = sorted_probs[:cutoff]
        sorted_idx = sorted_idx[:cutoff]
        prob_sum = sorted_probs.sum()
        if prob_sum <= 0:
            sorted_probs = torch.ones_like(sorted_probs) / len(sorted_probs)
        else:
            sorted_probs = sorted_probs / prob_sum
        if torch.isnan(sorted_probs).any() or (sorted_probs < 0).any():
            return sorted_idx[0].item()
        next_idx = torch.multinomial(sorted_probs, 1).item()
        return sorted_idx[next_idx].item()
    if torch.isnan(probs).any() or (probs < 0).any():
        return 0
    return torch.multinomial(probs, 1).item()


def _prepare_context_inputs(context: Dict[str, Any],
                            base_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    if 'prepared_inputs' in context:
        return context['prepared_inputs']

    prepared: Dict[str, torch.Tensor] = {}
    mode = context['mode']
    prompt_device = context['prompt'].device
    if mode == "two_star":
        proj_param_a = next(base_model.projector_a.parameters())
        proj_param_b = next(base_model.projector_b.parameters())
        prepared['star_a_features'] = context['star_a_features'].to(
            device=proj_param_a.device, dtype=proj_param_a.dtype)
        prepared['star_b_features'] = context['star_b_features'].to(
            device=proj_param_b.device, dtype=proj_param_b.dtype)
        prepared['star_a_indices'] = context['star_a_indices'].to(device=prompt_device, dtype=torch.long)
        prepared['star_b_indices'] = context['star_b_indices'].to(device=prompt_device, dtype=torch.long)
    else:
        proj_param = next(base_model.projector.parameters())
        features = context['input_spectra'].contiguous().view(context['input_spectra'].size(0), -1)
        prepared['latent_features'] = features.to(device=proj_param.device, dtype=proj_param.dtype)
        feature_idx = context['feature_start_idx'].to(device=prompt_device, dtype=torch.long)
        if feature_idx.ndim == 0:
            feature_idx = feature_idx.view(1)
        prepared['feature_start_idx'] = feature_idx
    context['prepared_inputs'] = prepared
    return prepared


def _build_forward_chunk(base_model: torch.nn.Module,
                         prepared_inputs: Dict[str, torch.Tensor],
                         mode: str,
                         use_cache: bool = False) -> Any:
    if mode == "two_star":
        def _forward(chunk_tokens: torch.Tensor, start_pos: int) -> Dict[str, torch.Tensor]:
            return base_model._forward_two_star_mode(
                input_ids=chunk_tokens,
                star_a_features=prepared_inputs['star_a_features'],
                star_b_features=prepared_inputs['star_b_features'],
                star_a_indices=prepared_inputs['star_a_indices'],
                star_b_indices=prepared_inputs['star_b_indices'],
                start_pos=start_pos,
                cache_rows=None,
                use_cache=use_cache,
            )
        return _forward

    def _forward(chunk_tokens: torch.Tensor, start_pos: int) -> Dict[str, torch.Tensor]:
        return base_model._forward_single_mode(
            input_ids=chunk_tokens,
            latent_features=prepared_inputs['latent_features'],
            feature_start_indices=prepared_inputs['feature_start_idx'],
            start_pos=start_pos,
            cache_rows=None,
            use_cache=use_cache,
        )
    return _forward


@torch.no_grad()
def _decode_generation_requests(model: torch.nn.Module,
                                requests: List[Dict[str, Any]],
                                tokenizer: TokenizerAdapter,
                                max_new_tokens: int,
                                temperature: float,
                                top_p: float,
                                max_parallel: int,
                                max_prompt_tokens: int) -> None:
    if not requests:
        return
    base = _unwrap_model(model)
    base_params = getattr(base.base_model, 'params', None)
    if base_params is not None:
        max_parallel = min(max_parallel, getattr(base_params, 'max_batch_size', max_parallel))

    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for req in requests:
        ctx = req['context']
        key = (ctx['mode'], ctx['prompt'].shape[1])
        grouped.setdefault(key, []).append(req)

    for req_group in grouped.values():
        contexts = [req['context'] for req in req_group]
        prompt_lengths = [ctx['prompt'].shape[1] for ctx in contexts]
        same_prompt_len = len(set(prompt_lengths)) == 1
        prompt_len = prompt_lengths[0]
        total_prompt_tokens = prompt_len * len(contexts)
        can_batch = (
            same_prompt_len and
            len(req_group) > 1 and
            len(req_group) <= max_parallel and
            total_prompt_tokens <= max_prompt_tokens
        )
        if can_batch:
            answers = generate_text_for_group(
                model=model,
                contexts=contexts,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            answers = [
                generate_text_from_context(
                    model=model,
                    context=req['context'],
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                for req in req_group
            ]
        for req, answer in zip(req_group, answers):
            req['on_answer'](answer)


def generate_text_from_context(model: torch.nn.Module,
                               context: Dict[str, Any],
                               tokenizer: TokenizerAdapter,
                               max_new_tokens: int = 100,
                               temperature: float = 0.7,
                               top_p: float = 0.9) -> str:
    """
    Generate text conditioned on the running prompt and update the prompt with
    the tokens that were just produced so the next question follows the answer.
    """
    profile_start = time.perf_counter()
    base_model = _unwrap_model(model)
    base_model.eval()

    prompt = context['prompt']
    mode = context['mode']
    device = prompt.device
    gen_ids: List[int] = []

    prepared_inputs = _prepare_context_inputs(context, base_model)
    forward_chunk = _build_forward_chunk(base_model, prepared_inputs, mode, use_cache=False)

    # Get EOS token IDs - handle both LLaMA and Qwen tokenizers
    eos_ids = []
    if tokenizer is not None:
        if hasattr(tokenizer, 'eos_id'):
            eos_ids.append(tokenizer.eos_id)
        # Qwen special tokens: 151643 (<|im_end|>), 151645 (<|im_start|>)
        if hasattr(tokenizer, 'special_tokens'):
            if '<|im_end|>' in tokenizer.special_tokens:
                eos_ids.append(tokenizer.special_tokens['<|im_end|>'])
            if '<|im_start|>' in tokenizer.special_tokens:
                eos_ids.append(tokenizer.special_tokens['<|im_start|>'])

    pad_id = getattr(tokenizer, 'pad_id', 0)
    
    for i in range(max_new_tokens):
        outputs = forward_chunk(prompt, start_pos=0)
        logits = outputs['logits'][:, -1, :]
        next_token = _sample_top_p(logits[0], temperature=temperature, top_p=top_p, pad_id=pad_id)
        gen_ids.append(next_token)
        
        if next_token in eos_ids:
            break
        prompt = torch.cat([prompt, torch.tensor([[next_token]], device=device, dtype=prompt.dtype)], dim=1)

    context['prompt'] = prompt
    _profile_record(f"generate_text_{mode}", profile_start, len(gen_ids))
    
    # Decode with error handling for invalid tokens
    if tokenizer is not None:
        try:
            return tokenizer.decode(gen_ids)
        except (KeyError, ValueError) as e:
            # Filter out invalid tokens and try again
            print(f"Warning: Token decoding error ({e}), filtering invalid tokens...")
            valid_tokens = []
            for token_id in gen_ids:
                try:
                    tokenizer.decode([token_id])  # Test if token is valid
                    valid_tokens.append(token_id)
                except (KeyError, ValueError):
                    continue  # Skip invalid tokens
            if valid_tokens:
                return tokenizer.decode(valid_tokens)
            return ''
    return ''


def generate_text_for_group(model: torch.nn.Module,
                            contexts: Sequence[Dict[str, Any]],
                            tokenizer: TokenizerAdapter,
                            max_new_tokens: int = 100,
                            temperature: float = 0.7,
                            top_p: float = 0.9) -> List[str]:
    if not contexts:
        return []
    base_model = _unwrap_model(model)
    profile_start = time.perf_counter()
    base_model.eval()
    batch_size = len(contexts)
    mode = contexts[0]['mode']
    prompts = [ctx['prompt'] for ctx in contexts]
    prompt = torch.cat(prompts, dim=0)
    device = prompt.device
    prepared_inputs = [_prepare_context_inputs(ctx, base_model) for ctx in contexts]
    if mode == "two_star":
        star_a_features = torch.cat([pi['star_a_features'] for pi in prepared_inputs], dim=0)
        star_b_features = torch.cat([pi['star_b_features'] for pi in prepared_inputs], dim=0)
        star_a_indices = torch.cat([pi['star_a_indices'] for pi in prepared_inputs], dim=0)
        star_b_indices = torch.cat([pi['star_b_indices'] for pi in prepared_inputs], dim=0)
    else:
        latent_features = torch.cat([pi['latent_features'] for pi in prepared_inputs], dim=0)
        feature_start_idx = torch.cat([pi['feature_start_idx'] for pi in prepared_inputs], dim=0)

    def _forward_chunk(chunk_tokens: torch.Tensor, start_pos: int, use_cache: bool = False) -> Dict[str, torch.Tensor]:
        if mode == "two_star":
            return base_model._forward_two_star_mode(
                input_ids=chunk_tokens,
                star_a_features=star_a_features,
                star_b_features=star_b_features,
                star_a_indices=star_a_indices,
                star_b_indices=star_b_indices,
                start_pos=start_pos,
                cache_rows=None,
                use_cache=use_cache,
            )
        return base_model._forward_single_mode(
            input_ids=chunk_tokens,
            latent_features=latent_features,
            feature_start_indices=feature_start_idx,
            start_pos=start_pos,
            cache_rows=None,
            use_cache=use_cache,
        )

    generated_tokens: List[List[int]] = [[] for _ in contexts]
    alive_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    pad_id = getattr(tokenizer, 'pad_id', 0)
    
    # Get EOS token IDs
    eos_ids = []
    if tokenizer is not None:
        if hasattr(tokenizer, 'eos_id'):
            eos_ids.append(tokenizer.eos_id)
        if hasattr(tokenizer, 'special_tokens'):
            if '<|im_end|>' in tokenizer.special_tokens:
                eos_ids.append(tokenizer.special_tokens['<|im_end|>'])
            if '<|im_start|>' in tokenizer.special_tokens:
                eos_ids.append(tokenizer.special_tokens['<|im_start|>'])

    # Optimized generation loop with KV caching
    start_pos = 0
    curr_input = prompt
    
    for i in range(max_new_tokens):
        if not alive_mask.any():
            break
            
        # Determine if we need to pass full prompt or just the last token
        if i == 0:
            # First pass: process the entire prompt
            pass
        else:
            # Subsequent passes: process only the last generated token
            curr_input = torch.tensor(next_tokens, device=device, dtype=torch.long).unsqueeze(1)

        outputs = _forward_chunk(curr_input, start_pos=start_pos, use_cache=True)
        
        # Determine how many tokens were processed to update start_pos correctly
        num_processed = curr_input.shape[1]
        
        logits = outputs['logits'][:, -1, :]
        next_tokens = []
        for idx in range(batch_size):
            if not alive_mask[idx]:
                next_tokens.append(pad_id)
                continue
            token = _sample_top_p(logits[idx], temperature=temperature, top_p=top_p, pad_id=pad_id)
            next_tokens.append(token)
            generated_tokens[idx].append(token)
            if token in eos_ids:
                alive_mask[idx] = False
        
        # Update start_pos for next iteration
        start_pos += num_processed
        
        # Update prompt trace (optional, but good for debugging/decoding)
        next_tensor = torch.tensor(next_tokens, device=device, dtype=torch.long).unsqueeze(1)
        prompt = torch.cat([prompt, next_tensor], dim=1)

    # Safely decode tokens, filtering out invalid ones
    def safe_decode(token_list):
        if tokenizer is None:
            return ''
        try:
            return tokenizer.decode(token_list)
        except (KeyError, ValueError) as e:
            # Filter out invalid tokens
            valid_tokens = []
            for token_id in token_list:
                try:
                    tokenizer.decode([token_id])
                    valid_tokens.append(token_id)
                except (KeyError, ValueError):
                    continue
            if valid_tokens:
                try:
                    return tokenizer.decode(valid_tokens)
                except (KeyError, ValueError):
                    return ''
            return ''
    
    decoded = [safe_decode(tokens) for tokens in generated_tokens]

    for row_idx, ctx in enumerate(contexts):
        ctx['prompt'] = prompt[row_idx:row_idx+1].clone()
    total_tokens = sum(len(tokens) for tokens in generated_tokens)
    _profile_record(f"batch_generate_{mode}", profile_start, total_tokens)
    return decoded



def main():
    cli_args = parse_args()
    seed_rng = seed_everything(cli_args.seed)
    global ACTIVE_PROFILER
    ACTIVE_PROFILER = InferenceProfiler() if cli_args.profile_timings else None

    if cli_args.config_path:
        with open(cli_args.config_path, 'r') as f:
            config = json.load(f)
    else:
        config = load_config_from_checkpoint_dir(cli_args.checkpoint_path)

    args = create_args_from_config(config, cli_args)
    
    # Handle overrides
    if cli_args.v2:
        print(f"Using V2 features path: {FEATURES_PATH_V2}")
        args.features_file = FEATURES_PATH_V2
        
    if getattr(args, 'use_multimodal', False):
        print("Using MULTIMODAL mode")
        args.features_file = MULTIMODAL_PATH
        
    multimodal_df = None
    if getattr(args, 'use_multimodal', False):
        print(f"Loading multimodal dataframe from {MULTIMODAL_DF_PATH}")
        if os.path.exists(MULTIMODAL_DF_PATH):
            multimodal_df = pd.read_csv(MULTIMODAL_DF_PATH, index_col=0)
            try:
                multimodal_df.index = multimodal_df.index.astype(int)
            except:
                pass
        else:
            print(f"Warning: Multimodal DF path {MULTIMODAL_DF_PATH} not found!")

    device, _, _ = setup()
    print(f"Using device: {device}")

    checkpoint_parent = os.path.dirname(cli_args.checkpoint_path)
    output_dir = cli_args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(checkpoint_parent, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, cli_args.json_filename)

    backend_config = ensure_backend_config(args)
    model = load_model(cli_args.checkpoint_path, args, device)
    backend_config = ensure_backend_config(args)
    tokenizer_backend = backend_config.tokenizer_backend
    tokenizer_path = backend_config.tokenizer_path
    if tokenizer_backend == 'llama' and tokenizer_path is None:
        _, tokenizer_path = get_model_path(args)
        backend_config.tokenizer_path = tokenizer_path
    
    print(f"Loading tokenizer (backend={tokenizer_backend})...")
    try:
        tokenizer = load_tokenizer_adapter(
            backend=tokenizer_backend,
            tokenizer_path=tokenizer_path,
            hf_model_name=backend_config.model_name_or_path if tokenizer_backend != 'llama' else None,
            trust_remote_code=backend_config.trust_remote_code,
            hf_revision=backend_config.revision,
        )
        print(f"[OK] Loaded tokenizer from {tokenizer_path or backend_config.model_name_or_path}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)
    print(f"[OK] Loaded tokenizer from {tokenizer_path}")

    # Load followup JSON data for description questions
    followup_data = None
    followup_json_path = getattr(args, 'followup_json_file', '/home/ilay.kamai/work/TalkingLatents/data/dataset/stellar_descriptions_questions.json')
    if followup_json_path and os.path.exists(followup_json_path):
        try:
            with open(followup_json_path, 'r') as f:
                followup_data = json.load(f)
            print(f"[OK] Loaded {len(followup_data)} followup description samples from {followup_json_path}")
        except Exception as e:
            print(f"Warning: Could not load followup JSON: {e}")

    # Load Luminosity and Mass data (NEW)
    l_m_data = load_luminosity_mass_data('/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/info_full.csv')

    dataset_type = getattr(args, 'dataset_type', 'standard')
    print(f"Dataset type: {dataset_type}")

    dataset_type = getattr(args, 'dataset_type', 'standard')
    print(f"Dataset type: {dataset_type}")

    if cli_args.two_star_test:
        print("Initializing 2-star comparison test mode...")
        # Check if features file is provided
        spectral_features = None
        features_file = getattr(args, 'features_file', None)
        if features_file and os.path.exists(features_file):
            print(f"Loading spectral features from {features_file}")
            spectral_features = np.load(features_file)
        
        _, _, test_loader = create_twostar_inference_dataloader(
            json_file=args.json_file,
            features_array=spectral_features,
            batch_size=getattr(args, 'batch_size', 4),
            tokenizer_path=tokenizer_path,
            tokenizer_backend=tokenizer_backend,
            tokenizer=tokenizer,
            num_spectral_features=getattr(args, 'num_spectral_features', 4),
            multimodal_df=multimodal_df
        )
        if test_loader is not None and hasattr(test_loader, "dataset"):
            setattr(test_loader.dataset, "backend_config", backend_config)

    elif dataset_type == 'diverse':
        # Load features locally for diverse dataset
        spectral_features = None
        features_file = getattr(args, 'features_file', None)
        if features_file and os.path.exists(features_file):
            print(f"Loading spectral features from {features_file}")
            spectral_features = np.load(features_file)
            print(f"Spectral features shape: {spectral_features.shape}")
        
        print(f"Creating diverse QA datasets from {args.json_file}...")
        # We assume max_seq_length is in args or default to 128
        max_len = getattr(args, 'max_seq_length', 128)
        
        # Call the diverse loader
        _, _, test_loader = create_diverse_dataloaders(
            json_file=args.json_file,
            features_array=spectral_features,
            batch_size=getattr(args, 'batch_size', 4),
            train_ratio=0.8, 
            val_ratio=0.2, 
            test_ratio=0.0, # Fixed 1000 samples for test
            random_state=args.seed,
            cache_dir=os.path.join(output_dir, 'cache'),
            tokenizer_path=tokenizer_path,
            tokenizer=tokenizer,
            tokenizer_backend=tokenizer_backend,
            max_length=max_len,
            num_workers=getattr(args, 'num_workers', 4),
            device=device,
            enable_followup=False, # We do inference manually
            num_spectral_features=getattr(args, 'num_spectral_features', 4),
            multimodal_df=multimodal_df
        )
        # Ensure backend config attached
        if test_loader is not None and hasattr(test_loader, "dataset"):
            setattr(test_loader.dataset, "backend_config", backend_config)
            
    else:
        # Standard loader
        _, _, test_loader = create_datasets_and_loaders(args, device, backend_config)

    print(f"Test loader has {len(test_loader.dataset)} samples")

    results: List[Dict[str, Any]] = []
    processed = 0
    
    # Steering setup
    steering_enabled = False
    concepts = {}
    alphas = [0.0]
    if getattr(cli_args, 'steering_concept_file', None):
        print(f"Loading steering concepts from {cli_args.steering_concept_file}")
        map_loc = f"cuda:{device}" if isinstance(device, int) else device
        concepts = torch.load(cli_args.steering_concept_file, map_location=map_loc)
        steering_enabled = True
        if getattr(cli_args, 'steering_alphas', None):
            alphas = [float(x) for x in cli_args.steering_alphas.split(',')]
        print(f"Steering enabled with alphas: {alphas} and concepts: {list(concepts.keys())}")
    else:
        # Default "dummy" concept to run loop once
        concepts = {'none': None}
        alphas = [0.0]

    for batch in tqdm(test_loader, desc="Running follow-up inference"):
        move_batch_to_device(batch, device)
        batch_size = batch['input_ids'].shape[0]
        
        # Save original spectra for steering
        original_masked_spectra = None
        if 'masked_spectra' in batch:
            original_masked_spectra = batch['masked_spectra'].clone()
            
        # Get stats for denormalization/steering if needed
        # We assume dataset has stats if normalisation happened
        # steer_inference.py discussion suggested steer in normalized space with v/sigma
        # or disabled normalization.
        # But here follow_up_inference uses standard loaders -> Normalized.
        # So we should use sigma from dataset to scale v.
        sigma = torch.ones(1).to(device)
        if hasattr(test_loader.dataset, 'get_feature_normalization_stats'):
             stats = test_loader.dataset.get_feature_normalization_stats()
             if stats:
                 # Ensure sigma has correct shape/device
                 sigma = torch.from_numpy(stats['std']).to(device)
                 if sigma.ndim == 1 and original_masked_spectra is not None:
                     if sigma.shape[0] != original_masked_spectra.shape[-1]:
                         # fallback or adjust?
                         pass
        
        batch_modes = infer_batch_modes(batch, batch_size)
        sample_entries: List[Dict[str, Any]] = []
        obsids = batch.get('obsids', [])
        target_texts = batch.get('target_texts', [])
        
        
        # Extract subclasses from stellar_data
        stellar_data_list = batch.get('stellar_data', [])
        subclasses = []
        if isinstance(stellar_data_list, list):
            for sd in stellar_data_list:
                if isinstance(sd, dict):
                    subclasses.append(sd.get('subclass', None))
                else:
                    subclasses.append(None)
        
        # Determine how many samples to process from this batch to respect max_samples
        n_to_process = batch_size
        if cli_args.max_samples is not None:
            remaining = cli_args.max_samples - processed
            if remaining <= 0:
                break
            n_to_process = min(batch_size, remaining)
        
        # Iterate over Steering configurations
        print(f"DEBUG: Entering steering loops. Concepts: {len(concepts)}, Alphas: {len(alphas)}")
        for concept_name, concept_vec in concepts.items():
            print(f"DEBUG: Processing concept: {concept_name}")
            if concept_vec is not None:
                 concept_vec = concept_vec.to(device)
                 # v_norm = v / sigma logic
                 v_norm_space = concept_vec / sigma
            for alpha in alphas:
                 print(concept_name, alpha)
                 # Initialize sample entries for this alpha group
                 sample_entries: List[Dict[str, Any]] = []

                 # Apply steering
                 if steering_enabled and concept_vec is not None and original_masked_spectra is not None:
                      # h_steered = h + alpha * v_norm
                      # h is original_masked_spectra (Normalized)
                      print("v_norm_space shape", v_norm_space.shape)
                      batch['masked_spectra'] = original_masked_spectra + alpha * v_norm_space
                      print('masked spectra shape', batch['masked_spectra'].shape)
                 elif original_masked_spectra is not None:
                      # Restore/Maintain original
                      batch['masked_spectra'] = original_masked_spectra

                 for sample_idx in range(n_to_process):
                    context, dataset_question, dataset_target = build_generation_context(
                        batch_data=batch,
                        batch_idx=sample_idx,
                    )
            
                    # --- Fix for 2-star metadata ---
                    if cli_args.two_star_test:
                        # In 2-star mode, the question and target are in the metadata, not standard fields
                        pass
                
                    # Two star metadata extraction logic
                    batch_meta_list = batch.get('metadata', [])
                    if sample_idx < len(batch_meta_list):
                       curr_meta = batch_meta_list[sample_idx]
                       if isinstance(curr_meta, dict):
                            dataset_question = curr_meta.get('question', dataset_question)
                            dataset_target = curr_meta.get('expected_answer', dataset_target)
                    # -------------------------------

                    stellar_params = get_stellar_params(batch, sample_idx)
                    
                    # Get full stellar_data dict
                    full_stellar_data = None
                    if stellar_data_list and sample_idx < len(stellar_data_list):
                        full_stellar_data = stellar_data_list[sample_idx]
                    
                    target_value = target_texts[sample_idx] if sample_idx < len(target_texts) else dataset_target
                    
                    # Create followup questions
                    if cli_args.two_star_test:
                        follow_up_specs = []
                    else:
                        # Use global sample index (processed + sample_idx) for determinism/indexing if needed
                        follow_up_specs = create_follow_up_questions(
                            stellar_params, 
                            seed_rng,
                            followup_data=followup_data,
                            sample_idx=processed + sample_idx,
                            obsid=obsids[sample_idx] if sample_idx < len(obsids) else None,
                            l_m_data=l_m_data
                        )

                    entry = {
                        'context': context,
                        'dataset_question': dataset_question,
                        'dataset_target': target_value,
                        'stellar_params': stellar_params,
                        'stellar_data': full_stellar_data,
                        'follow_up_specs': follow_up_specs,
                        'obsid': obsids[sample_idx] if len(obsids) > sample_idx else None,
                        'subclass': subclasses[sample_idx] if len(subclasses) > sample_idx else None,
                        'mode_label': batch_modes[sample_idx] if sample_idx < len(batch_modes) else 'single_star',
                        'steering_concept': concept_name if steering_enabled else None,
                        'steering_alpha': alpha if steering_enabled else 0.0
                    }
                    
                    # Add Lstar/Mstar/Rstar/SNR if available
                    obsid_key = entry['obsid']
                    if obsid_key and obsid_key in l_m_data:
                         star_info = l_m_data[obsid_key]
                         entry.update(star_info)
                    
                     # --- Generate Base Answer Immediately ---
                    base_text, _, _, _ = model.generate_response_from_batch(
                        batch_data=batch,
                        batch_idx=sample_idx,
                        tokenizer=tokenizer,
                        max_new_tokens=cli_args.max_new_tokens,
                        temperature=cli_args.temperature,
                        top_p=cli_args.top_p,
                    )
                    entry['base_answer'] = base_text
                    
                    # Update prompt for follow-ups
                    if tokenizer is not None:
                        gen_tokens = tokenizer.encode(base_text, bos=False, eos=False)
                        if gen_tokens:
                            tensor = torch.tensor([gen_tokens],
                                                  dtype=entry['context']['prompt'].dtype,
                                                  device=entry['context']['prompt'].device)
                            entry['context']['prompt'] = torch.cat([entry['context']['prompt'], tensor], dim=1)
                    
                    sample_entries.append(entry)

                 # --- Process Follow-ups for this Alpha Group ---
                 if not sample_entries:
                     continue

                 # Initialize follow-up answers list and prepare prompt
                 for entry in sample_entries:
                     entry['follow_up_answers'] = []
                     append_text_to_prompt(entry['context'], tokenizer, "\n")
 
                 max_follow_ups = max((len(entry['follow_up_specs']) for entry in sample_entries), default=0)
                 for round_idx in range(max_follow_ups):
                     round_requests: List[Dict[str, Any]] = []
                     for entry in sample_entries:
                         if round_idx >= len(entry['follow_up_specs']):
                             continue
                         spec = entry['follow_up_specs'][round_idx]
                         question_text = spec['question'].strip()
                         append_text_to_prompt(
                             entry['context'],
                             tokenizer,
                             f"\nFollow-up question: {question_text}\n",
                         )
 
                         def _make_on_answer(entry=entry, spec=spec, question_text=question_text):
                             def _assign(answer: str) -> None:
                                 result_entry = {k: v for k, v in spec.items()}
                                 result_entry['answer'] = answer
                                 entry['follow_up_answers'].append(result_entry)
                                 append_text_to_prompt(entry['context'], tokenizer, "\n")
                             return _assign
 
                         round_requests.append({
                             'context': entry['context'],
                             'on_answer': _make_on_answer(),
                         })
 
                     _decode_generation_requests(
                         model=model,
                         requests=round_requests,
                         tokenizer=tokenizer,
                         max_new_tokens=cli_args.max_new_tokens,
                         temperature=cli_args.temperature,
                         top_p=cli_args.top_p,
                         max_parallel=cli_args.max_decode_group,
                         max_prompt_tokens=cli_args.max_decode_tokens_per_group,
                     )
                 
                 # Save results
                 for entry_idx, entry in enumerate(sample_entries):
                    entry['context'].pop('prepared_inputs', None) # Cleanup to save memory
                     

                    results.append({
                         'sample_index': processed + entry_idx, # Use global index relative to base samples 
                         'obsid': entry['obsid'],
                         'subclass': entry.get('subclass'),
                         'mode': entry['mode_label'],
                         'dataset_question': entry['dataset_question'],
                         'dataset_target_answer': entry['dataset_target'],
                         'model_answer': entry['base_answer'],
                         'stellar_params': {k: sanitize_for_json(v) for k, v in entry['stellar_params'].items()},
                         'stellar_data': sanitize_for_json(entry.get('stellar_data')),
                         'follow_up_answers': entry['follow_up_answers'],
                         'steering_concept': entry.get('steering_concept'),
                         'steering_alpha': entry.get('steering_alpha'),
                     })
                    print(f"DEBUG: Appended result. Total results: {len(results)}. Concept: {entry.get('steering_concept')}, Alpha: {entry.get('steering_alpha')}")
        
        # Update processed count by the number of unique samples processed in this batch
        processed += n_to_process
        if cli_args.max_samples is not None and processed >= cli_args.max_samples:
             break

    backend_metadata = backend_config.to_dict() if hasattr(backend_config, "to_dict") else backend_config

    metadata = {
        'timestamp': datetime.utcnow().isoformat(),
        'checkpoint_path': cli_args.checkpoint_path,
        'config_path': cli_args.config_path or os.path.join(os.path.dirname(cli_args.checkpoint_path), 'training_config.json'),
        'num_samples': processed,
        'max_samples': cli_args.max_samples,
        'generation': {
            'max_new_tokens': cli_args.max_new_tokens,
            'temperature': cli_args.temperature,
            'top_p': cli_args.top_p,
        },
    }
    if backend_metadata is not None:
        metadata['backend'] = backend_metadata

    payload = {
        'metadata': metadata,
        'samples': sanitize_for_json(results),
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved follow-up generations for {processed} samples to {output_path}")

    if ACTIVE_PROFILER is not None:
        summary = ACTIVE_PROFILER.summary()
        print("[PROFILE] Timing summary:")
        for label, data in summary.items():
            print(f"  - {label}: {data['seconds']:.2f}s over {data['calls']} calls "
                  f"(avg {data['avg_seconds']:.3f}s, {data['tokens_per_second']:.1f} tok/s)")
        if cli_args.profile_output:
            profile_path = cli_args.profile_output
            if not os.path.isabs(profile_path):
                profile_path = os.path.join(checkpoint_parent, profile_path)
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            print(f"[PROFILE] Saved timing summary to {profile_path}")


if __name__ == '__main__':
    main()
