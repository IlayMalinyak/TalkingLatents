#!/usr/bin/env python3
"""
Inference script for the multimodal stellar model.
Loads a trained model, runs predictions on test set, and saves results.
"""

import os
import gc
import sys
import json
import argparse
import random
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Tuple, Optional, Union, Set
from copy import deepcopy
from tqdm import tqdm
os.system("pip install -q tiktoken fairscale fire blobfile torchdiffeq torchcfm >/dev/null 2>&1")

# Add root directory to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.simple_questions import (
    _load_llm_model,
    _load_spectra_model,
    get_model_path,
    create_optimizer_and_scheduler,
    setup
)
from src.simple_questions_multitok import (
    create_datasets_and_loaders,
    build_model_multitok,
    ensure_backend_config,
)
from src.tokenizer_adapter import load_tokenizer_adapter
from nn.llm_multi import MultimodalLlamaModelMultiTokens
from nn.train import LLMTrainer
from nn.optim import CQR
from data.transforms import GeneralSpectrumPreprocessor, ToTensor, Compose
import torch.nn.functional as F
import re
from src.baseline_mlp import MLPRegressor, normalize_batch, build_target_denormalizers
try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None
    try:
        from sklearn.neighbors import KDTree as SKKDTree
    except ImportError:
        SKKDTree = None
else:
    SKKDTree = None


def unnormalize_stellar_params(normalized_params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Unnormalize stellar parameters using the same bounds as in dataset_mixed.py"""
    
    # Stellar parameter bounds from dataset_mixed.py
    BOUNDS = {
        'MAX_TEFF': 7500, 'MIN_TEFF': 3000,
        'MAX_LOGG': 5.0, 'MIN_LOGG': 0,
        'MAX_FE_H': 0.5, 'MIN_FE_H': -3
    }
    
    unnormalized = {}
    
    for param, values in normalized_params.items():
        if param == 'Teff':
            unnormalized[param] = values * (BOUNDS['MAX_TEFF'] - BOUNDS['MIN_TEFF']) + BOUNDS['MIN_TEFF']
        elif param == 'logg':
            unnormalized[param] = values * (BOUNDS['MAX_LOGG'] - BOUNDS['MIN_LOGG']) + BOUNDS['MIN_LOGG']
        elif param == 'FeH':
            unnormalized[param] = values * (BOUNDS['MAX_FE_H'] - BOUNDS['MIN_FE_H']) + BOUNDS['MIN_FE_H']
        else:
            # For unknown parameters, return as-is
            unnormalized[param] = values
    
    return unnormalized


def extract_stellar_params_from_text(text: str) -> Dict[str, float]:
    """Extract stellar parameters from generated text"""
    
    params = {}
    
    # Define patterns for different parameter formats (handles spaces in decimals)
    patterns = {
        'Teff': [
            r'(?:T_?eff|effective temperature|temperature)[\s:=~]*(?:is\s+)?(?:approximately\s+)?(\d+(?:\.\s*\d+)?)\s*K?',
            r'(\d+(?:\.\s*\d+)?)\s*K',  # Pattern for temperature with optional spaced decimals
            r'(\d+\.\s*\d+)',  # Explicit spaced decimal pattern
        ],
        'logg': [
            r'(?:log\s*g|logg|surface gravity)[\s:=~]*(?:is\s+)?(?:approximately\s+)?(\d+(?:\.\s*\d+)?)',
            r'(?:log\s*g|logg)[\s:=~]*(\d+(?:\.\s*\d+)?)',
            r'(\d+\.\s*\d+)',  # Explicit spaced decimal pattern for logg
        ],
        'FeH': [
            r'\[fe/h\][\s:=~]*(?:is\s+)?(?:of\s+)?([+-]?\d+\.\s*\d+)',  # Decimal first - more specific [fe/h] pattern
            r'\[fe/h\][\s:=~]*(?:is\s+)?(?:of\s+)?([+-]?\d+)',  # Integer fallback
            r'(?:FeH|metallicity)[\s:=~]*(?:is\s+)?(?:of\s+)?([+-]?\d+\.\s*\d+)',  # Decimal first
            r'(?:FeH|metallicity)[\s:=~]*(?:is\s+)?(?:of\s+)?([+-]?\d+)',  # Integer fallback
            r'([+-]\d+\.\s*\d+)',  # Explicit negative decimal pattern
        ]
    }
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    for param, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                try:
                    # Take the first valid match and remove any spaces from decimal
                    raw_match = matches[0]
                    cleaned_match = raw_match.replace(' ', '')  # Remove spaces from "3. 96" -> "3.96"
                    value = float(cleaned_match)
                    params[param] = value
                    # Debug: print extracted values for first few samples
                    global extraction_count
                    if not hasattr(extract_stellar_params_from_text, 'extraction_count'):
                        extract_stellar_params_from_text.extraction_count = 0
                    extract_stellar_params_from_text.extraction_count += 1
                    
                    # if extract_stellar_params_from_text.extraction_count <= 10 or param == 'FeH':  # Show more FeH extractions
                    #     print(f"  Extracted {param} = {value} (type: {type(value)}) from raw_match='{raw_match}' -> cleaned='{cleaned_match}' in text snippet: '{text_lower[:100]}...'")
                    break  # Stop after finding first valid match for this parameter
                except (ValueError, IndexError):
                    continue
    
    return params


MODE_COLOR_MAP = {
    'single_star': 'tab:blue',
    'two_star': 'tab:orange',
}

MODE_DISPLAY_NAMES = {
    'single_star': 'Single Star',
    'two_star': 'Two Star (comparative)',
}

PHYSICAL_BOUNDS = {
    'Teff': (3000.0, 7500.0),
    'logg': (0.0, 5.0),
    'FeH': (-3.0, 0.5),
}

def normalize_mode_label(mode_value) -> str:
    """Convert various mode encodings to canonical labels."""
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
    """Infer per-sample mode labels for a batch."""
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


def average_losses_by_epoch(loss_list: List[Any], num_epochs: int) -> List[float]:
    """Average a per-iteration loss list into per-epoch values."""
    if not loss_list or num_epochs <= 0:
        return []

    sanitized: List[float] = []
    for loss in loss_list:
        if loss is None:
            sanitized.append(float('nan'))
        else:
            try:
                sanitized.append(float(loss))
            except (TypeError, ValueError):
                sanitized.append(float('nan'))

    arr = np.asarray(sanitized, dtype=float)
    if arr.size < num_epochs:
        return arr.tolist()

    splits = np.array_split(arr, num_epochs)
    averaged: List[float] = []
    for split in splits:
        valid = split[~np.isnan(split)]
        averaged.append(float(valid.mean()) if valid.size > 0 else float('nan'))

    return averaged






def parse_interpolation_pairs(pair_strings: Optional[List[str]]) -> List[Tuple[int, int]]:
    """Parse interpolation pair strings formatted as 'idx_a:idx_b'."""
    pairs: List[Tuple[int, int]] = []
    if not pair_strings:
        return pairs
    for entry in pair_strings:
        if not entry:
            continue
        try:
            left, right = entry.split(':')
            pairs.append((int(left.strip()), int(right.strip())))
        except ValueError:
            print(f"Warning: Could not parse interpolation pair '{entry}'. Expected format 'idx_a:idx_b'.")
    return pairs


def get_single_star_latent_dim(trainer: LLMTrainer) -> Optional[int]:
    projector = getattr(trainer.model, 'projector', None)
    if projector is not None and hasattr(projector, 'mlp'):
        first_layer = projector.mlp[0]
        if isinstance(first_layer, torch.nn.Linear):
            return first_layer.in_features
    return None
    


def get_tensor_from_sample(sample: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    value = sample.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value.detach().clone()
    else:
        try:
            tensor = torch.as_tensor(value)
        except Exception:
            return None
    tensor = tensor.float()
    if tensor.ndim == 0:
        return None
    return tensor


def find_interpolatable_tensor(sample: Dict[str, Any], target_dim: Optional[int]) -> Optional[torch.Tensor]:
    mode = normalize_mode_label(sample.get('mode'))
    if mode != 'single_star':
        return None
    candidate_keys = ['masked_spectra', 'features', 'spectra', 'x_raw']
    for key in candidate_keys:
        tensor = get_tensor_from_sample(sample, key)
        if tensor is None:
            continue
        if target_dim is not None and tensor.numel() != target_dim:
            continue
        return tensor
    return None


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

    key_alternatives = {
        'Teff': ('Teff', 'teff_k', 'teff', 'effective_temperature'),
        'logg': ('logg', 'log_g', 'log_g_surface'),
        'FeH': ('FeH', 'feh', '[Fe/H]', 'metallicity'),
    }
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

    keys = key_alternatives.get(param, (param,))
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


def collect_kiel_background(dataset) -> Tuple[np.ndarray, np.ndarray]:
    teff_values: List[float] = []
    logg_values: List[float] = []
    total = len(dataset)
    for idx in range(total):
        try:
            sample = dataset[idx]
        except Exception as exc:
            print(f"Warning: Failed to load sample {idx} for background ({exc})")
            continue
        teff = extract_physical_param(sample, 'Teff')
        logg = extract_physical_param(sample, 'logg')
        if teff is None or logg is None:
            continue
        if math.isnan(teff) or math.isnan(logg):
            continue
        teff_values.append(teff)
        logg_values.append(logg)
    if not teff_values or not logg_values:
        return np.array([]), np.array([])
    return np.asarray(teff_values, dtype=float), np.asarray(logg_values, dtype=float)


def sanitize_for_json(payload: Any) -> Any:
    """Convert nested structures to JSON-safe formats."""
    if isinstance(payload, dict):
        return {str(key): sanitize_for_json(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [sanitize_for_json(item) for item in payload]
    if isinstance(payload, torch.Tensor):
        return sanitize_for_json(payload.detach().cpu().tolist())
    if isinstance(payload, np.ndarray):
        return sanitize_for_json(payload.tolist())
    if isinstance(payload, (np.floating,)):
        value = float(payload)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(payload, float):
        return None if math.isnan(payload) or math.isinf(payload) else payload
    if isinstance(payload, (int, str, bool)) or payload is None:
        return payload
    try:
        # Attempt generic conversion (e.g., dataclasses with asdict)
        return sanitize_for_json(payload.__dict__)
    except AttributeError:
        return payload


def run_feature_interpolation(trainer: LLMTrainer,
                             args,
                               dataset,
                               collate_fn,
                               device: Union[int, torch.device],
                               pairs: List[Tuple[int, int]],
                               alphas: List[float],
                               output_dir: str,
                               quantiles: List[float],
                               min_teff_diff: float = 0.0,
                               min_logg_diff: float = 0.0) -> List[Dict[str, Any]]:
    """Interpolate latent features between sample pairs and record predicted stellar parameters."""
    if not pairs:
        return []

    model = getattr(trainer, 'model', None)
    model_was_training = False
    if model is not None:
        model_was_training = model.training
        model.eval()

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

    if isinstance(device, int):
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{device}')
        else:
            device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device) and 'cuda' in device.type and not torch.cuda.is_available():
        device = torch.device('cpu')

    collate = collate_fn
    if collate is None and hasattr(dataset, 'collate_fn'):
        collate = dataset.collate_fn
    if collate is None:
        print("Warning: No collate function available; skipping interpolation analysis.")
        return []
    

    def _retokenize_single_prompt(sample_dict: Dict[str, Any], prompt: str) -> None:
        if tokenizer is None:
            return
        if sample_dict.get('mode') not in {None, 'single'}:
            return
        token_key = None
        tokens = sample_dict.get('tokens')
        if tokens is not None and torch.is_tensor(tokens):
            token_key = 'tokens'
        else:
            tokens = sample_dict.get('input_ids')
            if tokens is None or not torch.is_tensor(tokens):
                return
            token_key = 'input_ids'

        had_metadata = 'metadata' in sample_dict
        metadata = sample_dict.get('metadata') or {}

        if 'feature_start_idx' in sample_dict and sample_dict.get('feature_start_idx') is not None:
            feature_start = int(sample_dict.get('feature_start_idx') or 0)
        else:
            feature_start = int(metadata.get('feature_start_idx', 0) or 0)

        if 'feature_length' in sample_dict and sample_dict.get('feature_length') is not None:
            feature_length = int(sample_dict.get('feature_length') or 0)
        else:
            feature_length = int(metadata.get('feature_length', 0) or 0)
        seq_len = tokens.numel()

        pad_id = tokenizer.pad_id if hasattr(tokenizer, 'pad_id') else 0
        new_tokens = torch.full_like(tokens, pad_id)

        if feature_length > 0 and feature_start + feature_length <= seq_len:
            new_tokens[feature_start:feature_start + feature_length] = -100
        prompt_ids = tokenizer.encode(prompt, bos=True, eos=False)
        prompt_tensor = torch.tensor(prompt_ids, dtype=tokens.dtype, device=tokens.device)
        max_prompt_len = max(0, seq_len - (feature_start + feature_length))
        prompt_tensor = prompt_tensor[:max_prompt_len]
        start = feature_start + feature_length
        end = start + prompt_tensor.numel()
        if prompt_tensor.numel() > 0:
            new_tokens[start:end] = prompt_tensor

        sample_dict[token_key] = new_tokens
        if token_key != 'tokens' and sample_dict.get('tokens') is not None and torch.is_tensor(sample_dict['tokens']):
            sample_dict['tokens'] = new_tokens.clone()
        if token_key != 'input_ids' and sample_dict.get('input_ids') is not None and torch.is_tensor(sample_dict['input_ids']):
            sample_dict['input_ids'] = new_tokens.clone()

        sample_dict['target_ids'] = torch.full_like(new_tokens, -100)

        span = sample_dict.get('span_indices')
        if torch.is_tensor(span) and span.numel() >= 2:
            span[0] = end
            span[1] = end

        sample_dict['input_text'] = prompt
        sample_dict['target_text'] = ''

        if 'question_start_idx' in sample_dict:
            sample_dict['question_start_idx'] = start
        if 'answer_start_idx' in sample_dict:
            sample_dict['answer_start_idx'] = end
        if 'target_length' in sample_dict:
            sample_dict['target_length'] = 0
        if 'input_length' in sample_dict:
            sample_dict['input_length'] = prompt_tensor.numel()
        if 'feature_start_idx' in sample_dict:
            sample_dict['feature_start_idx'] = feature_start
        if 'feature_length' in sample_dict:
            sample_dict['feature_length'] = feature_length

        metadata['input_text'] = prompt
        metadata['target_text'] = ''
        metadata['question_start_idx'] = start
        metadata['answer_start_idx'] = end
        metadata['target_length'] = 0
        if 'raw' in metadata and isinstance(metadata['raw'], dict):
            raw_meta = metadata['raw']
            raw_meta['input_text'] = prompt
            raw_meta['target_text'] = ''
            raw_meta['question_start_idx'] = start
            raw_meta['answer_start_idx'] = end
            raw_meta['target_length'] = 0
            raw_meta['feature_start_idx'] = feature_start
            raw_meta['feature_length'] = feature_length
            raw_meta['input_ids'] = new_tokens.detach().cpu().clone()
            raw_meta['target_ids'] = torch.full_like(new_tokens, -100).detach().cpu()
        if had_metadata:
            sample_dict['metadata'] = metadata
        elif metadata:
            sample_dict['metadata'] = metadata

    target_dim = get_single_star_latent_dim(trainer)

    os.makedirs(output_dir, exist_ok=True)

    anchor_labels_drawn = {'start': False, 'end': False}

    bounds = PHYSICAL_BOUNDS
    median_idx = len(quantiles) // 2 if quantiles else 0
    results: List[Dict[str, Any]] = []
    feature_candidate_keys = ['masked_spectra', 'features', 'spectra', 'x_raw']

    def _convert_norm_to_physical(param: str, norm_value: float) -> float:
        if np.isnan(norm_value):
            return float('nan')
        low, high = bounds.get(param, (0.0, 1.0))
        return norm_value * (high - low) + low

    def _tensor_to_median(tensor: torch.Tensor) -> float:
        tensor = tensor.detach().float()
        if tensor.ndim == 0:
            return tensor.item()
        if tensor.ndim == 1:
            idx = min(median_idx, tensor.numel() - 1)
            return tensor[idx].item()
        idx = min(median_idx, tensor.size(-1) - 1)
        return tensor[..., idx].item()

    def _extract_norm_predictions(preds: Any) -> Dict[str, float]:
        param_order = ['Teff', 'logg', 'FeH']
        norm: Dict[str, float] = {}
        if preds is None:
            return {param: float('nan') for param in param_order}
        if isinstance(preds, dict):
            for param in param_order:
                tensor = preds.get(param)
                if tensor is None:
                    norm[param] = float('nan')
                    continue
                if tensor.ndim > 1:
                    norm[param] = _tensor_to_median(tensor[0])
                else:
                    norm[param] = _tensor_to_median(tensor)
            return norm
        if not torch.is_tensor(preds):
            try:
                tensor = torch.as_tensor(preds)
            except Exception:
                return {param: float('nan') for param in param_order}
        else:
            tensor = preds
        tensor = tensor.detach().float()
        if tensor.ndim >= 2:
            if len(quantiles) > 0:
                num_params = len(param_order)
                expected = num_params * len(quantiles)
                flat = tensor.view(tensor.size(0), -1)
                if flat.size(1) == expected:
                    tensor = flat.view(tensor.size(0), num_params, len(quantiles))
            for idx, param in enumerate(param_order):
                if idx < tensor.size(1):
                    norm[param] = _tensor_to_median(tensor[0, idx])
                else:
                    norm[param] = float('nan')
            return norm
        if tensor.ndim == 1:
            chunks = torch.chunk(tensor, len(param_order))
            for idx, param in enumerate(param_order):
                part = chunks[idx] if idx < len(chunks) else None
                norm[param] = _tensor_to_median(part) if part is not None else float('nan')
            return norm
        return {param: float('nan') for param in param_order}

    def _extract_teff(sample_dict: Dict[str, Any]) -> Optional[float]:
        return extract_physical_param(sample_dict, 'Teff')

    def _extract_logg(sample_dict: Dict[str, Any]) -> Optional[float]:
        return extract_physical_param(sample_dict, 'logg')

    def _select_feature_tensor(sample: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Optional[str]]:
        """Find an interpolatable feature tensor and remember which key it came from."""
        for key in feature_candidate_keys:
            tensor = get_tensor_from_sample(sample, key)
            if tensor is None:
                continue
            tensor = tensor.view(-1).float()
            if target_dim is not None and tensor.numel() != target_dim:
                continue
            return tensor, key
        return None, None

    try:
        for idx_a, idx_b in pairs:
            try:
                sample_a = deepcopy(dataset[idx_a])
                sample_b = deepcopy(dataset[idx_b])
            except Exception as exc:
                print(f"Warning: Failed to load samples {idx_a}:{idx_b} for interpolation ({exc})")
                continue

            teff_a = _extract_teff(sample_a)
            teff_b = _extract_teff(sample_b)
            if min_teff_diff > 0.0 and teff_a is not None and teff_b is not None:
                diff = abs(teff_a - teff_b)
                if diff < min_teff_diff:
                    print(f"Warning: Teff difference {diff:.1f} K for samples {idx_a}:{idx_b} below threshold {min_teff_diff:.1f} K; skipping pair.")
                    continue
            logg_a = _extract_logg(sample_a)
            logg_b = _extract_logg(sample_b)
            if min_logg_diff > 0.0 and logg_a is not None and logg_b is not None:
                diff_logg = abs(logg_a - logg_b)
                if diff_logg < min_logg_diff:
                    print(f"Warning: logg difference {diff_logg:.2f} for samples {idx_a}:{idx_b} below threshold {min_logg_diff:.2f}; skipping pair.")
                    continue

            pair_record = {
                'pair': (idx_a, idx_b),
                'alpha_points': [],
                'teff_values': [teff_a, teff_b],
                'logg_values': [logg_a, logg_b],
            }
            tensor_a, key_a = _select_feature_tensor(sample_a)
            tensor_b, key_b = _select_feature_tensor(sample_b)
            if tensor_a is None or tensor_b is None:
                print(f"Warning: Samples {idx_a}:{idx_b} missing usable spectral features; skipping pair.")
                continue

            if tensor_a.numel() != tensor_b.numel():
                print(f"Warning: Feature size mismatch for samples {idx_a}:{idx_b}; skipping pair.")
                continue

            tensor_a = tensor_a.clone().view(-1)
            tensor_b = tensor_b.clone().view(-1)

            for alpha in alphas:
                alpha = float(max(0.0, min(1.0, alpha)))
                is_alpha_zero = math.isclose(alpha, 0.0, abs_tol=1e-6)
                is_alpha_one = math.isclose(alpha, 1.0, abs_tol=1e-6)

                if is_alpha_one:
                    synthetic_sample = deepcopy(sample_b)
                else:
                    synthetic_sample = deepcopy(sample_a)

                if normalize_mode_label(synthetic_sample.get('mode')) != 'single_star':
                    synthetic_sample['mode'] = 'single'

                interpolated_core = ((1.0 - alpha) * tensor_a + alpha * tensor_b).to(dtype=torch.float32)

                target_feature_keys: Optional[Set[str]] = None
                if not (is_alpha_zero or is_alpha_one):
                    target_feature_keys = set(k for k in feature_candidate_keys if synthetic_sample.get(k) is not None)
                    if key_a is not None:
                        target_feature_keys.add(key_a)
                    if key_b is not None:
                        target_feature_keys.add(key_b)
                    target_feature_keys.add('masked_spectra')

                    synthetic_sample['masked_spectra'] = interpolated_core.clone()
                    for feat_key in target_feature_keys:
                        if feat_key == 'masked_spectra':
                            continue  # already set
                        synthetic_sample[feat_key] = interpolated_core.clone()

                # Ensure metadata carries the interpolated tensors so collate functions pick them up
                meta = synthetic_sample.get('metadata')
                if isinstance(meta, dict):
                    if target_feature_keys is None:
                        target_feature_keys = set()
                    masked_value = synthetic_sample.get('masked_spectra')
                    if masked_value is not None:
                        meta['masked_spectra'] = masked_value
                    for feat_key in target_feature_keys:
                        if feat_key == 'masked_spectra':
                            continue
                        meta[feat_key] = synthetic_sample.get(feat_key)
                    raw_meta = meta.get('raw')
                    if isinstance(raw_meta, dict):
                        if masked_value is not None:
                            raw_meta['masked_spectra'] = masked_value
                        for feat_key in target_feature_keys:
                            if feat_key == 'masked_spectra':
                                continue
                            raw_meta[feat_key] = synthetic_sample.get(feat_key)

                batch = collate([synthetic_sample])
                batch_a = collate([sample_a])
                batch_b = collate([sample_b])
                batch_device: Dict[str, Any] = {}
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        batch_device[key] = value.to(device)
                    else:
                        batch_device[key] = value
                with torch.no_grad():
                    outputs = trainer.get_logits(batch_device, device, val=True)
                    outputs_a = trainer.get_logits(batch_a, device, val=True)
                    outputs_b = trainer.get_logits(batch_b, device, val=True)

                    generated_text, input_text, target_text, _ = trainer.model.generate_response_from_batch(
                                    batch_data=batch_device,
                                    batch_idx=0,
                                    tokenizer=tokenizer,
                                    max_new_tokens=50,
                                    temperature=0.0,
                                    top_p=1.0
                                )

                stellar_preds_raw = None
                if isinstance(outputs, dict) and 'stellar_predictions' in outputs:
                    stellar_preds_raw = outputs['stellar_predictions']
                elif hasattr(outputs, 'stellar_predictions'):
                    stellar_preds_raw = outputs.stellar_predictions

                # Use text predictions if numerical predictions are not available
                if stellar_preds_raw is None:
                    print(f"Warning: Missing stellar predictions for interpolation sample; using text predictions for alpha={alpha}")
                    # Extract from generated text
                    extracted = extract_stellar_params_from_text(generated_text)
                    if extracted:
                        teff_pred = float(extracted.get('Teff', float('nan')))
                        logg_pred = float(extracted.get('logg', float('nan')))
                        feh_pred = float(extracted.get('FeH', float('nan')))
                        # Set normalized versions to NaN since we don't have them
                        teff_norm = logg_norm = feh_norm = float('nan')
                    else:
                        print(f"Warning: Could not extract parameters from text for alpha={alpha}")
                        teff_pred = logg_pred = feh_pred = float('nan')
                        teff_norm = logg_norm = feh_norm = float('nan')
                else:
                    # Use numerical predictions
                    preds_tensor = torch.as_tensor(stellar_preds_raw, dtype=torch.float32)
                    if preds_tensor.ndim == 1:
                        preds_tensor = preds_tensor.view(1, -1)

                    model_for_params = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
                    num_params = len(getattr(model_for_params, 'stellar_params', ['Teff', 'logg', 'FeH']))
                    num_quantiles = max(len(quantiles), 1)

                    total_dim = preds_tensor.size(-1)
                    if preds_tensor.ndim == 2 and total_dim not in (num_params, num_params * num_quantiles):
                        if num_params > 0 and total_dim % num_params == 0:
                            num_quantiles = total_dim // num_params
                        else:
                            raise ValueError(f"Unexpected stellar prediction shape {preds_tensor.shape}")

                    if preds_tensor.ndim == 2:
                        if num_quantiles == 1 and total_dim == num_params:
                            preds_tensor = preds_tensor.view(preds_tensor.size(0), num_params, 1)
                        else:
                            preds_tensor = preds_tensor.view(preds_tensor.size(0), num_params, num_quantiles)

                    median_idx = min(num_quantiles // 2, num_quantiles - 1)
                    teff_norm = preds_tensor[0, 0, median_idx].item() if preds_tensor.size(1) > 0 else float('nan')
                    logg_norm = preds_tensor[0, 1, median_idx].item() if preds_tensor.size(1) > 1 else float('nan')
                    feh_norm = preds_tensor[0, 2, median_idx].item() if preds_tensor.size(1) > 2 else float('nan')

                    teff_pred = float(_convert_norm_to_physical('Teff', teff_norm))
                    logg_pred = float(_convert_norm_to_physical('logg', logg_norm))
                    feh_pred = float(_convert_norm_to_physical('FeH', feh_norm))

                y_numeric = batch_device.get('y_numeric')
                y_numeric_present = batch_device.get('y_numeric_present')
                mask_value = False
                if y_numeric_present is not None:
                    mask_entry = y_numeric_present[0]
                    if torch.is_tensor(mask_entry):
                        mask_value = bool(mask_entry.item())
                    else:
                        mask_value = bool(mask_entry)
                if y_numeric is not None and mask_value:
                    teff_true = float(_convert_norm_to_physical('Teff', float(y_numeric[0][0].item())))
                    logg_true = float(_convert_norm_to_physical('logg', float(y_numeric[0][1].item())))
                    feh_true = float(_convert_norm_to_physical('FeH', float(y_numeric[0][2].item())))
                else:
                    teff_true = logg_true = feh_true = float('nan')

                text_physical = {param: float('nan') for param in ['Teff', 'logg', 'FeH']}
                extracted = extract_stellar_params_from_text(generated_text)
                if extracted:
                    for param in ['Teff', 'logg', 'FeH']:
                        value = extracted.get(param)
                        if value is None:
                            numeric_value = float('nan')
                        else:
                            try:
                                numeric_value = float(value)
                            except (TypeError, ValueError):
                                numeric_value = float('nan')
                        text_physical[param] = numeric_value

                pair_record['alpha_points'].append({
                    'alpha': alpha,
                    'teff': teff_pred,
                    'logg': logg_pred,
                    'feh': feh_pred,
                    'teff_true': teff_true,
                    'logg_true': logg_true,
                    'feh_true': feh_true,
                    'teff_norm': teff_norm,
                    'logg_norm': logg_norm,
                    'feh_norm': feh_norm,
                    'text_teff': text_physical.get('Teff'),
                    'text_logg': text_physical.get('logg'),
                    'text_feh': text_physical.get('FeH'),
                })

            results.append(pair_record)
    finally:
        if model is not None and model_was_training:
            model.train()

    output_json = os.path.join(output_dir, 'interpolation_results.json')

    def _sanitize(value):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value

    serializable_results: List[Dict[str, Any]] = []
    for record in results:
        cleaned_record: Dict[str, Any] = {
            'pair': record.get('pair'),
            'teff_values': [_sanitize(v) for v in record.get('teff_values', [])],
            'logg_values': [_sanitize(v) for v in record.get('logg_values', [])],
            'alpha_points': []
        }
        for point in record.get('alpha_points', []):
            cleaned_point = {key: _sanitize(val) for key, val in point.items()}
            cleaned_record['alpha_points'].append(cleaned_point)
        serializable_results.append(cleaned_record)

    with open(output_json, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"[OK] Interpolation results saved to {output_json}")

    return results


def run_baseline_mlp_interpolation(checkpoint_path: str,
                                   dataset,
                                   pairs: List[Tuple[int, int]],
                                   alphas: List[float],
                                   device: torch.device) -> List[Dict[str, Any]]:
    """Generate interpolation trajectories using the baseline MLP regressor."""
    if not os.path.isfile(checkpoint_path):
        print(f"Warning: Baseline MLP checkpoint not found at {checkpoint_path}; skipping baseline comparison.")
        return []

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as exc:
        print(f"Warning: Failed to load baseline MLP checkpoint ({exc}); skipping baseline comparison.")
        return []

    config = checkpoint.get('config', {})
    targets: List[str] = list(config.get('targets', ['Teff', 'logg', 'FeH']))
    output_dim = len(targets)
    model_state = checkpoint.get('model_state')
    if not isinstance(model_state, dict):
        print("Warning: Baseline MLP checkpoint missing model parameters; skipping baseline comparison.")
        return []

    feature_mean = checkpoint.get('feature_mean')
    feature_std = checkpoint.get('feature_std')
    if isinstance(feature_mean, torch.Tensor):
        feature_dim = feature_mean.numel()
    else:
        first_weight_key = next((key for key in model_state if key.endswith('weight')), None)
        if first_weight_key is None:
            print("Warning: Unable to infer baseline MLP input dimension; skipping baseline comparison.")
            return []
        feature_dim = model_state[first_weight_key].shape[1]

    hidden_dim = int(config.get('hidden_dim', 512))
    hidden_layers = int(config.get('hidden_layers', 3))
    dropout = float(config.get('dropout', 0.0))

    if isinstance(device, int):
        baseline_device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        baseline_device = torch.device(device)
    elif isinstance(device, torch.device):
        baseline_device = device
    else:
        baseline_device = torch.device('cpu')

    model = MLPRegressor(
        input_dim=feature_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        dropout=dropout,
    ).to(baseline_device)
    model.load_state_dict(model_state)
    model.eval()

    print("baseline model loaded succesfully!")

    if isinstance(feature_mean, torch.Tensor):
        feature_mean = feature_mean.to(baseline_device)
    else:
        feature_mean = None
    if isinstance(feature_std, torch.Tensor):
        feature_std = feature_std.to(baseline_device)
    else:
        feature_std = None

    target_denormalizers = build_target_denormalizers(targets, config.get('target_denorm_config'))
    candidate_feature_keys = ['masked_spectra', 'features', 'spectra', 'x_raw']

    alpha_values = sorted({float(max(0.0, min(1.0, alpha))) for alpha in alphas})
    if not alpha_values:
        return []
    if 0.0 not in alpha_values:
        alpha_values.insert(0, 0.0)
    if 1.0 not in alpha_values:
        alpha_values.append(1.0)

    results: List[Dict[str, Any]] = []

    def _select_feature_vector(sample: Dict[str, Any]) -> Optional[torch.Tensor]:
        for key in candidate_feature_keys:
            tensor = get_tensor_from_sample(sample, key)
            if tensor is None:
                continue
            vector = tensor.view(-1).float()
            if vector.numel() != feature_dim:
                continue
            return vector.clone()
        return None

    for idx_a, idx_b in pairs:
        try:
            sample_a = deepcopy(dataset[idx_a])
            sample_b = deepcopy(dataset[idx_b])
        except Exception as exc:
            print(f"Warning: Failed to load samples {idx_a}:{idx_b} for baseline interpolation ({exc})")
            continue

        features_a = _select_feature_vector(sample_a)
        features_b = _select_feature_vector(sample_b)
        if features_a is None or features_b is None:
            print(f"Warning: Samples {idx_a}:{idx_b} missing baseline features; skipping pair for baseline interpolation.")
            continue

        features_a = features_a.to(baseline_device)
        features_b = features_b.to(baseline_device)

        teff_a = extract_physical_param(sample_a, 'Teff')
        teff_b = extract_physical_param(sample_b, 'Teff')
        logg_a = extract_physical_param(sample_a, 'logg')
        logg_b = extract_physical_param(sample_b, 'logg')

        pair_record: Dict[str, Any] = {
            'pair': (idx_a, idx_b),
            'alpha_points': [],
            'teff_values': [teff_a, teff_b],
            'logg_values': [logg_a, logg_b],
        }

        with torch.no_grad():
            for alpha in alpha_values:
                alpha = float(alpha)
                blended = (1.0 - alpha) * features_a + alpha * features_b
                # blended_norm = normalize_batch(blended, feature_mean, feature_std)

                preds_tensor = model(blended.unsqueeze(0)).squeeze(0).cpu()
                preds_np = preds_tensor.numpy()

                predicted: Dict[str, float] = {}
                for idx, target in enumerate(targets):
                    value = float(preds_np[idx]) if idx < len(preds_np) else float('nan')
                    denorm_spec = target_denormalizers.get(target) if target_denormalizers else None
                    if denorm_spec is not None:
                        value = float(denorm_spec.denormalize(np.asarray([value], dtype=np.float32))[0])
                    predicted[target] = value
                    print(f"baseline mlp predicted {target}: ", value)

                pair_record['alpha_points'].append({
                    'alpha': alpha,
                    'teff': predicted.get('Teff', float('nan')),
                    'logg': predicted.get('logg', float('nan')),
                    'feh': predicted.get('FeH', float('nan')),
                    'text_teff': None,
                    'text_logg': None,
                    'text_feh': None,
                })

        results.append(pair_record)

    return results


def select_random_interpolation_pairs(dataset,
                                      trainer: LLMTrainer,
                                      num_pairs: int,
                                      seed: int = 42,
                                      min_teff_diff: float = 0.0,
                                      min_logg_diff: float = 0.0) -> List[Tuple[int, int]]:
    """Randomly choose interpolation pairs that have valid single-star features."""
    if num_pairs <= 0:
        return []
    target_dim = get_single_star_latent_dim(trainer)
    total_samples = len(dataset)
    indices = list(range(total_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    valid: List[Tuple[int, Optional[float], Optional[float]]] = []
    for idx in indices:
        try:
            sample = dataset[idx]
        except Exception as exc:
            print(f"Warning: Failed to load sample {idx} while searching for interpolation pairs ({exc})")
            continue
        if find_interpolatable_tensor(sample, target_dim) is None:
            continue
        teff = extract_physical_param(sample, 'Teff')
        logg = extract_physical_param(sample, 'logg')
        valid.append((idx, teff, logg))
        if len(valid) >= num_pairs * 4:
            break

    if len(valid) < 2:
        print("Warning: Not enough valid samples found for interpolation.")
        return []

    pairs: List[Tuple[int, int]] = []
    used_indices = set()
    for i, (idx_a, teff_a, logg_a) in enumerate(valid):
        if len(pairs) >= num_pairs:
            break
        if i in used_indices:
            continue
        best_match = None
        for j in range(i + 1, len(valid)):
            if j in used_indices:
                continue
            idx_b, teff_b, logg_b = valid[j]
            if min_teff_diff > 0.0 and teff_a is not None and teff_b is not None:
                if abs(teff_a - teff_b) < min_teff_diff:
                    continue
            if min_logg_diff > 0.0 and logg_a is not None and logg_b is not None:
                if abs(logg_a - logg_b) < min_logg_diff:
                    continue
            best_match = j
            break
        if best_match is not None:
            used_indices.add(i)
            used_indices.add(best_match)
            idx_b_val = valid[best_match][0]
            pairs.append((idx_a, idx_b_val))

    if len(pairs) < num_pairs:
        print(f"Warning: Only found {len(pairs)} interpolation pairs out of requested {num_pairs}.")

    return pairs


def plot_interpolation_kiel(pair_results: List[Dict[str, Any]],
                            output_dir: str,
                            background_points: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:
    """Plot Kiel diagrams (Teff-logg) for interpolation trajectories."""
    if not pair_results:
        print("No interpolation results to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)
    anchor_labels_drawn = {'start': False, 'end': False}

    for record in pair_results:
        points = sorted(record['alpha_points'], key=lambda x: x['alpha'])
        teff = np.array([p['teff'] for p in points], dtype=float)
        logg = np.array([p['logg'] for p in points], dtype=float)
        text_teff = np.array([p.get('text_teff', np.nan) for p in points], dtype=float)
        text_logg = np.array([p.get('text_logg', np.nan) for p in points], dtype=float)
        alphas = np.array([p['alpha'] for p in points], dtype=float)

        valid_mask = ~(np.isnan(teff) | np.isnan(logg))
        if not valid_mask.any():
            print(f"Warning: Interpolation pair {record['pair']} produced no valid numerical predictions; skipping plot.")
            continue

        teff_valid = teff[valid_mask]
        logg_valid = logg[valid_mask]
        alphas_valid = alphas[valid_mask]

        text_mask = ~(np.isnan(text_teff) | np.isnan(text_logg))
        text_teff_valid = text_teff[text_mask]
        text_logg_valid = text_logg[text_mask]

        fig, ax = plt.subplots(figsize=(8, 6))
        if background_points is not None and len(background_points[0]) > 0:
            bg_teff, bg_logg = background_points
            ax.scatter(bg_teff, bg_logg, color='lightgray', alpha=0.15, s=12, edgecolors='none', label='Dataset')
        ax.set_xlim(8000, 4000)
        ax.set_ylim(5.5, 2)
        scatter = ax.scatter(teff_valid, logg_valid, c=alphas_valid, cmap='viridis', s=60, edgecolors='k')
        line_color = 'tab:blue'
        ax.plot(teff_valid, logg_valid, linestyle='-', color=line_color, linewidth=2.0, alpha=0.9, label='Model (numerical)')

        # if text_teff_valid.size > 0:
            # ax.plot(text_teff_valid, text_logg_valid, linestyle='--', color=line_color, linewidth=2.0, alpha=0.9, label='Text-derived')
            # ax.scatter(text_teff_valid, text_logg_valid, color=line_color, s=40, alpha=0.75)

        anchor_teff = record.get('teff_values') or []
        anchor_logg = record.get('logg_values') or []
        if len(anchor_teff) >= 2 and len(anchor_logg) >= 2:
            if anchor_teff[0] is not None and anchor_logg[0] is not None:
                ax.scatter([anchor_teff[0]], [anchor_logg[0]], color='black', marker='*', s=90, alpha=0.75,
                           label='Sample A (data)' if not anchor_labels_drawn['start'] else None)
                anchor_labels_drawn['start'] = True
            if anchor_teff[1] is not None and anchor_logg[1] is not None:
                ax.scatter([anchor_teff[1]], [anchor_logg[1]], color='dimgray', marker='P', s=85, alpha=0.75,
                           label='Sample B (data)' if not anchor_labels_drawn['end'] else None)
                anchor_labels_drawn['end'] = True

        ax.scatter([teff_valid[0]], [logg_valid[0]], color='red', s=80, marker='^', label='alpha=0')
        ax.scatter([teff_valid[-1]], [logg_valid[-1]], color='blue', s=80, marker='s', label='alpha=1')

        ax.set_xlabel('Effective Temperature Teff (K)')
        ax.set_ylabel('Surface Gravity log g')
        ax.set_title(f"Latent Interpolation Pair {record['pair'][0]} -> {record['pair'][1]}")
        ax.invert_yaxis()
        ax.invert_xaxis()

        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Interpolation alpha')

        filename = os.path.join(output_dir, f'interpolation_pair_{record['pair'][0]}_{record['pair'][1]}.png')
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] Interpolation plot saved to {filename}")



def plot_interpolation_kiel_combined(pair_results: List[Dict[str, Any]],
                                     output_dir: str,
                                     background_points: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:
    """Plot numeric and text interpolation trajectories on separate Kiel diagrams."""
    if not pair_results:
        print("No interpolation results to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    fig_num, ax_num = plt.subplots(figsize=(8, 6))
    fig_text, ax_text = plt.subplots(figsize=(8, 6))

    ax_num.set_xlim(4000, 8000)
    ax_num.set_ylim(2, 5.5)
    ax_text.set_xlim(4000, 8000)
    ax_text.set_ylim(2, 5.5)

    if background_points is not None and len(background_points[0]) > 0:
        bg_teff, bg_logg = background_points
        ax_num.scatter(bg_teff, bg_logg, color='lightgray', alpha=0.3, s=12, edgecolors='none', label='Dataset')
        ax_text.scatter(bg_teff, bg_logg, color='lightgray', alpha=0.3, s=12, edgecolors='none')

    cmap = plt.get_cmap('tab20')
    pair_colors: Dict[Tuple[int, int], Any] = {}
    labels_used_num: set[str] = set()
    labels_used_text: set[str] = set()
    labels_missing_text: set[str] = set()
    anchor_labels_num = {'start': False, 'end': False}
    anchor_labels_text = {'start': False, 'end': False}

    for idx, record in enumerate(pair_results):
        points = sorted(record['alpha_points'], key=lambda x: x['alpha'])
        teff = np.array([p['teff'] for p in points], dtype=float)
        logg = np.array([p['logg'] for p in points], dtype=float)
        text_teff = np.array([p.get('text_teff', np.nan) for p in points], dtype=float)
        text_logg = np.array([p.get('text_logg', np.nan) for p in points], dtype=float)

        valid_mask = ~(np.isnan(teff) | np.isnan(logg))
        if not valid_mask.any():
            print(f"Warning: Interpolation pair {record['pair']} produced no valid numerical predictions; skipping in combined plot.")
            continue

        teff_valid = teff[valid_mask]
        logg_valid = logg[valid_mask]
        pair_key = tuple(record['pair'])
        if pair_key not in pair_colors:
            pair_colors[pair_key] = cmap(len(pair_colors) % cmap.N)
        color = pair_colors[pair_key]
        label = f"{record['pair'][0]}->{record['pair'][1]}"

        line_label = label if label not in labels_used_num else None
        if line_label is not None:
            labels_used_num.add(line_label)
        ax_num.plot(teff_valid, logg_valid, color=color, linewidth=2.0, alpha=0.85, label=line_label)
        ax_num.scatter(teff_valid, logg_valid, color=color, s=35, alpha=0.85)
        ax_num.scatter([teff_valid[0]], [logg_valid[0]], color=color, marker='^', s=70)
        ax_num.scatter([teff_valid[-1]], [logg_valid[-1]], color=color, marker='s', s=70)

        anchor_teff = record.get('teff_values') or []
        anchor_logg = record.get('logg_values') or []
        if len(anchor_teff) >= 2 and len(anchor_logg) >= 2:
            if anchor_teff[0] is not None and anchor_logg[0] is not None:
                ax_num.scatter([anchor_teff[0]], [anchor_logg[0]], color='black', marker='*', s=80, alpha=0.7,
                               label='Sample A (data)' if not anchor_labels_num['start'] else None)
                anchor_labels_num['start'] = True
            if anchor_teff[1] is not None and anchor_logg[1] is not None:
                ax_num.scatter([anchor_teff[1]], [anchor_logg[1]], color='dimgray', marker='P', s=75, alpha=0.7,
                               label='Sample B (data)' if not anchor_labels_num['end'] else None)
                anchor_labels_num['end'] = True

        text_mask = ~(np.isnan(text_teff) | np.isnan(text_logg))
        if text_mask.any():
            text_teff_valid = text_teff[text_mask]
            text_logg_valid = text_logg[text_mask]
            dashed_label = f"{label} (text)" if f"{label} (text)" not in labels_used_text else None
            if dashed_label is not None:
                labels_used_text.add(dashed_label)
            ax_text.plot(text_teff_valid, text_logg_valid, color=color, linestyle='--', linewidth=1.8, alpha=0.8, label=dashed_label)
            ax_text.scatter(text_teff_valid, text_logg_valid, color=color, s=30, alpha=0.75)
            ax_text.scatter([text_teff_valid[0]], [text_logg_valid[0]], color=color, marker='^', s=60)
            ax_text.scatter([text_teff_valid[-1]], [text_logg_valid[-1]], color=color, marker='s', s=60)
        else:
            missing_label = f"{label} (no text)" if f"{label} (no text)" not in labels_missing_text else None
            if missing_label is not None:
                labels_missing_text.add(missing_label)
                ax_text.scatter([teff_valid[0]], [logg_valid[0]], color=color, marker='x', s=55, alpha=0.7, label=missing_label)
            else:
                ax_text.scatter([teff_valid[0]], [logg_valid[0]], color=color, marker='x', s=55, alpha=0.7)
            ax_text.scatter([teff_valid[-1]], [logg_valid[-1]], color=color, marker='x', s=55, alpha=0.7)

        if len(anchor_teff) >= 2 and len(anchor_logg) >= 2:
            if anchor_teff[0] is not None and anchor_logg[0] is not None:
                ax_text.scatter([anchor_teff[0]], [anchor_logg[0]], color='black', marker='*', s=70, alpha=0.65,
                                label='Sample A (data)' if not anchor_labels_text['start'] else None)
                anchor_labels_text['start'] = True
            if anchor_teff[1] is not None and anchor_logg[1] is not None:
                ax_text.scatter([anchor_teff[1]], [anchor_logg[1]], color='dimgray', marker='P', s=65, alpha=0.65,
                                label='Sample B (data)' if not anchor_labels_text['end'] else None)
                anchor_labels_text['end'] = True

    ax_num.set_xlabel('Effective Temperature Teff (K)')
    ax_num.set_ylabel('Surface Gravity log g')
    ax_num.set_title('Combined Numeric Interpolation Trajectories')
    ax_num.invert_yaxis()
    ax_num.invert_xaxis()
    ax_num.grid(True, alpha=0.3)
    if labels_used_num:
        ax_num.legend(loc='best', fontsize=8, ncol=2)

    ax_text.set_xlabel('Effective Temperature Teff (K)')
    ax_text.set_ylabel('Surface Gravity log g')
    ax_text.set_title('Combined Text-Derived Interpolation Trajectories')
    ax_text.invert_yaxis()
    ax_text.invert_xaxis()
    ax_text.grid(True, alpha=0.3)
    if labels_used_text:
        ax_text.legend(loc='best', fontsize=8, ncol=2)

    numeric_path = os.path.join(output_dir, 'interpolation_pairs_numeric.png')
    text_path = os.path.join(output_dir, 'interpolation_pairs_text.png')
    fig_num.tight_layout()
    fig_text.tight_layout()
    fig_num.savefig(numeric_path, dpi=150, bbox_inches='tight')
    fig_text.savefig(text_path, dpi=150, bbox_inches='tight')
    plt.close(fig_num)
    plt.close(fig_text)
    print(f"[OK] Numeric interpolation plot saved to {numeric_path}")
    print(f"[OK] Text interpolation plot saved to {text_path}")


def plot_interpolation_kiel_multi_model(model_results: "OrderedDict[str, List[Dict[str, Any]]]",
                                        output_dir: str,
                                        background_points: Optional[Tuple[np.ndarray, np.ndarray]],
                                        interpolation_pairs: List[Tuple[int, int]]) -> None:
    """Plot interpolation trajectories from multiple models on shared Kiel diagrams (combined view)."""
    if not model_results:
        print("Warning: No model results available for multi-model interpolation plot.")
        return

    if not interpolation_pairs:
        print("Warning: No interpolation pairs provided for multi-model interpolation plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Pre-compute lookup tables for faster access
    pair_lookup: Dict[str, Dict[Tuple[int, int], Dict[str, Any]]] = {}
    for label, records in model_results.items():
        mapping: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for record in records or []:
            pair = record.get('pair')
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                mapping[(int(pair[0]), int(pair[1]))] = record
        pair_lookup[label] = mapping

    if not pair_lookup:
        print("Warning: Unable to assemble pair lookup for interpolation comparison plot.")
        return

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', list(plt.cm.tab10.colors))
    marker_cycle = ['o', 's', '^', 'D', 'P', 'X', '*', 'v']
    labels_order = list(pair_lookup.keys())
    style_by_label = {
        label: (
            color_cycle[idx % len(color_cycle)],
            marker_cycle[idx % len(marker_cycle)],
        )
        for idx, label in enumerate(labels_order)
    }

    num_pairs = len(interpolation_pairs)
    ncols = min(3, max(1, num_pairs))
    nrows = math.ceil(num_pairs / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5.2 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    legend_handles: OrderedDict[str, Any] = OrderedDict()
    any_model_plotted = False

    for pair_idx, pair in enumerate(interpolation_pairs):
        ax = axes_flat[pair_idx]
        pair_tuple = (int(pair[0]), int(pair[1]))

        if background_points is not None and len(background_points[0]) > 0:
            bg_teff, bg_logg = background_points
            dataset_scatter = ax.scatter(
                bg_teff,
                bg_logg,
                color='lightgray',
                alpha=0.9,
                s=14,
                edgecolors='none',
                label=None,
            )
            if 'Dataset' not in legend_handles:
                legend_handles['Dataset'] = dataset_scatter

        anchor_label_flags = {'start': False, 'end': False}
        model_drawn_for_axis: set[str] = set()
        plotted_this_pair = False

        for label, mapping in pair_lookup.items():
            record = mapping.get(pair_tuple)
            if not record:
                continue

            points = sorted(record.get('alpha_points', []), key=lambda x: x.get('alpha', 0.0))
            teff = np.array([p.get('teff', np.nan) for p in points], dtype=float)
            logg = np.array([p.get('logg', np.nan) for p in points], dtype=float)
            valid_mask = ~(np.isnan(teff) | np.isnan(logg))
            if not valid_mask.any():
                continue

            teff_valid = teff[valid_mask]
            logg_valid = logg[valid_mask]
            color, marker = style_by_label[label]

            line_label = None if label in model_drawn_for_axis else label
            line = ax.plot(
                teff_valid,
                logg_valid,
                color=color,
                linewidth=2.0,
                alpha=0.5,
                label=line_label,
            )
            ax.scatter(
                teff_valid,
                logg_valid,
                color=color,
                s=45,
                marker=marker,
                edgecolors='k',
                linewidths=0.4,
                alpha=0.5,
            )
            ax.scatter(
                [teff_valid[0]],
                [logg_valid[0]],
                color=color,
                marker='^',
                s=80,
                edgecolors='k',
                linewidths=0.6,
            )
            ax.scatter(
                [teff_valid[-1]],
                [logg_valid[-1]],
                color=color,
                marker='s',
                s=80,
                edgecolors='k',
                linewidths=0.6,
            )

            if label not in legend_handles and line:
                legend_handles[label] = line[0]

            model_drawn_for_axis.add(label)
            plotted_this_pair = True

            teff_values = record.get('teff_values') or []
            logg_values = record.get('logg_values') or []
            if len(teff_values) >= 2 and len(logg_values) >= 2:
                if teff_values[0] is not None and logg_values[0] is not None:
                    anchor_label = None
                    if not anchor_label_flags['start']:
                        anchor_label = 'Sample A (data)'
                    star_scatter = ax.scatter(
                        [teff_values[0]],
                        [logg_values[0]],
                        color='black',
                        marker='*',
                        s=110,
                        alpha=0.85,
                        label=anchor_label,
                    )
                    if anchor_label and anchor_label not in legend_handles:
                        legend_handles[anchor_label] = star_scatter
                    anchor_label_flags['start'] = True
                if teff_values[1] is not None and logg_values[1] is not None:
                    anchor_label = None
                    if not anchor_label_flags['end']:
                        anchor_label = 'Sample B (data)'
                    end_scatter = ax.scatter(
                        [teff_values[1]],
                        [logg_values[1]],
                        color='dimgray',
                        marker='P',
                        s=105,
                        alpha=0.85,
                        label=anchor_label,
                    )
                    if anchor_label and anchor_label not in legend_handles:
                        legend_handles[anchor_label] = end_scatter
                    anchor_label_flags['end'] = True

        if not plotted_this_pair:
            ax.set_visible(False)
            continue

        any_model_plotted = True

        ax.set_xlim(8000, 4000)
        ax.set_ylim(5.5, 2.0)
        ax.set_xlabel('Effective Temperature Teff (K)')
        ax.set_ylabel('Surface Gravity log g')
        ax.set_title(f"Pair {pair_tuple[0]} → {pair_tuple[1]}")
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        ax.invert_yaxis()

    # Hide any unused axes in the grid
    for extra_ax in axes_flat[len(interpolation_pairs):]:
        extra_ax.set_visible(False)

    if not any_model_plotted:
        plt.close(fig)
        print("Warning: No valid interpolation data available for multi-model comparison plot.")
        return

    if legend_handles:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc='upper center',
            ncol=min(4, len(legend_handles)),
            frameon=True,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    filename = os.path.join(output_dir, 'comparison_interpolation_combined.png')
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Combined comparison plot saved to {filename}")


def plot_interpolation_metrics_summary(metrics_by_model: OrderedDict[str, Dict[str, Any]],
                                       output_dir: str,
                                       kind_label: str = 'numeric') -> None:
    """Create bar plots summarizing averaged interpolation metrics for each model."""
    if not metrics_by_model:
        print("Warning: No metrics available for interpolation summary plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    metrics_to_plot = [
        ('mean_log_density', 'Mean Log Density'),
        ('mean_density', 'Mean Density'),
        ('mean_nn_distance', 'Mean NN Distance'),
        ('mean_smoothness', 'Mean Smoothness'),
    ]

    plotted_any = False
    labels = list(metrics_by_model.keys())
    kind_display = 'Numeric' if kind_label == 'numeric' else 'Text'
    bar_colors = {
        'numeric': 'tab:blue',
        'text': 'tab:purple',
    }
    bar_color = bar_colors.get(kind_label, 'tab:blue')

    for metric_key, metric_label in metrics_to_plot:
        values: List[float] = []
        valid_labels: List[str] = []
        for label in labels:
            summary = metrics_by_model[label].get('summary') if metrics_by_model[label] else None
            value = summary.get(metric_key) if summary else None
            if isinstance(value, (int, float)):
                values.append(float(value))
                valid_labels.append(label)

        if not values:
            continue

        plotted_any = True
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(valid_labels)), 5))
        x_positions = np.arange(len(valid_labels))
        bars = ax.bar(x_positions, values, color=bar_color, alpha=0.85)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(valid_labels, rotation=25, ha='right')
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} ({kind_display} - averaged across pairs)")
        ax.grid(axis='y', linestyle='--', alpha=0.2)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha='center',
                va='bottom',
                fontsize=8,
            )

        fig.tight_layout()
        metric_filename = os.path.join(output_dir, f'metric_{kind_label}_{metric_key}.png')
        fig.savefig(metric_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] Metric summary plot saved to {metric_filename}")

    if not plotted_any:
        print(f"Warning: No {kind_display.lower()} metric summaries available to plot.")


def plot_interpolation_pairwise_vs_baseline(model_results: "OrderedDict[str, List[Dict[str, Any]]]",
                                            baseline_label: str,
                                            output_dir: str,
                                            kind: str,
                                            background_points: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                            baseline_kind: Optional[str] = None,
                                            metrics_by_model: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    """Plot interpolation trajectories comparing each model against a baseline for numeric or text predictions."""
    if not model_results or baseline_label not in model_results:
        print(f"Warning: Baseline '{baseline_label}' not available for {kind} comparison plots.")
        return

    baseline_records = model_results.get(baseline_label) or []
    if not baseline_records:
        print(f"Warning: No baseline records available for {kind} comparison plots.")
        return

    if kind not in {'numeric', 'text'}:
        raise ValueError(f"Unsupported interpolation comparison kind '{kind}'")

    baseline_kind = baseline_kind or kind
    if baseline_kind not in {'numeric', 'text'}:
        raise ValueError(f"Unsupported baseline data kind '{baseline_kind}'")

    def _resolve_keys(data_kind: str) -> Tuple[str, str, str, str]:
        if data_kind == 'numeric':
            return 'teff', 'logg', 'Numerical', 'numeric'
        return 'text_teff', 'text_logg', 'Text-derived', 'text'

    teff_key, logg_key, kind_title, plot_suffix = _resolve_keys(kind)
    baseline_teff_key, baseline_logg_key, baseline_title, _ = _resolve_keys(baseline_kind)

    baseline_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for record in baseline_records:
        pair = record.get('pair')
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        baseline_map[(int(pair[0]), int(pair[1]))] = record

    os.makedirs(output_dir, exist_ok=True)

    def _extract_xy(record: Optional[Dict[str, Any]],
                    key_teff: str,
                    key_logg: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if record is None:
            return None
        points = sorted(record.get('alpha_points', []), key=lambda x: x.get('alpha', 0.0))
        teff_vals = np.array([p.get(key_teff, np.nan) for p in points], dtype=float)
        logg_vals = np.array([p.get(key_logg, np.nan) for p in points], dtype=float)
        alphas = np.array([p.get('alpha', np.nan) for p in points], dtype=float)
        valid_mask = ~(np.isnan(teff_vals) | np.isnan(logg_vals))
        if not valid_mask.any():
            return None
        return teff_vals[valid_mask], logg_vals[valid_mask], alphas[valid_mask]

    def _get_logL_per_len_for_pair(model_label: str, pair_key: Tuple[int, int]) -> Optional[float]:
        if not metrics_by_model:
            return None
        metrics = metrics_by_model.get(model_label)
        if not isinstance(metrics, dict):
            return None
        per_pair_metrics = metrics.get('per_pair')
        if not isinstance(per_pair_metrics, list):
            return None
        for entry in per_pair_metrics:
            entry_pair = entry.get('pair')
            if not isinstance(entry_pair, (list, tuple)) or len(entry_pair) != 2:
                continue
            try:
                entry_pair_tuple = (int(entry_pair[0]), int(entry_pair[1]))
            except (TypeError, ValueError):
                continue
            if entry_pair_tuple != pair_key:
                continue
            value = entry.get('logL_per_len')
            try:
                value_float = float(value)
            except (TypeError, ValueError):
                return None
            return value_float if math.isfinite(value_float) else None
        return None

    def _format_label(base: str,
                      title: str,
                      logL_per_len: Optional[float]) -> str:
        if logL_per_len is not None:
            return f"{base} ({title}, logL_per_len={logL_per_len:.2f})"
        return f"{base} ({title}, logL_per_len=None)"

    baseline_point_color = 'blue' if baseline_kind == 'numeric' else 'purple'
    baseline_line_color = 'navy' if baseline_kind == 'numeric' else 'purple'
    model_point_color = 'orange' if kind == 'numeric' else 'green'
    model_line_color = 'darkorange' if kind == 'numeric' else 'darkgreen'

    for label, records in model_results.items():
        if label == baseline_label:
            continue

        pair_records = []
        for record in records or []:
            pair = record.get('pair')
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            pair_tuple = (int(pair[0]), int(pair[1]))
            baseline_record = baseline_map.get(pair_tuple)
            baseline_xy = _extract_xy(baseline_record, baseline_teff_key, baseline_logg_key)
            model_xy = _extract_xy(record, teff_key, logg_key)
            if baseline_xy is None and model_xy is None:
                continue
            pair_records.append((pair_tuple, baseline_xy, model_xy))

        if not pair_records:
            print(f"Warning: No overlapping interpolation data for {label} vs baseline ({kind_title}).")
            continue

        num_pairs = len(pair_records)
        ncols = min(3, max(1, num_pairs))
        nrows = math.ceil(num_pairs / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5.2 * nrows))
        axes_flat = np.atleast_1d(axes).flatten()

        for idx, (pair, baseline_xy, model_xy) in enumerate(pair_records):
            print(f"----idx: {idx}---- ")
            ax = axes_flat[idx]

            if background_points is not None and len(background_points[0]) > 0:
                ax.scatter(
                    background_points[0],
                    background_points[1],
                    color='lightgray',
                    alpha=0.15,
                    s=12,
                    edgecolors='none',
                )

            plotted_any = False
            legend_handles: List[Any] = []
            legend_labels: List[str] = []

            if baseline_xy is not None:
                teff_base, logg_base, _ = baseline_xy
                baseline_logL_per_len = _get_logL_per_len_for_pair(baseline_label, pair)
                print("baseline logL - ", baseline_logL_per_len)
                label_text = _format_label(
                    baseline_label,
                    baseline_title,
                    baseline_logL_per_len,
                )

                scatter_base = ax.scatter(
                    teff_base,
                    logg_base,
                    s=45,
                    edgecolors='k',
                    linewidths=0.3,
                    alpha=0.8,
                    color=baseline_point_color,
                    label=label_text,
                )
                ax.plot(
                    teff_base,
                    logg_base,
                    color=baseline_line_color,
                    alpha=0.8,
                )
                legend_handles.append(scatter_base)
                legend_labels.append(scatter_base.get_label())
                plotted_any = True
            else:
                ax.text(
                    0.02,
                    0.98,
                    f'No {kind_title.lower()} data for baseline',
                    transform=ax.transAxes,
                    ha='left',
                    va='top',
                    fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.6),
                )

            if model_xy is not None:
                teff_model, logg_model, _ = model_xy
                model_logL_per_len = _get_logL_per_len_for_pair(label, pair)
                print("model logL: ", model_logL_per_len)
                label_text = _format_label(
                    label,
                    kind_title,
                    model_logL_per_len,
                )

                scatter_model = ax.scatter(
                    teff_model,
                    logg_model,
                    s=45,
                    edgecolors='k',
                    linewidths=0.3,
                    alpha=0.8,
                    color=model_point_color,
                    label=label_text,
                )
                ax.plot(
                    teff_model,
                    logg_model,
                    color=model_line_color,
                    alpha=0.8,
                )
                legend_handles.append(scatter_model)
                legend_labels.append(scatter_model.get_label())
                plotted_any = True
            else:
                ax.text(
                    0.98,
                    0.02,
                    f'No {kind_title.lower()} data for {label}',
                    transform=ax.transAxes,
                    ha='right',
                    va='bottom',
                    fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.6),
                )

            ax.set_xlim(8000, 4000)
            ax.set_ylim(5.5, 2.0)
            ax.set_xlabel('Effective Temperature Teff (K)')
            ax.set_ylabel('Surface Gravity log g')
            ax.set_title(f'Pair {pair[0]} → {pair[1]} ({kind_title})')
            ax.grid(True, alpha=0.3)
            ax.invert_xaxis()
            ax.invert_yaxis()

            if legend_handles:
                ax.legend(legend_handles, legend_labels, loc='best', fontsize=8)

            if not plotted_any:
                ax.text(
                    0.5,
                    0.5,
                    'No valid data',
                    transform=ax.transAxes,
                    ha='center',
                    va='center',
                    fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                )

        # Hide unused axes
        for extra_ax in axes_flat[len(pair_records):]:
            extra_ax.set_visible(False)

        fig.tight_layout()
        filename = f'comparison_{plot_suffix}_vs_{sanitize_label(baseline_label)}_{sanitize_label(label)}.png'
        fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] {kind_title} comparison plot saved for {label} vs {baseline_label}")

def compare_models_on_interpolations(base_results: List[Dict[str, Any]],
                                     interpolation_pairs: List[Tuple[int, int]],
                                     background_points: Optional[Tuple[np.ndarray, np.ndarray]],
                                     base_trainer: LLMTrainer,
                                     inference_args,
                                     base_args: argparse.Namespace,
                                     device: torch.device,
                                     quantiles: List[float],
                                     plots_dir: str,
                                     dataset,
                                     collate_fn) -> None:
    """Run interpolation on additional checkpoints using identical pairs and overlay results."""
    comparison_paths = [path for path in (inference_args.comparison_checkpoint_paths or []) if path]
    comparison_labels = inference_args.comparison_labels or []
    if comparison_labels and len(comparison_labels) != len(comparison_paths):
        print("Warning: Number of comparison labels does not match number of checkpoints; falling back to auto labels.")

    interpolation_alphas = inference_args.interpolation_alphas or [i / 9 for i in range(10)]
    model_results: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
    metrics_by_model_numeric: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    metrics_by_model_text: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def _print_metrics_summary(label: str, metrics: Dict[str, Any], descriptor: str) -> None:
        summary = metrics.get('summary') or {}
        if not summary:
            return

        def _fmt(value: Optional[float]) -> str:
            return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"

        print(
            f"Interpolation metrics [{descriptor}] ({label}): "
            f"mean log-density={_fmt(summary.get('mean_log_density'))}, "
            f"mean NN distance={_fmt(summary.get('mean_nn_distance'))}, "
            f"mean smoothness={_fmt(summary.get('mean_smoothness'))}"
        )

    base_label = derive_model_label(inference_args.checkpoint_path)
    model_results[base_label] = base_results
    base_metrics_numeric = compute_interpolation_metrics(base_results, background_points)
    if base_metrics_numeric:
        metrics_by_model_numeric[base_label] = base_metrics_numeric
        _print_metrics_summary(base_label, base_metrics_numeric, 'numeric')

    base_metrics_text = compute_interpolation_metrics(
        base_results,
        background_points,
        teff_key='text_teff',
        logg_key='text_logg',
    )
    if base_metrics_text:
        metrics_by_model_text[base_label] = base_metrics_text
        _print_metrics_summary(base_label, base_metrics_text, 'text')

    collate_base = collate_fn or getattr(dataset, 'collate_fn', None)
    model_cache: Dict[Tuple, torch.nn.Module] = {}
    reusable_model = getattr(base_trainer, 'model', None) if base_trainer is not None else None
    base_signature = _extract_arch_signature(base_args) if base_args is not None else None

    tuned_cfg_path = os.path.join(ROOT_DIR, 'src', 'llm_config_tuned.json')
    base_cfg_path = os.path.join(ROOT_DIR, 'src', 'llm_config.json')

    comparison_output_dir = os.path.join(plots_dir, 'interpolation', 'comparison')
    os.makedirs(comparison_output_dir, exist_ok=True)
    combined_results_json: Dict[str, Any] = {base_label: sanitize_for_json(base_results)}

    baseline_checkpoint = getattr(inference_args, 'baseline_mlp_checkpoint_path', None)
    if baseline_checkpoint:
        if not os.path.isabs(baseline_checkpoint):
            baseline_checkpoint = os.path.join(ROOT_DIR, baseline_checkpoint)
    else:
        baseline_checkpoint = os.path.join(ROOT_DIR, 'logs', 'baseline_mlp', 'baseline_mlp.pt')

    baseline_label = getattr(inference_args, 'baseline_mlp_label', 'Baseline MLP')
    baseline_results = run_baseline_mlp_interpolation(
        checkpoint_path=baseline_checkpoint,
        dataset=dataset,
        pairs=interpolation_pairs,
        alphas=interpolation_alphas,
        device=device,
    )
    if baseline_results:
        model_results[baseline_label] = baseline_results
        combined_results_json[baseline_label] = sanitize_for_json(baseline_results)
        baseline_metrics_numeric = compute_interpolation_metrics(baseline_results, background_points)
        if baseline_metrics_numeric:
            metrics_by_model_numeric[baseline_label] = baseline_metrics_numeric
            baseline_metrics_for_text = deepcopy(baseline_metrics_numeric)
            baseline_metrics_for_text['reference_kind'] = 'numeric_baseline'
            metrics_by_model_text[baseline_label] = baseline_metrics_for_text
            _print_metrics_summary(baseline_label, baseline_metrics_numeric, 'numeric')

    # Free base trainer GPU memory before loading comparison models to avoid OOM
    if base_trainer is not None:
        base_model = reusable_model
        base_param_device: Optional[torch.device] = None
        if base_model is not None:
            try:
                base_param_device = next(base_model.parameters()).device  # type: ignore[attr-defined]
            except StopIteration:
                base_param_device = None
            except Exception:
                base_param_device = None

        if base_model is not None and base_param_device is not None and base_param_device.type == 'cuda':
            base_model = base_model.to('cpu')
            base_trainer.model = base_model
            reusable_model = base_model
            if hasattr(base_trainer, 'optimizer'):
                base_trainer.optimizer = None  # type: ignore[attribute-defined-outside-init]
            if hasattr(base_trainer, 'scheduler'):
                base_trainer.scheduler = None  # type: ignore[attribute-defined-outside-init]
            if hasattr(base_trainer, 'scaler'):
                base_trainer.scaler = None  # type: ignore[attribute-defined-outside-init]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if reusable_model is not None and base_signature is not None:
        model_cache[base_signature] = reusable_model

    for idx, checkpoint_path in enumerate(comparison_paths):
        label = None
        if comparison_labels:
            label = comparison_labels[idx] if idx < len(comparison_labels) else None
        if not label:
            label = derive_model_label(checkpoint_path)
        label = label.strip()
        print(f"\n=== Running interpolation comparison for model '{label}' ===")

        comp_namespace = argparse.Namespace(**vars(inference_args))
        comp_namespace.checkpoint_path = checkpoint_path
        comp_namespace.output_dir = os.path.join(inference_args.output_dir, f'comparison_{sanitize_label(label)}')

        try:
            comp_config = load_config_from_checkpoint_dir(checkpoint_path)
        except Exception as exc:
            print(f"Warning: Failed to load config for {checkpoint_path} ({exc}); skipping.")
            continue

        comp_args = create_args_from_config(comp_config, comp_namespace)
        comp_args.output_dir = comp_namespace.output_dir
        comp_args.single_sample_prob = 1.0
        comp_args.mode = "single_star"

        backend_config = ensure_backend_config(comp_args)

        arch_signature = _extract_arch_signature(comp_args)
        cached_model = model_cache.get(arch_signature)
        try:
            model = load_model(checkpoint_path, comp_args, device, model=cached_model)
        except Exception as exc:
            print(f"Warning: Failed to prepare model for {label} ({exc}); skipping.")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        model_cache[arch_signature] = model

        try:
            train_loader_cmp, val_loader_cmp, test_loader_cmp = create_datasets_and_loaders(comp_args, device, backend_config)
        except Exception as exc:
            print(f"Warning: Failed to create datasets for {label} ({exc}); skipping.")
            model.to('cpu')
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        try:
            optimizer_cmp, scheduler_cmp, scaler_cmp = create_optimizer_and_scheduler(model, comp_args, train_loader_cmp)
        except Exception as exc:
            print(f"Warning: Failed to create optimizer for {label} ({exc}); skipping.")
            del train_loader_cmp, val_loader_cmp, test_loader_cmp
            model.to('cpu')
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if os.path.isfile(tuned_cfg_path):
            with open(tuned_cfg_path, 'r') as f:
                tuned_cfg = json.load(f)
            lora_params = tuned_cfg.get('lora_params', {})
        else:
            with open(base_cfg_path, 'r') as f:
                base_cfg = json.load(f)
            lora_params = base_cfg.get('lora_params', {})

        trainer_cmp = LLMTrainer(
            model=model,
            optimizer=optimizer_cmp,
            criterion=torch.nn.CrossEntropyLoss(),
            train_dataloader=train_loader_cmp,
            val_dataloader=val_loader_cmp,
            device=device,
            world_size=getattr(base_trainer, 'world_size', 1),
            output_dim=1,
            scheduler=scheduler_cmp,
            max_iter=comp_args.max_iter,
            log_path=comp_args.output_dir,
            exp_name=comp_args.exp_name,
            lora_params=lora_params,
            scaler=scaler_cmp,
            use_amp=comp_args.use_amp,
            max_grad_norm=comp_args.max_grad_norm,
            mode=comp_args.mode,
            curriculum_decay_steps=comp_args.curriculum_decay_steps,
            quantiles=comp_args.quantiles,
        )
        trainer_cmp.combined_mode = (comp_args.mode == "combined")

        interpolation_dir_model = os.path.join(plots_dir, 'interpolation', sanitize_label(label))
        effective_collate = collate_base or getattr(test_loader_cmp, 'collate_fn', None)
        if effective_collate is None:
            print(f"Warning: No collate function available for comparison model '{label}'; skipping.")
            del trainer_cmp, optimizer_cmp, scheduler_cmp, scaler_cmp
            del train_loader_cmp, val_loader_cmp, test_loader_cmp
            model.to('cpu')
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        try:
            results = run_feature_interpolation(
                trainer=trainer_cmp,
                args=comp_args,
                dataset=dataset,
                collate_fn=effective_collate,
                device=device,
                pairs=interpolation_pairs,
                alphas=interpolation_alphas,
                output_dir=interpolation_dir_model,
                quantiles=quantiles,
                min_teff_diff=0.0,
                min_logg_diff=0.0,
            )
        except Exception as exc:
            print(f"Warning: Interpolation failed for {label} ({exc}); skipping.")
            del trainer_cmp, optimizer_cmp, scheduler_cmp, scaler_cmp
            del train_loader_cmp, val_loader_cmp, test_loader_cmp
            model.to('cpu')
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if not results:
            print(f"Warning: No interpolation results produced for {label}; skipping.")
            del trainer_cmp, optimizer_cmp, scheduler_cmp, scaler_cmp
            del train_loader_cmp, val_loader_cmp, test_loader_cmp
            model.to('cpu')
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        model_results[label] = results
        combined_results_json[label] = sanitize_for_json(results)
        metrics_numeric = compute_interpolation_metrics(results, background_points)
        if metrics_numeric:
            metrics_by_model_numeric[label] = metrics_numeric
            _print_metrics_summary(label, metrics_numeric, 'numeric')

        metrics_text = compute_interpolation_metrics(
            results,
            background_points,
            teff_key='text_teff',
            logg_key='text_logg',
        )
        if metrics_text:
            metrics_by_model_text[label] = metrics_text
            _print_metrics_summary(label, metrics_text, 'text')

        # Clean up to free GPU memory before loading the next model
        del trainer_cmp, optimizer_cmp, scheduler_cmp, scaler_cmp
        del train_loader_cmp, val_loader_cmp, test_loader_cmp
        model.to('cpu')
        if arch_signature == base_signature and base_trainer is not None:
            base_trainer.model = model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if base_signature is not None and base_signature in model_cache:
        try:
            base_model_restore = model_cache[base_signature]
            load_model(inference_args.checkpoint_path, base_args, device, model=base_model_restore)
            base_model_restore.to('cpu')
            if base_trainer is not None:
                base_trainer.model = base_model_restore
        except Exception as exc:
            print(f"Warning: Failed to restore base model weights ({exc})")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if len(model_results) <= 1:
        print("No additional models produced interpolation results; skipping comparison plot.")
        return

    comparison_plot_dir = os.path.join(plots_dir, 'interpolation', 'comparison')
    plot_interpolation_kiel_multi_model(model_results, comparison_plot_dir, background_points, interpolation_pairs)

    numeric_baseline_label = baseline_label if baseline_results else base_label
    text_baseline_label = numeric_baseline_label
    plot_interpolation_pairwise_vs_baseline(
        model_results,
        numeric_baseline_label,
        comparison_plot_dir,
        kind='numeric',
        background_points=background_points,
        metrics_by_model=metrics_by_model_numeric,
    )
    plot_interpolation_pairwise_vs_baseline(
        model_results,
        text_baseline_label,
        comparison_plot_dir,
        kind='text',
        background_points=background_points,
        baseline_kind='numeric',
        metrics_by_model=metrics_by_model_text,
    )

    combined_json_path = os.path.join(comparison_plot_dir, 'comparison_interpolation_results.json')
    try:
        with open(combined_json_path, 'w') as f:
            json.dump(combined_results_json, f, indent=2, allow_nan=False)
        print(f"[OK] Combined comparison results saved to {combined_json_path}")
    except Exception as exc:
        print(f"Warning: Failed to save combined comparison results ({exc})")

    metrics_payload: Dict[str, Dict[str, Any]] = {}
    if metrics_by_model_numeric:
        metrics_payload['numeric'] = {label: sanitize_for_json(metrics) for label, metrics in metrics_by_model_numeric.items()}
    if metrics_by_model_text:
        metrics_payload['text'] = {label: sanitize_for_json(metrics) for label, metrics in metrics_by_model_text.items()}

    if metrics_payload:
        metrics_path = os.path.join(comparison_plot_dir, 'comparison_interpolation_metrics.json')
        try:
            with open(metrics_path, 'w') as f:
                json.dump(metrics_payload, f, indent=2, allow_nan=False)
            print(f"[OK] Comparison metrics saved to {metrics_path}")
        except Exception as exc:
            print(f"Warning: Failed to save comparison metrics ({exc})")

    if metrics_by_model_numeric:
        plot_interpolation_metrics_summary(
            metrics_by_model_numeric,
            comparison_plot_dir,
            kind_label='numeric',
        )
    if metrics_by_model_text:
        plot_interpolation_metrics_summary(
            metrics_by_model_text,
            comparison_plot_dir,
            kind_label='text',
        )


def run_interpolation(trainer: LLMTrainer,
                      test_loader,
                      device: torch.device,
                      args,
                      inference_args,
                      quantiles: List[float],
                      plots_dir: str) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]], Tuple[np.ndarray, np.ndarray]]:
    """Orchestrate interpolation pair selection, execution, and plotting."""
    dataset = test_loader.dataset
    background_points = collect_kiel_background(dataset)
    if background_points[0].size:
        print(f"Collected {background_points[0].size} background samples for Kiel plots.")
    else:
        print("Warning: Unable to collect background points for Kiel plots.")

    print("interpolation pairs: ", inference_args.interpolation_pairs)
    interpolation_pairs = parse_interpolation_pairs(inference_args.interpolation_pairs)
    if not interpolation_pairs and getattr(inference_args, 'interpolation_num_pairs', 0) > 0:
        interpolation_pairs = select_random_interpolation_pairs(
            dataset,
            trainer,
            inference_args.interpolation_num_pairs,
            inference_args.interpolation_seed,
            inference_args.interpolation_min_teff_diff,
            getattr(inference_args, 'interpolation_min_logg_diff', 0.0),
        )
        if interpolation_pairs:
            print(f"Selected {len(interpolation_pairs)} random interpolation pairs (seed {inference_args.interpolation_seed}).")
        else:
            print("Warning: Unable to select interpolation pairs automatically.")

    interpolation_results: List[Dict[str, Any]] = []
    if interpolation_pairs:
        print("running on interpolated features")
        interpolation_alphas = inference_args.interpolation_alphas or [i / 9 for i in range(10)]
        interpolation_dir = os.path.join(plots_dir, 'interpolation')
        interpolation_results = run_feature_interpolation(
            trainer=trainer,
            args=args,
            dataset=dataset,
            collate_fn=getattr(test_loader, 'collate_fn', None),
            device=device,
            pairs=interpolation_pairs,
            alphas=interpolation_alphas,
            output_dir=interpolation_dir,
            quantiles=quantiles,
            min_teff_diff=inference_args.interpolation_min_teff_diff,
            min_logg_diff=getattr(inference_args, 'interpolation_min_logg_diff', 0.0),
        )

        if inference_args.plot_results and interpolation_results:
            plot_interpolation_kiel(interpolation_results, interpolation_dir, background_points)
            plot_interpolation_kiel_combined(interpolation_results, interpolation_dir, background_points)

        metrics_payload = compute_interpolation_metrics(interpolation_results, background_points)
        if metrics_payload:
            metrics_path = os.path.join(interpolation_dir, 'interpolation_metrics.json')
            try:
                with open(metrics_path, 'w') as f:
                    json.dump(metrics_payload, f, indent=2)
                print(f"[OK] Interpolation metrics saved to {metrics_path}")
            except Exception as exc:
                print(f"Warning: Failed to save interpolation metrics ({exc})")

            summary = metrics_payload.get('summary') or {}
            if summary:
                def _fmt(value: Optional[float]) -> str:
                    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"
                print(
                    "Interpolation metrics summary: "
                    f"mean log-density={_fmt(summary.get('mean_log_density'))}, "
                    f"mean NN distance={_fmt(summary.get('mean_nn_distance'))}, "
                    f"mean smoothness={_fmt(summary.get('mean_smoothness'))}"
                )

    return interpolation_results, interpolation_pairs, background_points


def _prepare_background_density(background_points: Tuple[np.ndarray, np.ndarray],
                                k_neighbors: int = 32) -> Optional[Tuple[Any, np.ndarray, int]]:
    """Prepare KD-tree based density helper for background points."""
    if not background_points:
        return None

    bg_teff, bg_logg = background_points
    if bg_teff is None or bg_logg is None:
        return None
    if bg_teff.size == 0 or bg_logg.size == 0:
        return None

    coords = np.column_stack((bg_teff.astype(float), bg_logg.astype(float)))
    if coords.shape[0] < 2:
        return None

    effective_k = min(max(k_neighbors, 1), coords.shape[0])

    if cKDTree is not None:
        tree = cKDTree(coords)

        def query_fn(points: np.ndarray, k: int) -> np.ndarray:
            dist, _ = tree.query(points, k=k)
            return dist
    elif 'SKKDTree' in globals() and SKKDTree is not None:
        tree = SKKDTree(coords)

        def query_fn(points: np.ndarray, k: int) -> np.ndarray:
            dist, _ = tree.query(points, k=k, return_distance=True)
            return dist
    else:
        return None

    return query_fn, coords, effective_k


def _estimate_log_likelihood(teff: np.ndarray,
                             logg: np.ndarray,
                             density_helper: Optional[Tuple[Any, np.ndarray, int]]) -> Optional[np.ndarray]:
    """Estimate log-likelihood values for sets of Teff/log g points."""
    if density_helper is None:
        return None
    if teff.size == 0 or logg.size == 0:
        return None

    query_fn, coords, effective_k = density_helper
    points = np.column_stack((teff.astype(float), logg.astype(float)))
    dist_matrix = query_fn(points, effective_k)
    if effective_k == 1:
        dist_matrix = dist_matrix.reshape(-1, 1)
    dist_matrix = np.maximum(dist_matrix, 1e-6)
    radius_k = dist_matrix[:, -1]
    area = math.pi * np.square(np.maximum(radius_k, 1e-6))
    density_vals = effective_k / (coords.shape[0] * area)
    log_density_vals = np.log(density_vals + 1e-12)
    return log_density_vals


def run_feature_similarity_trajectory(trainer: LLMTrainer,
                                     args,
                                     dataset,
                                     collate_fn,
                                     device: Union[int, torch.device],
                                     star1_idx: int,
                                     max_trajectory_length: int = 10,
                                     output_dir: str = None,
                                     random_state: int = 42) -> Dict[str, Any]:
    """
    Create a trajectory of feature predictions using one fixed star1 and 
    a sequence of star2s with decreasing similarity to star1.
    
    Args:
        trainer: The trained model
        args: Training arguments  
        dataset: The dataset (should use random_pairing=False for similarity-based pairing)
        collate_fn: Data collation function
        device: Device to run inference on
        star1_idx: Index of the fixed star1 in the dataset
        max_trajectory_length: Maximum number of star2s in the trajectory
        output_dir: Directory to save results
        random_state: Random seed for reproducibility
        
    Returns:
        Dict containing predicted and real feature trajectories, stellar parameters, etc.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    
    # Ensure we're using similarity-based pairing for finding neighbors
    if getattr(dataset, 'random_pairing', True):
        print("Warning: Dataset is using random pairing. Setting to similarity-based for trajectory.")
        dataset.random_pairing = False
        dataset._precomputed_pairs = None  # Clear existing pairs
        dataset._precompute_star_pairs(cache_dir=None)  # Recompute with similarity
    
    model = getattr(trainer, 'model', None)
    model_was_training = False
    if model is not None:
        model_was_training = model.training
        model.eval()
    
    if isinstance(device, int):
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{device}')
        else:
            device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device) and 'cuda' in device.type and not torch.cuda.is_available():
        device = torch.device('cpu')
    
    # Get star1 data
    star1_sample = dataset[star1_idx]
    star1_params = star1_sample['stellar_params_star1']
    star1_features = star1_sample['features_star1'].cpu().numpy()
    
    print(f"Star1 parameters: Teff={star1_params['Teff']:.2f}, logg={star1_params['logg']:.2f}, FeH={star1_params['FeH']:.2f}")
    
    # Extract all valid stellar parameters for neighbor search
    all_params = []
    valid_indices = []
    
    for idx in range(len(dataset.split_indices)):
        try:
            sample = dataset.raw_data[dataset.split_indices[idx]]
            stellar_data = sample.get('stellar_data', {})
            params = dataset._extract_stellar_params(stellar_data)
            
            # Check if parameters are valid and different from star1
            if not any(v is None for v in params.values()) and idx != star1_idx:
                normalized = dataset._normalize_params(params)
                all_params.append(normalized)
                valid_indices.append(idx)
        except:
            continue
    
    if len(all_params) < max_trajectory_length:
        print(f"Warning: Only {len(all_params)} valid samples available, using all of them")
        max_trajectory_length = len(all_params)
    
    # Normalize star1 parameters for neighbor search
    star1_normalized = dataset._normalize_params(star1_params)
    
    # Find neighbors with increasing distance
    all_params = np.array(all_params)
    nn_index = NearestNeighbors(n_neighbors=len(all_params), algorithm='ball_tree', metric='euclidean')
    nn_index.fit(all_params)
    
    # Query for all neighbors
    distances, indices = nn_index.kneighbors([star1_normalized], n_neighbors=min(max_trajectory_length, len(all_params)))
    
    # Create trajectory: star2 indices with increasing distance from star1
    trajectory_indices = []
    trajectory_distances = []
    for i, nn_idx in enumerate(indices[0][:max_trajectory_length]):
        dataset_idx = valid_indices[nn_idx]
        trajectory_indices.append(dataset_idx)
        trajectory_distances.append(distances[0][i])
    
    print(f"Created trajectory with {len(trajectory_indices)} star2s, distances: {trajectory_distances}")
    
    # Run predictions for each star2 in trajectory
    predicted_features = []
    real_features = []
    star2_params_list = []
    
    collate = collate_fn
    if collate is None and hasattr(dataset, 'collate_fn'):
        collate = dataset.collate_fn
    
    for i, star2_idx in enumerate(trajectory_indices):
        # Create modified sample with fixed star1 and current star2
        sample = dataset[star1_idx].copy() if hasattr(dataset[star1_idx], 'copy') else dict(dataset[star1_idx])
        
        # Get star2 data
        star2_sample = dataset[star2_idx]
        star2_params = star2_sample['stellar_params_star2']
        star2_features = star2_sample['features_star2'].cpu().numpy()
        
        # Update sample with new star2 data
        sample['stellar_params_star2'] = star2_params
        sample['features_star2'] = star2_sample['features_star2']
        sample['y_numeric_star2'] = star2_sample['y_numeric_star2']
        
        # Create new question text for this star2
        question = (f"this is a star with Teff {star1_params['Teff']:.2f} K, "
                   f"logg {star1_params['logg']:.2f}, and FeH {star1_params['FeH']:.2f}. "
                   f"describe a star with Teff {star2_params['Teff']:.2f} K, "
                   f"logg {star2_params['logg']:.2f}, and FeH {star2_params['FeH']:.2f}")
        sample['input_text'] = question
        
        # Run inference
        batch = collate([sample]) if collate else [sample]
        
        # try:
        with torch.no_grad():
            if hasattr(trainer, 'predict_batch'):
                outputs = trainer.predict_batch(batch, device=device)
            elif hasattr(trainer.model, 'predict_features'):
                batch_device = {}
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch_device[k] = v.to(device)
                    else:
                        batch_device[k] = v
                outputs = trainer.model(batch_device)
            else:
                print(f"Warning: Cannot run prediction for trajectory point {i}")
                continue
            
            # Extract predicted features
            if 'predicted_features' in outputs:
                pred_feat = outputs['predicted_features'][0].cpu().numpy()
            elif 'feature_predictions' in outputs:
                pred_feat = outputs['feature_predictions'][0].cpu().numpy()
            else:
                print(f"Warning: No predicted features found in outputs at trajectory point {i}")
                continue
            
            predicted_features.append(pred_feat)
            real_features.append(star2_features)
            star2_params_list.append(star2_params)
            
            print(f"Trajectory point {i+1}/{len(trajectory_indices)}: "
                    f"Teff={star2_params['Teff']:.2f}, distance={trajectory_distances[i]:.4f}")
                
        # except Exception as e:
        #     print(f"Error processing trajectory point {i}: {e}")
        #     continue
    
    # Restore model training state
    if model is not None and model_was_training:
        model.train()
    
    # Package results
    result = {
        'star1_idx': star1_idx,
        'star1_params': star1_params,
        'star1_features': star1_features,
        'trajectory_indices': trajectory_indices,
        'trajectory_distances': trajectory_distances,
        'star2_params_list': star2_params_list,
        'predicted_features': np.array(predicted_features) if predicted_features else np.array([]),
        'real_features': np.array(real_features) if real_features else np.array([]),
        'trajectory_length': len(predicted_features)
    }
    
    # Save results if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'feature_trajectory_star1_{star1_idx}.json')
        
        # Convert numpy arrays to lists for JSON serialization
        json_result = result.copy()
        for key in ['star1_features', 'predicted_features', 'real_features']:
            if key in json_result and isinstance(json_result[key], np.ndarray):
                json_result[key] = json_result[key].tolist()
        
        try:
            with open(output_file, 'w') as f:
                json.dump(json_result, f, indent=2)
            print(f"Saved trajectory results to {output_file}")
        except Exception as e:
            print(f"Warning: Failed to save trajectory results: {e}")
    
    return result


def plot_feature_trajectory_umap(trajectory_result: Dict[str, Any],
                                output_dir: str = None,
                                background_features: np.ndarray = None,
                                background_params: List[Dict] = None,
                                n_neighbors: int = 15,
                                min_dist: float = 0.1,
                                random_state: int = 42):
    """
    Visualize predicted vs real feature trajectories in UMAP space.
    
    Args:
        trajectory_result: Result from run_feature_similarity_trajectory
        output_dir: Directory to save plots
        background_features: Background features to show as context (optional)
        background_params: Background stellar parameters (optional)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter  
        random_state: Random seed for UMAP
    """
    try:
        import umap
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        print(f"Error: Missing required package for UMAP visualization: {e}")
        return
    
    predicted_features = trajectory_result['predicted_features']
    real_features = trajectory_result['real_features']
    star1_features = trajectory_result['star1_features']
    trajectory_distances = trajectory_result['trajectory_distances']
    star2_params_list = trajectory_result['star2_params_list']
    
    if len(predicted_features) == 0 or len(real_features) == 0:
        print("Warning: No features to plot")
        return
    
    # Combine all features for UMAP fitting
    trajectory_features = np.vstack([
        star1_features.reshape(1, -1),
        predicted_features,
        real_features
    ])
    
    # Include background features if provided
    if background_features is not None:
        all_features = np.vstack([trajectory_features, background_features])
        n_background = len(background_features)
    else:
        all_features = trajectory_features
        n_background = 0
    
    # Fit UMAP
    print(f"Fitting UMAP on {all_features.shape[0]} feature vectors...")
    umap_model = umap.UMAP(
        n_neighbors=min(n_neighbors, all_features.shape[0] - 1),
        min_dist=min_dist,
        random_state=random_state,
        n_components=2
    )
    
    embeddings = umap_model.fit_transform(all_features)
    
    # Split embeddings
    n_trajectory = len(trajectory_features)
    trajectory_embeddings = embeddings[:n_trajectory]
    
    star1_embedding = trajectory_embeddings[0:1]
    pred_embeddings = trajectory_embeddings[1:1+len(predicted_features)]
    real_embeddings = trajectory_embeddings[1+len(predicted_features):]
    
    # Background embeddings (if any)
    if n_background > 0:
        background_embeddings = embeddings[n_trajectory:]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot background features if provided
    if n_background > 0:
        plt.scatter(background_embeddings[:, 0], background_embeddings[:, 1], 
                   c='lightgray', s=20, alpha=0.5, 
                   label=f'Background features (n={n_background})', zorder=1)
    
    # Plot trajectories
    plt.plot(pred_embeddings[:, 0], pred_embeddings[:, 1], 
             'o-', color='red', alpha=0.7, linewidth=2, markersize=8,
             label='Predicted trajectory')
    plt.plot(real_embeddings[:, 0], real_embeddings[:, 1], 
             'o-', color='blue', alpha=0.7, linewidth=2, markersize=8,
             label='Real trajectory')
    
    # Plot star1 (fixed reference)
    plt.scatter(star1_embedding[0, 0], star1_embedding[0, 1], 
               color='green', s=200, marker='*', 
               label='Star1 (reference)', zorder=5)
    
    # Add distance annotations
    # for i, (pred_emb, real_emb, dist, params) in enumerate(zip(
    #     pred_embeddings, real_embeddings, trajectory_distances, star2_params_list)):
        
    #     plt.annotate(f'{i+1}\nd={dist:.3f}', 
    #                 xy=pred_emb, xytext=(5, 5), 
    #                 textcoords='offset points', fontsize=8,
    #                 bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3))
    #     plt.annotate(f'{i+1}\nd={dist:.3f}', 
    #                 xy=real_emb, xytext=(5, 5), 
    #                 textcoords='offset points', fontsize=8,
    #                 bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.3))
    
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title(f'Feature Prediction Trajectory (Star1 idx={trajectory_result["star1_idx"]})\n'
              f'Predicted vs Real Features in UMAP Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add parameter info as text
    star1_params = trajectory_result['star1_params']
    info_text = (f"Star1: Teff={star1_params['Teff']:.0f}K, "
                f"logg={star1_params['logg']:.1f}, FeH={star1_params['FeH']:.2f}")
    plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_file = os.path.join(output_dir, f'feature_trajectory_umap_star1_{trajectory_result["star1_idx"]}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Saved UMAP trajectory plot to {plot_file}")
    
    plt.show()
    
    # Create distance vs similarity plot
    plt.figure(figsize=(10, 6))
    
    # Calculate feature similarities (cosine similarity)
    from sklearn.metrics.pairwise import cosine_similarity
    
    star1_feat = star1_features.reshape(1, -1)
    pred_similarities = [cosine_similarity(star1_feat, pred.reshape(1, -1))[0, 0] 
                        for pred in predicted_features]
    real_similarities = [cosine_similarity(star1_feat, real.reshape(1, -1))[0, 0] 
                        for real in real_features]
    
    plt.subplot(1, 2, 1)
    plt.plot(trajectory_distances, pred_similarities, 'o-', color='red', label='Predicted')
    plt.plot(trajectory_distances, real_similarities, 'o-', color='blue', label='Real')
    plt.xlabel('Parameter Distance from Star1')
    plt.ylabel('Feature Cosine Similarity to Star1')
    plt.title('Similarity vs Parameter Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Feature prediction error vs distance
    plt.subplot(1, 2, 2)
    mse_errors = [np.mean((pred - real)**2) for pred, real in zip(predicted_features, real_features)]
    plt.plot(trajectory_distances, mse_errors, 'o-', color='purple')
    plt.xlabel('Parameter Distance from Star1')
    plt.ylabel('Feature Prediction MSE')
    plt.title('Prediction Error vs Distance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        plot_file = os.path.join(output_dir, f'feature_trajectory_analysis_star1_{trajectory_result["star1_idx"]}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Saved analysis plot to {plot_file}")
    
    plt.show()


def run_multiple_feature_trajectories(trainer: LLMTrainer,
                                     args,
                                     dataset,
                                     collate_fn,
                                     device: Union[int, torch.device],
                                     num_trajectories: int = 5,
                                     max_trajectory_length: int = 10,
                                     output_dir: str = None,
                                     random_state: int = 42) -> List[Dict[str, Any]]:
    """
    Run feature similarity trajectories for multiple random star1s.
    
    Args:
        trainer: The trained model
        args: Training arguments
        dataset: The dataset 
        collate_fn: Data collation function
        device: Device to run inference on
        num_trajectories: Number of different star1s to test
        max_trajectory_length: Maximum number of star2s in each trajectory
        output_dir: Directory to save results
        random_state: Random seed for reproducibility
        
    Returns:
        List of trajectory results
    """
    import numpy as np
    
    # Set random seed for star1 selection
    np.random.seed(random_state)
    
    # Randomly select star1 indices
    star1_indices = np.random.choice(len(dataset), size=min(num_trajectories, len(dataset)), replace=False)
    
    print(f"Running {len(star1_indices)} feature trajectories...")
    
    # Extract background features from a sample of the dataset for UMAP background
    print("Extracting background features for UMAP visualization...")
    background_sample_size = min(1000, len(dataset))  # Sample up to 1000 points for background
    background_indices = np.random.choice(len(dataset), size=background_sample_size, replace=False)
    
    background_features = []
    background_params = []
    
    for bg_idx in background_indices:
        sample = dataset[bg_idx]
        features = sample.get('features_star2', sample.get('features'))
        if features is not None:
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()
            background_features.append(features)
            
            # Extract stellar parameters if available
            params = {}
            for key in ['Teff', 'logg', 'FeH']:
                if key in sample:
                    params[key] = sample[key]
            background_params.append(params)
    
    if background_features:
        background_features = np.array(background_features)
        print(f"Extracted {len(background_features)} background features for UMAP")
    else:
        background_features = None
        background_params = None
        print("Warning: No background features found")

    results = []
    for i, star1_idx in enumerate(star1_indices):
        print(f"\n=== Trajectory {i+1}/{len(star1_indices)} (Star1 idx={star1_idx}) ===")
        
        # try:
        result = run_feature_similarity_trajectory(
            trainer=trainer,
            args=args,
            dataset=dataset,
            collate_fn=collate_fn,
            device=device,
            star1_idx=int(star1_idx),
            max_trajectory_length=max_trajectory_length,
            output_dir=output_dir,
            random_state=random_state + i
        )
        
        results.append(result)
        
        # Generate plots for this trajectory
        if output_dir and result['trajectory_length'] > 0:
            try:
                plot_feature_trajectory_umap(result, 
                                           output_dir=output_dir, 
                                           background_features=background_features,
                                           background_params=background_params,
                                           random_state=random_state + i)
            except Exception as e:
                print(f"Warning: Failed to plot trajectory {i+1}: {e}")
                    
        # except Exception as e:
        #     print(f"Error processing trajectory {i+1} (Star1 idx={star1_idx}): {e}")
        #     continue
    
    print(f"\nCompleted {len(results)} trajectories successfully")
    
    # Save combined results
    if output_dir and results:
        os.makedirs(output_dir, exist_ok=True)
        combined_file = os.path.join(output_dir, 'all_feature_trajectories.json')
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = []
        for result in results:
            json_result = result.copy()
            for key in ['star1_features', 'predicted_features', 'real_features']:
                if key in json_result and isinstance(json_result[key], np.ndarray):
                    json_result[key] = json_result[key].tolist()
            json_results.append(json_result)
        
        try:
            with open(combined_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"Saved combined trajectory results to {combined_file}")
        except Exception as e:
            print(f"Warning: Failed to save combined results: {e}")
    
    return results


def compute_interpolation_metrics(interpolation_results: List[Dict[str, Any]],
                                  background_points: Tuple[np.ndarray, np.ndarray],
                                  k_neighbors: int = 32,
                                  teff_key: str = 'teff',
                                  logg_key: str = 'logg',
                                  tube_factor: float = 1.0) -> Optional[Dict[str, Any]]:
    """
    Compute path likelihood metrics for interpolation trajectories.

    Metrics include:
    - Path-level log-likelihood: line integral of local density along the trajectory.
    - NN distance diagnostics.
    - Smoothness (curvature proxy).
    
    Parameters
    ----------
    interpolation_results : list of dicts
        Each record contains trajectory info (alpha_points).
    background_points : tuple of arrays
        (teff, logg) arrays of background stars.
    k_neighbors : int
        Number of neighbors for density estimate.
    teff_key, logg_key : str
        Keys in trajectory points.
    tube_factor : float
        Scale factor for local tube radius around the trajectory.
    """
    if not interpolation_results:
        return None

    density_helper = _prepare_background_density(background_points, k_neighbors=k_neighbors)
    if density_helper is None:
        print("Warning: Cannot compute interpolation metrics without background points.")
        return None

    query_fn, coords, effective_k = density_helper

    per_pair_metrics: List[Dict[str, Any]] = []

    for record in interpolation_results:
        points = sorted(record.get('alpha_points', []), key=lambda x: x.get('alpha', 0.0))
        if not points:
            continue

        alphas = np.array([p.get('alpha', np.nan) for p in points], dtype=float)
        teff = np.array([p.get(teff_key, np.nan) for p in points], dtype=float)
        logg = np.array([p.get(logg_key, np.nan) for p in points], dtype=float)

        valid_mask = ~(np.isnan(alphas) | np.isnan(teff) | np.isnan(logg))
        if not valid_mask.any():
            continue

        alphas_valid = alphas[valid_mask]
        coords_valid = np.column_stack((teff[valid_mask], logg[valid_mask])).astype(float)

        # Distances to background neighbors
        dist_matrix = query_fn(coords_valid, effective_k)
        if effective_k == 1:
            dist_matrix = dist_matrix.reshape(-1, 1)
        dist_matrix = np.maximum(dist_matrix, 1e-6)

        radius_k = dist_matrix[:, -1]
        area = math.pi * np.square(np.maximum(radius_k, 1e-6))
        density_vals = effective_k / (coords.shape[0] * area)
        log_density_vals = np.log(density_vals + 1e-12)

        nn_dist = dist_matrix[:, 0]

        # Arc-length parameterization of path
        seg_lengths = np.linalg.norm(np.diff(coords_valid, axis=0), axis=1)
        seg_lengths = np.r_[seg_lengths, seg_lengths[-1]]  # extend to same size as coords
        total_length = float(np.sum(seg_lengths))

        # Tube correction: area of a 2D disk with radius proportional to local kNN
        eps = tube_factor * radius_k
        log_A = np.log(np.pi) + 2 * np.log(np.maximum(eps, 1e-8))

        # Path log-likelihood: line integral of density along trajectory
        logL_path = float(np.sum(seg_lengths * (log_density_vals + log_A)))
        logL_per_len = logL_path / total_length if total_length > 0 else float('nan')

        # Smoothness via finite differences
        if coords_valid.shape[0] >= 3:
            delta = np.diff(coords_valid, axis=0)
            delta_alpha = np.diff(alphas_valid)
            delta_alpha = np.maximum(delta_alpha, 1e-6)
            velocities = delta / delta_alpha[:, None]
            if velocities.shape[0] >= 2:
                accelerations = np.diff(velocities, axis=0)
                smoothness = float(np.mean(np.linalg.norm(accelerations, axis=1)))
            else:
                smoothness = 0.0
        else:
            smoothness = 0.0

        per_pair_metrics.append({
            'pair': list(record.get('pair', (None, None))),
            'logL_path': logL_path,
            'logL_per_len': logL_per_len,
            'mean_nn_distance': float(np.mean(nn_dist)),
            'max_nn_distance': float(np.max(nn_dist)),
            'smoothness': smoothness,
            'path_length': total_length,
        })

    if not per_pair_metrics:
        print("Warning: Unable to compute interpolation metrics; no valid trajectory points.")
        return None

    summary = {
        'mean_logL_path': float(np.mean([m['logL_path'] for m in per_pair_metrics])),
        'mean_logL_per_len': float(np.mean([m['logL_per_len'] for m in per_pair_metrics])),
        'mean_nn_distance': float(np.mean([m['mean_nn_distance'] for m in per_pair_metrics])),
        'max_nn_distance': float(np.max([m['max_nn_distance'] for m in per_pair_metrics])),
        'mean_smoothness': float(np.mean([m['smoothness'] for m in per_pair_metrics])),
        'smoothness_std': float(np.std([m['smoothness'] for m in per_pair_metrics])),
    }

    return {
        'summary': summary,
        'per_pair': per_pair_metrics,
        'k_neighbors': int(effective_k),
        'background_point_count': int(coords.shape[0]),
        'teff_key': teff_key,
        'logg_key': logg_key,
    }



def plot_stellar_vs_text_consistency(stellar_preds_quantiles: Dict[str, np.ndarray], 
                                   text_predictions: List[str], 
                                   output_dir: str):
    """Plot consistency between stellar parameter predictions and text-extracted predictions"""
    
    if not stellar_preds_quantiles or not text_predictions:
        print("No data available for consistency plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract median predictions from quantiles for consistency comparison
    stellar_medians = {}
    for param, quantiles in stellar_preds_quantiles.items():
        median_idx = quantiles.shape[1] // 2
        stellar_medians[param] = quantiles[:, median_idx]
    
    # Extract parameters from all text predictions
    text_extracted = {'Teff': [], 'logg': [], 'FeH': []}
    stellar_values = {'Teff': [], 'logg': [], 'FeH': []}
    
    for i, text in enumerate(text_predictions):
        text_params = extract_stellar_params_from_text(text)
        
        # Only include samples where we have both stellar and text predictions
        include_sample = True
        for param in ['Teff', 'logg', 'FeH']:
            if param not in text_params or i >= len(stellar_medians[param]):
                include_sample = False
                break
        
        if include_sample:
            for param in ['Teff', 'logg', 'FeH']:
                text_extracted[param].append(text_params[param])
                stellar_values[param].append(stellar_medians[param][i])
    
    # Convert to numpy arrays
    for param in ['Teff', 'logg', 'FeH']:
        text_extracted[param] = np.array(text_extracted[param])
        stellar_values[param] = np.array(stellar_values[param])
    
    # Parameter info for labeling
    param_info = {
        'Teff': {'label': 'Effective Temperature (K)', 'range': (3000, 8000)},
        'logg': {'label': 'Surface Gravity (log g)', 'range': (0, 5)},
        'FeH': {'label': 'Metallicity [Fe/H]', 'range': (-3, 1)}
    }
    
    # Create individual plots
    for param in ['Teff', 'logg', 'FeH']:
        if len(text_extracted[param]) == 0:
            print(f"Warning: No extracted {param} values from text, skipping consistency plot")
            continue
            
        stellar_vals = stellar_values[param]
        text_vals = text_extracted[param]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(stellar_vals, text_vals, alpha=0.6, s=20)
        
        # Add diagonal line for perfect consistency
        min_val = min(np.min(stellar_vals), np.min(text_vals))
        max_val = max(np.max(stellar_vals), np.max(text_vals))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Consistency')
        
        plt.xlabel(f'Stellar Prediction - {param_info[param]["label"]}')
        plt.ylabel(f'Text Extracted - {param_info[param]["label"]}')
        plt.title(f'Stellar vs Text Prediction Consistency - {param}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculate consistency metrics
        mae = np.mean(np.abs(stellar_vals - text_vals))
        rmse = np.sqrt(np.mean((stellar_vals - text_vals)**2))
        r_squared = np.corrcoef(stellar_vals, text_vals)[0, 1]**2 if len(stellar_vals) > 1 else 0
        
        # Add metrics to plot
        plt.text(0.05, 0.95, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR^2: {r_squared:.3f}\nN: {len(stellar_vals)}',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{param}_consistency.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Consistency plot saved for {param}")
    
    # Create combined consistency plot
    if all(len(text_extracted[param]) > 0 for param in ['Teff', 'logg', 'FeH']):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, param in enumerate(['Teff', 'logg', 'FeH']):
            stellar_vals = stellar_values[param]
            text_vals = text_extracted[param]
            
            axes[i].scatter(stellar_vals, text_vals, alpha=0.6, s=20)
            
            min_val = min(np.min(stellar_vals), np.min(text_vals))
            max_val = max(np.max(stellar_vals), np.max(text_vals))
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            axes[i].set_xlabel(f'Stellar - {param_info[param]["label"]}')
            axes[i].set_ylabel(f'Text - {param_info[param]["label"]}')
            axes[i].set_title(f'{param} Consistency')
            axes[i].grid(True, alpha=0.3)
            
            # Calculate and display metrics
            mae = np.mean(np.abs(stellar_vals - text_vals))
            rmse = np.sqrt(np.mean((stellar_vals - text_vals)**2))
            r_squared = np.corrcoef(stellar_vals, text_vals)[0, 1]**2 if len(stellar_vals) > 1 else 0
            
            axes[i].text(0.05, 0.95, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR^2: {r_squared:.3f}\nN: {len(stellar_vals)}',
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'stellar_text_consistency_combined.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print("[OK] Combined consistency plot saved")


def llm_predict_stellar_bulk(trainer,
                             data_loader,
                             device,
                             quantiles,
                             max_iter=np.inf,
                             max_samples: Optional[int] = None):
    """
    Bulk prediction method for LLMTrainer to extract stellar parameter predictions
    following the pattern from APOGEE script and MaskedRegressorTrainer
    """
    trainer.model.eval()
    
    # Initialize arrays similar to MaskedRegressorTrainer.predict
    num_params = 3  # Teff, logg, FeH
    num_quantiles = len(quantiles)
    
    preds = np.zeros((0, num_params, num_quantiles))  # [samples, params, quantiles]
    targets = np.zeros((0, num_params))              # [samples, params]
    obsids = []  # Track obsids for verification
    modes = []   # Track sample mode for plotting/analysis
    
    print(f"Running bulk stellar prediction...")
    
    processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if max_samples is not None and processed >= max_samples:
                break
            # Move batch to device
            if isinstance(batch, dict):
                for key in batch:
                    if key in batch and batch[key] is not None and torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(device)
            
            try:
                # Get model outputs
                outputs = trainer.get_logits(batch, device, val=True)
                
                # Extract stellar predictions
                stellar_preds_tensor = None
                if isinstance(outputs, dict) and 'stellar_predictions' in outputs:
                    stellar_preds_tensor = outputs['stellar_predictions']
                elif hasattr(outputs, 'stellar_predictions'):
                    stellar_preds_tensor = outputs.stellar_predictions
                
                if stellar_preds_tensor is not None:
                    batch_size = stellar_preds_tensor.shape[0]
                    take = batch_size
                    if max_samples is not None:
                        take = min(take, max_samples - processed)
                        if take <= 0:
                            break
                    
                    # Reshape to [batch_size, num_params, num_quantiles] 
                    preds_reshaped = stellar_preds_tensor.view(batch_size, num_params, num_quantiles)
                    preds_slice = preds_reshaped[:take].cpu().numpy()
                    
                    # Append to predictions array
                    preds = np.concatenate([preds, preds_slice], axis=0)
                    
                    # Track per-sample modes
                    batch_modes = infer_batch_modes(batch, batch_size)
                    modes.extend(batch_modes[:take])

                    # Extract obsids for this batch
                    batch_obsids = list(batch.get('obsids', []))
                    if len(batch_obsids) < batch_size:
                        batch_obsids.extend([None] * (batch_size - len(batch_obsids)))
                    obsids.extend(batch_obsids[:take])
                
                    # Extract targets following the same logic as training
                    batch_targets = np.full((take, num_params), np.nan)
                    
                    # Single-star mode
                    if 'y_numeric' in batch and batch['y_numeric'] is not None:
                        gt_params = batch['y_numeric'][:take]
                        gt_mask = batch['y_numeric_present'][:take]
                        
                        if gt_mask.any():
                            valid_indices = gt_mask.cpu().numpy()
                            valid_gt = gt_params[gt_mask].cpu().numpy()
                            batch_targets[gt_mask.cpu().numpy()] = valid_gt
                    
                    # Two-star mode (use star A)
                    elif 'y_numeric_a' in batch and batch['y_numeric_a'] is not None:
                        gt_params_a = batch['y_numeric_a'][:take]
                        gt_mask_a = batch['y_numeric_a_present'][:take]
                        
                        if gt_mask_a.any():
                            valid_indices_a = gt_mask_a.cpu().numpy()
                            valid_gt_a = gt_params_a[gt_mask_a].cpu().numpy()
                            batch_targets[valid_indices_a] = valid_gt_a
                    if 'y_numeric_b' in batch and batch['y_numeric_b'] is not None:
                        gt_params_b = batch['y_numeric_b'][:take]
                        gt_mask_b = batch['y_numeric_b_present'][:take]

                        if gt_mask_b.any():
                            valid_indices_b = gt_mask_b.cpu().numpy()
                            valid_gt_b = gt_params_b[gt_mask_b].cpu().numpy()
                            batch_targets[valid_indices_b] = valid_gt_b
                    
                    targets = np.concatenate([targets, batch_targets], axis=0)

                    processed += take
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(data_loader)}")
            if batch_idx > max_iter:
                break
    
    print(f"[OK] Bulk prediction completed: preds shape {preds.shape}, targets shape {targets.shape}, obsids: {len(obsids)}")
    return preds, targets, obsids, modes


def llm_predict_with_text_generation(trainer,
                                   data_loader,
                                   device,
                                   max_iter=np.inf,
                                   generate_text=True,
                                   max_new_tokens=50,
                                   args=None,
                                   max_samples: Optional[int] = None):
    """
    Enhanced prediction that collects both stellar predictions and generated text
    """
    trainer.model.eval()
    
    # Explicitly load tokenizer using get_model_path
    tokenizer = None
    if generate_text and args is not None:
        try:
            _, tokenizer_path = get_model_path(args)
            print("trying to loaf tokenizer from: ", tokenizer_path)
            tokenizer = Tokenizer(model_path=tokenizer_path)
            print(f"[OK] Loaded tokenizer from {tokenizer_path}")
        except Exception as e:
            print(f"Warning: Failed to load tokenizer: {e}")
            generate_text = False
            exit()
    elif generate_text:
        print("Warning: No args provided for tokenizer loading")
        generate_text = False
    
    # Collect predictions
    all_text_predictions = []
    all_text_targets = []
    all_questions = []
    all_modes = []
    text_obsids = []  # Track obsids for verification
    
    print(f"Running predictions with text generation for {len(data_loader)} interations...")
    print("max samples: ", max_samples)
    
    processed = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(data_loader))
        for batch_idx, batch in pbar:
            if max_samples is not None and processed >= max_samples:
                break
            # Move batch to device
            if isinstance(batch, dict):
                for key in batch:
                    if key in batch and batch[key] is not None and torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(device)
            
            batch_size = batch['input_ids'].shape[0] if 'input_ids' in batch else 1
            
            # Extract obsids for this batch
            batch_obsids = batch.get('obsids', [])
            # Infer per-sample modes ahead of generation
            batch_modes = infer_batch_modes(batch, batch_size)
            
            # Generate text for each sample in batch
            limit = 20*batch_size if max_samples is None else min(20*batch_size, max_samples - processed)
            # print("limit: ", limit)
            if limit <= 0:
                break

            if generate_text:
                debug_modes = batch.get('mode')
                if batch_idx < 3:
                    if debug_modes is not None:
                        print('batch modes:', debug_modes)
                    else:
                        print('batch modes (inferred):', batch_modes)
                model_for_generation = trainer.model
                if isinstance(model_for_generation, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                    model_for_generation = model_for_generation.module
                generation_kwargs = {
                    'tokenizer': tokenizer,
                    'max_new_tokens': max_new_tokens,
                    'temperature': 0.2,
                    'top_p': 0.8,
                }
                for sample_idx in range(batch_size):
                    # print("sample idx: ", sample_idx)
                    try:
                        # Generate response for this sample
                        generated_text, input_text, target_text, _ = model_for_generation.generate_response_from_batch(
                            batch_data=batch,
                            batch_idx=sample_idx,
                            **generation_kwargs,
                        )
                        # print("input text: ", input_text)
                        # print("target text: ", target_text)
                        # print("gen text: ", generated_text)
                        all_text_predictions.append(generated_text)
                        all_text_targets.append(target_text)
                        all_questions.append(input_text)
                        all_modes.append(batch_modes[sample_idx] if sample_idx < len(batch_modes) else 'single_star')
                        
                        # Add corresponding obsid
                        if sample_idx < len(batch_obsids):
                            text_obsids.append(batch_obsids[sample_idx])
                        else:
                            text_obsids.append(None)
                        
                    except Exception as e:
                        print(f"Error generating text for sample {sample_idx} in batch {batch_idx}: {e}")
                        all_text_predictions.append("")
                        all_text_targets.append("")
                        all_questions.append("")
                        all_modes.append(batch_modes[sample_idx] if sample_idx < len(batch_modes) else 'single_star')
                        
                        # Add corresponding obsid even for failed cases
                        if sample_idx < len(batch_obsids):
                            text_obsids.append(batch_obsids[sample_idx])
                        else:
                            text_obsids.append(None)
            
                processed += batch_size

            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(data_loader)} ({processed} samples)")
                
            if batch_idx > max_iter:
                print("braking !! batch idx: ", batch_idx, "max iter: ", max_iter)
                break
    
    print(f"[OK] Text generation completed: {len(all_text_predictions)} samples, obsids: {len(text_obsids)}")
    return all_text_predictions, all_text_targets, text_obsids, all_questions, all_modes


def _split_prediction_array(pred_array: np.ndarray) -> Dict[str, np.ndarray]:
    """Slice a (N, num_params, num_quantiles) array into a dict keyed by parameter."""
    if pred_array is None:
        return {}
    pred_array = np.asarray(pred_array)
    if pred_array.size == 0:
        return {}
    if pred_array.ndim == 2:
        pred_array = pred_array[:, :, np.newaxis]
    if pred_array.ndim != 3:
        raise ValueError(f"Expected predictions array with 3 dimensions, got shape {pred_array.shape}")
    param_order = ['Teff', 'logg', 'FeH']
    result: Dict[str, np.ndarray] = {}
    for idx, param in enumerate(param_order):
        if idx < pred_array.shape[1]:
            result[param] = pred_array[:, idx, :].astype(np.float32)
    return result


def _split_target_array(target_array: np.ndarray) -> Dict[str, np.ndarray]:
    """Slice a (N, num_params) array into a dict keyed by parameter."""
    if target_array is None:
        return {}
    target_array = np.asarray(target_array)
    if target_array.size == 0:
        return {}
    if target_array.ndim == 1:
        target_array = target_array.reshape(-1, 1)
    if target_array.ndim != 2:
        raise ValueError(f"Expected targets array with 2 dimensions, got shape {target_array.shape}")
    param_order = ['Teff', 'logg', 'FeH']
    result: Dict[str, np.ndarray] = {}
    for idx, param in enumerate(param_order):
        if idx < target_array.shape[1]:
            result[param] = target_array[:, idx].astype(np.float32)
    return result


def _unnormalize_param_dict(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert normalized parameter arrays back to physical units using dataset bounds."""
    if not data:
        return {}
    return unnormalize_stellar_params(data)


def save_question_answer_triples(questions: List[str],
                                text_targets: List[str], 
                                text_predictions: List[str],
                                text_obsids: List,
                                text_modes: List[str],
                                output_dir: str,
                                max_samples: int = 100) -> None:
    """
    Save question-answer triples as JSON file for evaluation analysis.
    Uses a representative sample to avoid excessive file size.
    """
    import json
    import os
    from datetime import datetime
    
    if not questions or not text_targets or not text_predictions:
        print("Warning: No question-answer data to save")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use representative sample
    n_samples = min(len(questions), max_samples)
    
    # Create structured data
    qa_triples = []
    for i in range(n_samples):
        triple = {
            'sample_id': i,
            'obsid': text_obsids[i] if i < len(text_obsids) else None,
            'mode': text_modes[i] if i < len(text_modes) else 'unknown',
            'question': questions[i],
            'true_answer': text_targets[i],
            'generated_answer': text_predictions[i]
        }
        qa_triples.append(triple)
    
    # Create metadata
    backend_metadata = None
    if backend_config is not None:
        backend_metadata = backend_config.to_dict() if hasattr(backend_config, "to_dict") else backend_config

    metadata = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(questions),
        'saved_samples': n_samples,
        'sampling_note': f'Representative sample of {n_samples} out of {len(questions)} total samples'
    }
    if backend_metadata is not None:
        metadata['backend'] = backend_metadata
    
    # Combine data and metadata
    output_data = {
        'metadata': metadata,
        'question_answer_triples': qa_triples
    }
    
    # Save to JSON file
    output_path = os.path.join(output_dir, 'question_answer_triples.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Question-answer triples saved to {output_path}")
    print(f"  Saved {n_samples} representative samples out of {len(questions)} total")


def run_prediction_evaluation(trainer: LLMTrainer,
                              test_loader,
                              device: torch.device,
                              args,
                              inference_args,
                              quantiles: List[float],
                              plots_dir: str,
                              backend_config = None) -> None:
    """Generate stellar predictions, text generations, consistency plots, and save outputs."""
    print("Running text generation for consistency analysis...")
    max_samples = getattr(inference_args, 'consistency_max_samples', None)
    if max_samples is not None and max_samples <= 0:
        max_samples = None

    text_predictions, text_targets, text_obsids, questions, text_modes = llm_predict_with_text_generation(
        trainer=trainer,
        data_loader=test_loader,
        device=device,
        max_iter=1,  # TODO not hardcoded
        generate_text=True,
        max_new_tokens=512,
        args=args,
        max_samples=max_samples,
    )

    # Save question-answer triples as JSON for analysis
    qa_output_dir = os.path.join(plots_dir, 'question_answer_data')
    save_question_answer_triples(
        questions=questions,
        text_targets=text_targets,
        text_predictions=text_predictions,
        text_obsids=text_obsids,
        text_modes=text_modes,
        output_dir=qa_output_dir,
        max_samples=100  # Representative sample size
    )

    text_param_values: Dict[str, List[float]] = {'Teff': [], 'logg': [], 'FeH': []}
    for text in text_predictions:
        extracted = extract_stellar_params_from_text(text or '')
        for param in ['Teff', 'logg', 'FeH']:
            value = extracted.get(param)
            if value is None:
                numeric_value = float('nan')
            else:
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    numeric_value = float('nan')
            if math.isnan(numeric_value):
                text_param_values[param].append(float('nan'))
            else:
                text_param_values[param].append(numeric_value)

    print("Collecting stellar parameter predictions for evaluation...")
    stellar_preds_np, stellar_targets_np, stellar_obsids, stellar_modes = llm_predict_stellar_bulk(
        trainer=trainer,
        data_loader=test_loader,
        device=device,
        quantiles=quantiles,
        max_iter=np.inf,
        max_samples=max_samples,
    )

    stellar_preds_norm = _split_prediction_array(stellar_preds_np)
    stellar_targets_norm = _split_target_array(stellar_targets_np)

    stellar_preds_physical = _unnormalize_param_dict(stellar_preds_norm)
    stellar_targets_physical = _unnormalize_param_dict(stellar_targets_norm)

    # Align text predictions with available targets (truncate to shared length)
    text_preds_physical: Dict[str, np.ndarray] = {}
    if text_param_values and stellar_targets_physical:
        target_length = next(iter(stellar_targets_physical.values())).shape[0] if stellar_targets_physical else 0
        text_length = len(text_predictions)
        effective_length = min(target_length, text_length)
        if effective_length > 0:
            for param, values in text_param_values.items():
                array = np.asarray(values[:effective_length], dtype=float)
                text_preds_physical[param] = array
            text_modes = text_modes[:effective_length] if text_modes else []
        else:
            text_preds_physical = {}

    # if inference_args.save_predictions:
    #     predictions_path = os.path.join(inference_args.output_dir, 'test_predictions.json')
    #     save_predictions(
    #         text_predictions=text_predictions,
    #         text_targets=text_targets,
    #         questions=questions,
    #         text_modes=text_modes,
    #         stellar_preds_quantiles=stellar_preds_physical,
    #         stellar_targets=stellar_targets_physical,
    #         stellar_modes=stellar_modes,
    #         output_path=predictions_path,
    #         backend_metadata=backend_metadata,
    #     )

    if inference_args.plot_results:
        stellar_plot_dir = os.path.join(plots_dir, 'stellar')
        
        # Only plot numerical predictions if they exist
        if stellar_preds_physical and any(len(v) > 0 for v in stellar_preds_physical.values()):
            plot_stellar_parameters(
                stellar_preds_quantiles=stellar_preds_physical,
                stellar_targets=stellar_targets_physical,
                output_dir=stellar_plot_dir,
                modes=stellar_modes,
                method_label='Numeric Model',
            )
        else:
            print("Skipping numerical stellar parameter plots (no numerical predictions available)")

        if text_preds_physical:
            plot_scalar_stellar_parameters(
                stellar_preds=text_preds_physical,
                stellar_targets=stellar_targets_physical,
                output_dir=stellar_plot_dir,
                modes=text_modes,
                method_label='Text Extracted',
            )
        else:
            print("Skipping text paramters plot")

        consistency_dir = os.path.join(plots_dir, 'consistency_plots')
        # Only plot consistency if we have numerical predictions
        if stellar_preds_physical and any(len(v) > 0 for v in stellar_preds_physical.values()):
            plot_stellar_vs_text_consistency(
                stellar_preds_quantiles=stellar_preds_physical,
                text_predictions=text_predictions,
                output_dir=consistency_dir,
            )
        else:
            print("Skipping consistency plots (no numerical predictions available)")


def parse_inference_args():
    """Parse command line arguments for inference"""
    parser = argparse.ArgumentParser(description='Inference script for stellar model')
    
    # Model and data arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the saved model checkpoint')
    parser.add_argument('--config_path', type=str, 
                        help='Path to training config file (auto-detected from checkpoint dir if not provided)')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save inference results')
    parser.add_argument('--llm_backend', type=str, choices=['llama', 'hf', 'qwen'], default='llama',
                        help='Backbone family to use for inference.')
    parser.add_argument('--llm_root', type=str, default=os.environ.get('LLM_ROOT', '/home/ilay.kamai/work/.llama'),
                       help='Root directory containing LLaMA models (or set env LLM_ROOT)')
    parser.add_argument('--llm_model', type=str, default='Llama3.1-8B',
                        help='LLaMA model name relative to --llm_root.')
    parser.add_argument('--llm_path', type=str, default=None,
                        help='Explicit path to a LLaMA checkpoint directory (overrides --llm_root/--llm_model).')
    parser.add_argument('--hf_model_name', type=str, default='Qwen/Qwen2.5-32B-Instruct',
                        help='HF repo or local path when --llm_backend=hf/qwen.')
    parser.add_argument('--hf_revision', type=str, default=None,
                        help='Specific HF revision/commit.')
    parser.add_argument('--hf_trust_remote_code', action='store_true', default=False,
                        help='Allow execution of custom HF modeling code.')
    parser.add_argument('--hf_device_map', type=str, default=None,
                        help='Device map string/dict for HF AutoModel (e.g., "auto").')
    parser.add_argument('--hf_quantization', type=str, choices=['none', '8bit', '4bit'],
                        default=os.environ.get('HF_QUANTIZATION', 'none'),
                        help='Quantization strategy for HF backbone.')
    parser.add_argument('--hf_cache_dir', type=str,
                        default=os.environ.get('HF_CACHE_DIR', os.environ.get('HF_HOME')),
                        help='HF cache directory override.')
    parser.add_argument('--hf_max_memory_gb', type=float, default=None,
                        help='Soft per-device memory cap for HF loader.')
    parser.add_argument('--hf_auth_token', type=str, default=os.environ.get('HF_TOKEN'),
                        help='Optional HF auth token.')
    parser.add_argument('--llm_precision', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp16',
                        help='Precision override for Meta backbone.')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False,
                        help='Enable gradient checkpointing when supported (LLaMA backend).')
    
    # Data arguments (can be overridden from config)
    parser.add_argument('--json_file', type=str, 
                        default='/home/ilay.kamai/work/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json',
                        help='Path to main dataset JSON file')
    parser.add_argument('--comparative_json_file', type=str,
                        default='/home/ilay.kamai/work/TalkingLatents/data/dataset/comparative_dataset.json',
                        help='Path to comparative dataset JSON file')
    parser.add_argument('--features_file', type=str,
                        default='/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/features.npy',
                        help='Path to pre-computed spectral features')
    
    # Inference settings
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on')
    parser.add_argument('--save_predictions', action='store_true', default=True,
                        help='Save predictions to JSON file')
    parser.add_argument('--plot_results', action='store_true', default=True,
                        help='Generate plots of true vs predicted values')
    parser.add_argument('--interpolation_pairs', type=str, nargs='*', default=None,
                        help="Pairs of dataset indices to interpolate, format 'idx_a:idx_b'")
    parser.add_argument('--interpolation_alphas', type=float, nargs='*', default=None,
                        help='Interpolation alphas between 0 and 1 (default: 10-point grid from 0.0 to 1.0)')
    parser.add_argument('--interpolation_min_teff_diff', type=float, default=400.0,
                        help='Minimum Teff difference in Kelvin required for interpolation pairs')
    parser.add_argument('--interpolation_min_logg_diff', type=float, default=0.3,
                        help='Minimum log g difference required for interpolation pairs')
    parser.add_argument('--consistency_max_samples', type=int, default=512,
                        help='Maximum number of samples to use for consistency plots (<=0 uses entire dataset)')

    parser.add_argument('--interpolation_num_pairs', type=int, default=0,
                        help='Randomly choose this many interpolation pairs if explicit pairs are not provided')
    parser.add_argument('--interpolation_seed', type=int, default=42,
                        help='Random seed for interpolation pair selection')

    parser.add_argument('--comparison_checkpoint_paths', type=str, nargs='*', default=None,
                        help='Additional checkpoint paths to compare during interpolation analysis')
    parser.add_argument('--comparison_labels', type=str, nargs='*', default=None,
                        help='Optional labels corresponding to comparison checkpoints')
    parser.add_argument('--baseline_mlp_checkpoint_path', type=str, default=None,
                        help='Path to the baseline MLP checkpoint used for interpolation comparison')
    parser.add_argument('--baseline_mlp_label', type=str, default='Baseline MLP',
                        help='Display label for the baseline MLP in comparison plots')
    parser.add_argument('--predict_features', action='store_true', default=False,
                        help='Run feature prediction inference and generate UMAP visualizations')
    parser.add_argument('--predict_stellar_params', action='store_true', default=True,
                        help='Enable stellar parameter prediction during inference')
    parser.add_argument('--use_cfm', action='store_true', default=False,
                        help='Use Conditional Flow Matching')
    parser.add_argument('--cfm_weight', type=float, default=0.1,
                        help='Weight for CFM loss')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.159, 0.5, 0.841],
                        help='Quantiles for CQR stellar parameter prediction (default: ~1-sigma + median)')
    parser.add_argument('--feature_dim', type=int, default=2048,
                        help='Dimension of features to predict (default: 2048)')
    parser.add_argument('--feature_loss_weight', type=float, default=1.0,
                        help='Weight for feature prediction loss')
    parser.add_argument('--max_iter', type=int, default=np.inf,
                        help='maximum test interatiob')

    return parser.parse_args()


def derive_model_label(checkpoint_path: str) -> str:
    """Create a readable label from a checkpoint path."""
    trimmed_path = checkpoint_path.rstrip(os.sep)
    path = Path(trimmed_path if trimmed_path else checkpoint_path)

    if path.suffix:
        candidate = path.parent.name
        if not candidate:
            candidate = path.stem or path.name
    else:
        candidate = path.name or str(path)

    return candidate or str(path)

def sanitize_label(label: str) -> str:
    sanitized = re.sub(r'[^A-Za-z0-9._-]+', '_', label.strip())
    sanitized = sanitized.strip('_')
    return sanitized or 'model'



def load_config_from_checkpoint_dir(checkpoint_path: str) -> Dict:
    """Load training config from checkpoint directory"""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, 'training_config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"No training config found at {config_path}")


def create_args_from_config(config: Dict, inference_args: argparse.Namespace) -> argparse.Namespace:
    """Create args namespace from training config, overriding with inference args."""
    
    # Start with inference_args as the base
    args = argparse.Namespace(**vars(inference_args))
    
    # Update with all config values (config takes precedence for any overlaps)
    for key, value in config.items():
        setattr(args, key, value)
    
    return args


def _extract_arch_signature(args: argparse.Namespace) -> Tuple:
    """Create a lightweight signature that captures architecture-defining args."""
    fields = [
        'spectral_embedding_dim',
        'hidden_dim',
        'num_spectral_features',
        'use_cfm',
        'cfm_weight',
        'enable_classification',
        'llm_precision',
        'gradient_checkpointing',
        'features_file',
        'llm_root',
        'llm_model',
        'llm_path',
    ]
    signature: List[Any] = []
    for field in fields:
        signature.append(getattr(args, field, None))
    quantiles = getattr(args, 'quantiles', None)
    if quantiles is None:
        signature.append(tuple())
    else:
        signature.append(tuple(float(q) for q in quantiles))
    return tuple(signature)


def load_model(checkpoint_path: str,
               args: argparse.Namespace,
               device: Union[int, torch.device],
               model: Optional[torch.nn.Module] = None) -> torch.nn.Module:
    """Load checkpoint weights into a model, optionally reusing an existing instance."""
    
    print(f"Loading model from {checkpoint_path}")

    if not hasattr(args, 'llm_precision'):
        args.llm_precision = 'fp16'
    if not hasattr(args, 'gradient_checkpointing'):
        args.gradient_checkpointing = False
    backend_config = ensure_backend_config(args)
    
    # Convert device to torch.device if it's an int (local rank)
    if isinstance(device, int):
        device = torch.device(f'cuda:{device}')
    
    reuse_note = ""
    if model is None:
        # Build the model architecture
        model = build_model_multitok(args, device, world_size=1, backend_config=backend_config)
    else:
        reuse_note = " (reusing cached architecture)"
        model.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    del checkpoint
    
    # Handle DataParallel/DistributedDataParallel prefixes
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value

    # Load state dict
    target_model = model.module if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model
    missing_keys, unexpected_keys = target_model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys when loading {checkpoint_path}: {missing_keys[:5]}{' ...' if len(missing_keys) > 5 else ''}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys when loading {checkpoint_path}: {unexpected_keys[:5]}{' ...' if len(unexpected_keys) > 5 else ''}")
    del new_state_dict, state_dict
    model.eval()

    model.to(device)
    print(f"[OK] Model loaded successfully{reuse_note}")
    return model


def load_checkpoint_weights_into_model(model: torch.nn.Module,
                                       checkpoint_path: str,
                                       device: Union[int, torch.device]) -> None:
    """Load a checkpoint's weights into an existing model instance."""
    if model is None:
        raise ValueError("load_checkpoint_weights_into_model received a null model.")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    del checkpoint

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith('module.') else key
        new_state_dict[new_key] = value

    target_model = model.module if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model
    missing_keys, unexpected_keys = target_model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys when loading {checkpoint_path}: {missing_keys[:5]}{' ...' if len(missing_keys) > 5 else ''}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys when loading {checkpoint_path}: {unexpected_keys[:5]}{' ...' if len(unexpected_keys) > 5 else ''}")

    del new_state_dict, state_dict
    model.to(device)
    model.to(device)




def save_predictions(text_predictions: List[str],
                     text_targets: List[str],
                     questions: List[str],
                     text_modes: List[str],
                     stellar_preds_quantiles: Dict,
                     stellar_targets: Dict,
                     stellar_modes: List[str],
                     output_path: str,
                     backend_metadata: Optional[Dict[str, Any]] = None):
    """Save predictions to JSON file"""
    max_text_items = max(len(questions), len(text_predictions), len(text_targets), len(text_modes))
    text_results: Dict[str, Dict[str, Any]] = {}
    for idx in range(max_text_items):
        entry: Dict[str, Any] = {
            'question': questions[idx] if idx < len(questions) else '',
            'generated_answer': text_predictions[idx] if idx < len(text_predictions) else '',
            'true_answer': text_targets[idx] if idx < len(text_targets) else '',
        }
        if idx < len(text_modes):
            entry['mode'] = text_modes[idx]
        text_results[str(idx)] = entry

    stellar_results = {
        'predictions_quantiles': {},
        'predictions_median': {},
        'targets': {},
        'modes': [normalize_mode_label(m) for m in stellar_modes] if stellar_modes else [],
    }

    if stellar_preds_quantiles:
        for param, values in stellar_preds_quantiles.items():
            stellar_results['predictions_quantiles'][param] = values.tolist()
            median_idx = values.shape[1] // 2
            stellar_results['predictions_median'][param] = values[:, median_idx].tolist()

    if stellar_targets:
        for param, values in stellar_targets.items():
            stellar_results['targets'][param] = values.tolist()

    results = {
        'text_samples': text_results,
        'stellar': stellar_results,
        'metadata': {
            'num_text_samples': max_text_items,
            'num_stellar_samples': next(iter(stellar_preds_quantiles.values())).shape[0] if stellar_preds_quantiles else 0,
            'stellar_parameters': list(stellar_preds_quantiles.keys()) if stellar_preds_quantiles else [],
            'num_quantiles': stellar_preds_quantiles[list(stellar_preds_quantiles.keys())[0]].shape[1] if stellar_preds_quantiles else 0,
        }
    }
    if backend_metadata is not None:
        results['metadata']['backend'] = backend_metadata

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Predictions saved to {output_path}")
    if max_text_items:
        print(f"  - {max_text_items} text entries saved")
    if stellar_preds_quantiles:
        num_quantiles = results['metadata']['num_quantiles']
        print(f"  - {results['metadata']['num_stellar_samples']} stellar parameter predictions with {num_quantiles} quantiles each")

def plot_stellar_parameters(stellar_preds_quantiles: Dict,
                            stellar_targets: Dict,
                            output_dir: str,
                            modes: Optional[List[str]] = None,
                            method_label: str = 'Numeric Model'):
    """Plot true vs predicted stellar parameters with confidence intervals, color-coded by mode."""
    
    if not stellar_preds_quantiles or not stellar_targets:
        print("No stellar parameters to plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)

    method_slug = sanitize_label(method_label) or 'method'
    
    # Parameter labels and units
    param_info = {
        'Teff': {'label': 'Effective Temperature (K)', 'range': (3000, 8000)},
        'logg': {'label': 'Surface Gravity (log g)', 'range': (0, 5)},
        'FeH': {'label': 'Metallicity [Fe/H]', 'range': (-3, 1)}
    }

    mode_array = None
    if modes is not None and isinstance(stellar_preds_quantiles, dict) and stellar_preds_quantiles:
        sample_count = next(iter(stellar_preds_quantiles.values())).shape[0]
        normalized_modes = [normalize_mode_label(m) for m in modes]
        if len(normalized_modes) < sample_count:
            print(f"Warning: modes length ({len(normalized_modes)}) is shorter than sample count ({sample_count}); skipping mode coloring.")
        else:
            if len(normalized_modes) > sample_count:
                print(f"Warning: modes length ({len(normalized_modes)}) exceeds sample count ({sample_count}); trimming extras.")
            mode_array = np.asarray(normalized_modes[:sample_count], dtype=object)
    
    if isinstance(stellar_preds_quantiles, dict) and isinstance(stellar_targets, dict):
        # Plot each parameter separately
        for param in stellar_preds_quantiles.keys():
            if param not in stellar_targets:
                continue
                
            preds_quantiles = stellar_preds_quantiles[param]  # [num_samples, num_quantiles]
            targets = stellar_targets[param]  # [num_samples]
            
            # Extract median and confidence intervals
            median_idx = preds_quantiles.shape[1] // 2
            median_preds = preds_quantiles[:, median_idx]
            
            # Use 10th and 90th percentiles for confidence interval (assuming quantiles are evenly spaced)
            num_quantiles = preds_quantiles.shape[1]
            lower_idx = max(0, int(0.1 * num_quantiles))
            upper_idx = min(num_quantiles - 1, int(0.9 * num_quantiles))
            lower_preds = preds_quantiles[:, lower_idx]
            upper_preds = preds_quantiles[:, upper_idx]
            
            # Filter out NaN values for plotting and metrics
            valid_mask = ~(np.isnan(targets) | np.isnan(median_preds))
            if not valid_mask.any():
                print(f"Warning: No valid data for parameter {param}, skipping plot")
                continue
                
            valid_targets = targets[valid_mask]
            valid_median_preds = median_preds[valid_mask]
            valid_lower_preds = lower_preds[valid_mask]
            valid_upper_preds = upper_preds[valid_mask]
            valid_modes = mode_array[valid_mask] if mode_array is not None else None
            
            plt.figure(figsize=(10, 8))
            
            # Sort by targets for better confidence interval visualization
            sort_idx = np.argsort(valid_targets)
            sorted_targets = valid_targets[sort_idx]
            sorted_median = valid_median_preds[sort_idx]
            sorted_lower = valid_lower_preds[sort_idx]
            sorted_upper = valid_upper_preds[sort_idx]
            
            # Plot confidence interval
            interval_handle = plt.fill_between(sorted_targets, sorted_lower, sorted_upper, alpha=0.3, 
                                               label='1-sigma Confidence Interval', color='lightblue')
            
            # Add diagonal line for perfect prediction
            min_val = min(np.min(valid_targets), np.min(valid_median_preds))
            max_val = max(np.max(valid_targets), np.max(valid_median_preds))
            line_handle, = plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

            scatter_handles = []
            scatter_labels = []
            
            if valid_modes is not None and len(valid_modes) == len(valid_targets):
                for mode_name in ['single_star', 'two_star']:
                    mode_mask = (valid_modes == mode_name)
                    if mode_mask.any():
                        scatter = plt.scatter(
                            valid_targets[mode_mask],
                            valid_median_preds[mode_mask],
                            alpha=0.7,
                            s=25,
                            color=MODE_COLOR_MAP.get(mode_name, 'gray'),
                        )
                        scatter_handles.append(scatter)
                        scatter_labels.append(MODE_DISPLAY_NAMES.get(mode_name, mode_name.replace('_', ' ').title()))
            else:
                scatter = plt.scatter(valid_targets, valid_median_preds, alpha=0.7, s=25,
                                      color=MODE_COLOR_MAP.get('single_star', 'tab:blue'))
                scatter_handles = [scatter]
                scatter_labels = ['Median Prediction']
            
            plt.xlabel(f'True {param_info.get(param, {}).get("label", param)}')
            plt.ylabel(f'Predicted {param_info.get(param, {}).get("label", param)}')
            plt.title(f'True vs Predicted {param} ({method_label}, CQR Calibrated)')
            handles = [interval_handle, line_handle] + scatter_handles
            labels = ['1-sigma Confidence Interval', 'Perfect Prediction'] + scatter_labels
            plt.legend(handles, labels)
            plt.grid(True, alpha=0.3)
            
            # Calculate metrics on valid data only
            mae = np.mean(np.abs(valid_median_preds - valid_targets))
            rmse = np.sqrt(np.mean((valid_median_preds - valid_targets)**2))
            r_squared = np.corrcoef(valid_targets, valid_median_preds)[0, 1]**2
            
            # Calculate coverage (what fraction of targets fall within confidence interval)
            coverage = np.mean((valid_targets >= valid_lower_preds) & (valid_targets <= valid_upper_preds))
            
            # Add metrics to plot
            plt.text(0.05, 0.95, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR^2: {r_squared:.3f}\nCoverage: {coverage:.1%}\nN: {len(valid_targets)}',
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            plt.tight_layout()
            filename = f'{param}_true_vs_predicted_cqr_calibrated_{method_slug}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[OK] Plot with CQR calibrated confidence intervals saved for {param} ({method_label}) in {output_dir}")
    
    # Create combined plot with confidence intervals
    if isinstance(stellar_preds_quantiles, dict) and len(stellar_preds_quantiles) == 3:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for i, param in enumerate(['Teff', 'logg', 'FeH']):
            if param not in stellar_preds_quantiles or param not in stellar_targets:
                continue
                
            preds_quantiles = stellar_preds_quantiles[param]
            targets = stellar_targets[param]
            
            # Extract median and confidence intervals
            median_idx = preds_quantiles.shape[1] // 2
            median_preds = preds_quantiles[:, median_idx]
            
            num_quantiles = preds_quantiles.shape[1]
            lower_idx = max(0, int(0.1 * num_quantiles))
            upper_idx = min(num_quantiles - 1, int(0.9 * num_quantiles))
            lower_preds = preds_quantiles[:, lower_idx]
            upper_preds = preds_quantiles[:, upper_idx]
            
            # Filter out NaN values
            valid_mask = ~(np.isnan(targets) | np.isnan(median_preds))
            if not valid_mask.any():
                print(f"Warning: No valid data for parameter {param}, skipping from combined plot")
                continue
            
            valid_targets = targets[valid_mask]
            valid_median_preds = median_preds[valid_mask]
            valid_lower_preds = lower_preds[valid_mask]
            valid_upper_preds = upper_preds[valid_mask]
            valid_modes = mode_array[valid_mask] if mode_array is not None else None
            
            # Sort for better visualization
            sort_idx = np.argsort(valid_targets)
            sorted_targets = valid_targets[sort_idx]
            sorted_lower = valid_lower_preds[sort_idx]
            sorted_upper = valid_upper_preds[sort_idx]
            
            # Plot confidence interval
            interval_handle = axes[i].fill_between(sorted_targets, sorted_lower, sorted_upper, alpha=0.3, color='lightblue')
            
            min_val = min(np.min(valid_targets), np.min(valid_median_preds))
            max_val = max(np.max(valid_targets), np.max(valid_median_preds))
            line_handle, = axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

            scatter_handles = []
            scatter_labels = []
            if valid_modes is not None and len(valid_modes) == len(valid_targets):
                for mode_name in ['single_star', 'two_star']:
                    mode_mask = (valid_modes == mode_name)
                    if mode_mask.any():
                        scatter = axes[i].scatter(
                            valid_targets[mode_mask],
                            valid_median_preds[mode_mask],
                            alpha=0.7,
                            s=20,
                            color=MODE_COLOR_MAP.get(mode_name, 'gray'),
                        )
                        scatter_handles.append(scatter)
                        scatter_labels.append(MODE_DISPLAY_NAMES.get(mode_name, mode_name.replace('_', ' ').title()))
            else:
                scatter = axes[i].scatter(valid_targets, valid_median_preds, alpha=0.7, s=20,
                                          color=MODE_COLOR_MAP.get('single_star', 'tab:blue'))
                scatter_handles = [scatter]
                scatter_labels = ['Median Prediction']
            
            axes[i].set_xlabel(f'True {param_info.get(param, {}).get("label", param)}')
            axes[i].set_ylabel(f'Predicted {param_info.get(param, {}).get("label", param)}')
            axes[i].set_title(f'{param} ({method_label})')
            axes[i].grid(True, alpha=0.3)
            
            # Calculate and display metrics
            mae = np.mean(np.abs(valid_median_preds - valid_targets))
            rmse = np.sqrt(np.mean((valid_median_preds - valid_targets)**2))
            r_squared = np.corrcoef(valid_targets, valid_median_preds)[0, 1]**2
            coverage = np.mean((valid_targets >= valid_lower_preds) & (valid_targets <= valid_upper_preds))
            
            axes[i].text(0.05, 0.95, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR^2: {r_squared:.3f}\nCov: {coverage:.1%}\nN: {len(valid_targets)}',
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            handles = [interval_handle, line_handle] + scatter_handles
            labels = ['1-sigma Confidence Interval', 'Perfect Prediction'] + scatter_labels
            axes[i].legend(handles, labels)
        
        plt.tight_layout()
        combined_filename = f'stellar_parameters_combined_cqr_calibrated_{method_slug}.png'
        plt.savefig(os.path.join(output_dir, combined_filename), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Combined plot with CQR calibrated confidence intervals saved ({method_label})")


def plot_scalar_stellar_parameters(stellar_preds: Dict[str, np.ndarray],
                                   stellar_targets: Dict[str, np.ndarray],
                                   output_dir: str,
                                   modes: Optional[List[str]] = None,
                                   method_label: str = 'Text Extracted') -> None:
    """Plot true vs predicted stellar parameters for single-value predictions (e.g., text extractions)."""
    if not stellar_preds or not stellar_targets:
        print("No scalar stellar parameters to plot")
        return

    os.makedirs(output_dir, exist_ok=True)
    method_slug = sanitize_label(method_label) or 'method'

    param_info = {
        'Teff': {'label': 'Effective Temperature (K)', 'range': (3000, 8000)},
        'logg': {'label': 'Surface Gravity (log g)', 'range': (0, 5)},
        'FeH': {'label': 'Metallicity [Fe/H]', 'range': (-3, 1)}
    }

    sample_count = min(
        min((len(values) for values in stellar_preds.values()), default=0),
        min((len(values) for values in stellar_targets.values()), default=0),
    )
    if sample_count == 0:
        print("Warning: No overlapping samples for scalar stellar plots")
        return

    mode_array = None
    if modes is not None and len(modes) >= sample_count:
        normalized_modes = [normalize_mode_label(m) for m in modes[:sample_count]]
        mode_array = np.asarray(normalized_modes, dtype=object)

    for param, preds_values in stellar_preds.items():
        if param not in stellar_targets:
            continue
        preds = np.asarray(preds_values[:sample_count], dtype=float)
        targets = np.asarray(stellar_targets[param][:sample_count], dtype=float)

        valid_mask = ~(np.isnan(preds) | np.isnan(targets))
        if not valid_mask.any():
            print(f"Warning: No valid scalar data for parameter {param}, skipping plot")
            continue

        valid_preds = preds[valid_mask]
        valid_targets = targets[valid_mask]
        valid_modes = mode_array[valid_mask] if mode_array is not None else None

        plt.figure(figsize=(10, 8))
        diag_min = min(np.min(valid_targets), np.min(valid_preds))
        diag_max = max(np.max(valid_targets), np.max(valid_preds))
        plt.plot([diag_min, diag_max], [diag_min, diag_max], 'r--', linewidth=2, label='Perfect Prediction')

        if valid_modes is not None:
            for mode_name in ['single_star', 'two_star']:
                mode_mask = (valid_modes == mode_name)
                if mode_mask.any():
                    plt.scatter(
                        valid_targets[mode_mask],
                        valid_preds[mode_mask],
                        alpha=0.7,
                        s=25,
                        color=MODE_COLOR_MAP.get(mode_name, 'gray'),
                        label=MODE_DISPLAY_NAMES.get(mode_name, mode_name.replace('_', ' ').title()),
                    )
        else:
            plt.scatter(valid_targets, valid_preds, alpha=0.7, s=25,
                        color=MODE_COLOR_MAP.get('single_star', 'tab:blue'),
                        label='Predicted')

        plt.xlabel(f'True {param_info.get(param, {}).get("label", param)}')
        plt.ylabel(f'Predicted {param_info.get(param, {}).get("label", param)}')
        plt.title(f'True vs Predicted {param} ({method_label})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        mae = np.mean(np.abs(valid_preds - valid_targets))
        rmse = np.sqrt(np.mean((valid_preds - valid_targets) ** 2))
        r_squared = np.corrcoef(valid_targets, valid_preds)[0, 1]**2 if len(valid_targets) > 1 else float('nan')

        plt.text(
            0.05,
            0.95,
            f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR^2: {r_squared:.3f}\nN: {len(valid_targets)}',
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
        )

        plt.tight_layout()
        filename = f'{param}_true_vs_predicted_{method_slug}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Scalar plot saved for {param} ({method_label})")

    # Combined plot across parameters if all available
    params = ['Teff', 'logg', 'FeH']
    if all(param in stellar_preds and param in stellar_targets for param in params):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for idx, param in enumerate(params):
            preds = np.asarray(stellar_preds[param][:sample_count], dtype=float)
            targets = np.asarray(stellar_targets[param][:sample_count], dtype=float)
            valid_mask = ~(np.isnan(preds) | np.isnan(targets))
            if not valid_mask.any():
                continue
            valid_preds = preds[valid_mask]
            valid_targets = targets[valid_mask]
            valid_modes = mode_array[valid_mask] if mode_array is not None else None

            axes[idx].plot(
                [min(valid_targets.min(), valid_preds.min()), max(valid_targets.max(), valid_preds.max())],
                [min(valid_targets.min(), valid_preds.min()), max(valid_targets.max(), valid_preds.max())],
                'r--',
                linewidth=2,
                label='Perfect Prediction'
            )

            if valid_modes is not None:
                for mode_name in ['single_star', 'two_star']:
                    mode_mask = (valid_modes == mode_name)
                    if mode_mask.any():
                        axes[idx].scatter(
                            valid_targets[mode_mask],
                            valid_preds[mode_mask],
                            alpha=0.7,
                            s=20,
                            color=MODE_COLOR_MAP.get(mode_name, 'gray'),
                            label=MODE_DISPLAY_NAMES.get(mode_name, mode_name.replace('_', ' ').title())
                        )
            else:
                axes[idx].scatter(
                    valid_targets,
                    valid_preds,
                    alpha=0.7,
                    s=20,
                    color=MODE_COLOR_MAP.get('single_star', 'tab:blue'),
                    label='Predicted'
                )

            axes[idx].set_xlabel(f'True {param_info.get(param, {}).get("label", param)}')
            axes[idx].set_ylabel(f'Predicted {param_info.get(param, {}).get("label", param)}')
            axes[idx].set_title(f'{param} ({method_label})')
            axes[idx].grid(True, alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', ncol=3)
        plt.tight_layout(rect=(0, 0, 1, 0.92))
        combined_filename = f'stellar_parameters_combined_{method_slug}.png'
        plt.savefig(os.path.join(output_dir, combined_filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Combined scalar plot saved ({method_label})")

def plot_training_fit_results(checkpoint_path: str, output_dir: str) -> None:
    """Pot aggregated training losses stored in fit_res.json."""
    fit_res_path = os.path.join(os.path.dirname(checkpoint_path), 'fit_res.json')
    if not os.path.isfile(fit_res_path):
        print(f"Warning: No fit_res.json found at {fit_res_path}, skipping training loss plot")
        return

    try:
        with open(fit_res_path, 'r') as f:
            fit_res = json.load(f)
    except Exception as exc:
        print(f"Warning: Failed to load fit_res.json ({exc}), skipping training loss plot")
        return

    epochs = fit_res.get('epochs', [])
    num_epochs = len(epochs)
    if num_epochs == 0:
        print("Warning: fit_res.json missing epoch information; skipping training loss plot")
        return

    train_losses = average_losses_by_epoch(fit_res.get('train_loss', []), num_epochs)
    ce_losses = average_losses_by_epoch(fit_res.get('ce_losses', []), num_epochs)
    stellar_losses = average_losses_by_epoch(fit_res.get('stellar_losses', []), num_epochs)
    val_losses = average_losses_by_epoch(fit_res.get('val_loss', []), num_epochs)

    series = [
        ("Train Loss", train_losses, 'tab:blue'),
        ("Cross-Entropy Loss", ce_losses, 'tab:green'),
        ("Stellar Loss", stellar_losses, 'tab:orange'),
    ]
    if val_losses and any(not np.isnan(v) for v in val_losses):
        series.append(("Validation Loss", val_losses, 'tab:red'))

    valid_lengths = [len(values) for _, values, _ in series if values]
    if not valid_lengths:
        print("Warning: No valid loss series available to plot")
        return

    plot_len = min(valid_lengths)
    if plot_len == 0:
        print("Warning: Loss series contain no data after aggregation")
        return

    def _trim(values):
        arr = np.asarray(values, dtype=float)
        if arr.size >= plot_len:
            arr = arr[:plot_len]
        return arr

    x_epochs = epochs[:plot_len] if len(epochs) >= plot_len else list(range(plot_len))

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for label, values, color in series:
        arr = _trim(values)
        if arr.size == 0:
            continue
        mask = ~np.isnan(arr)
        if not mask.any():
            continue
        plt.plot(np.asarray(x_epochs)[mask], arr[mask], label=label, color=color, linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Components Over Epochs')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'training_losses.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Training loss plot saved to {output_path}")


def create_non_conditioned_batch(batch, tokenizer=None):
    """
    Create a non-conditioned version of the batch by removing the first star's parameters
    from the question text. The question format is typically:
    'This is a star with Teff1, logg1, FeH1. Describe a star with Teff2, logg2, FeH2.'
    We want to extract just: 'Describe a star with Teff2, logg2, FeH2.'
    """
    try:
        # We need to reconstruct the text from input_ids to modify it
        if tokenizer is None:
            # If no tokenizer available, we can't modify the text
            return None
            
        # Get the input_ids for this batch
        input_ids = batch['input_ids']
        batch_size = input_ids.shape[0]
        
        modified_input_ids = []
        
        for i in range(batch_size):
            # Decode the current question
            current_ids = input_ids[i]
            # Remove padding tokens for decoding
            pad_token_id = tokenizer.pad_id if hasattr(tokenizer, 'pad_id') else 0
            current_ids = current_ids[current_ids != pad_token_id]
            current_text = tokenizer.decode(current_ids.cpu().numpy())
            
            # Try to extract the second part of the question
            # Look for patterns like "Describe a star with" or "describe a star with"
            describe_patterns = ["Describe a star with", "describe a star with", "Describe the star with", "describe the star with"]
            
            modified_text = None
            for pattern in describe_patterns:
                if pattern in current_text:
                    # Find the position and extract everything from there
                    pos = current_text.find(pattern)
                    modified_text = current_text[pos:].strip()
                    break
            
            if modified_text is None:
                # Fallback: try to find the second sentence by splitting on periods
                sentences = current_text.split('.')
                if len(sentences) >= 2:
                    # Take the last meaningful sentence
                    for j in range(len(sentences) - 1, -1, -1):
                        if sentences[j].strip() and ('star' in sentences[j].lower() or 'describe' in sentences[j].lower()):
                            modified_text = sentences[j].strip()
                            if not modified_text.endswith('.'):
                                modified_text += '.'
                            break
            
            if modified_text is None:
                # Last resort: use original text
                modified_text = current_text
            
            # Encode the modified text back to tokens
            modified_ids = tokenizer.encode(modified_text)
            # Pad to original length
            if len(modified_ids) < len(current_ids):
                padding_length = len(current_ids) - len(modified_ids)
                modified_ids = modified_ids + [pad_token_id] * padding_length
            elif len(modified_ids) > len(current_ids):
                # Truncate if too long
                modified_ids = modified_ids[:len(current_ids)]
            
            modified_input_ids.append(torch.tensor(modified_ids))
        
        # Create new batch with modified input_ids
        modified_batch = batch.copy()
        modified_batch['input_ids'] = torch.stack(modified_input_ids)
        
        return modified_batch
        
    except Exception as e:
        print(f"Warning: Failed to create non-conditioned batch: {e}")
        return None


def run_feature_prediction_evaluation(trainer: LLMTrainer,
                                     test_loader,
                                     device: torch.device,
                                     args,
                                     inference_args,
                                     plots_dir: str) -> None:
    """Run feature prediction evaluation with UMAP visualization, comparing conditioned vs non-conditioned predictions."""
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    # Try to import UMAP, fallback to t-SNE if not available
    try:
        import umap.umap_ as umap
        use_umap = True
        print("Using UMAP for dimensionality reduction")
    except ImportError:
        print("UMAP not available, using t-SNE instead")
        use_umap = False
    
    # Load tokenizer for non-conditioned batch creation
    tokenizer = None
    try:
        _, tokenizer_path = get_model_path(args)
        print("trying to loaf tokenizer from: ", tokenizer_path)
        tokenizer = Tokenizer(model_path=tokenizer_path)
        print(f"[OK] Loaded tokenizer from {tokenizer_path}")
    except Exception as e:
        print(f"Warning: Failed to load tokenizer: {e}")
    
    def collect_predictions(data_loader, batch_modifier=None, condition_name="Original"):
        """Collect predictions with optional batch modification."""
        true_features = []
        predicted_features = []
        sample_indices = []
        true_params = []
        true_snr = []
        true_class = []
        giant = []
        
        print(f"Collecting predictions for {condition_name} dataset...")
        
        trainer.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                # Optionally modify batch (for non-conditioned version)
                if batch_modifier:
                    batch = batch_modifier(batch, tokenizer)
                    if batch is None:
                        continue
                
                # Get true features (star2 is the target)
                true_feat = batch['features_star2'].to(device)  # (batch_size, feature_dim)
                true_features.append(true_feat.cpu().numpy())
                
                # Run model inference to get predicted features
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                masked_spectra = batch['features_star1'].to(device)  # Star1 features as input

                y_numeric = batch['y_numeric_star2'] # (batch_size, 3) Teff,logg,FeH
                true_params.append(y_numeric.numpy())

                stellar_data = batch['stellar_data_star2']
                true_snr.append(stellar_data['snrg'])
                true_class.append(stellar_data['subclass'])
                giant.append(stellar_data['is_giant'])

                batch_device = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch_device[key] = value.to(device)
                    else:
                        batch_device[key] = value
                    
                # Get model outputs with feature prediction
                outputs = trainer.model(batch_device)            
                # Extract predicted features
                if 'predicted_features' in outputs.keys():
                    pred_feat = outputs['predicted_features']  # (batch_size, feature_dim)
                    predicted_features.append(pred_feat.cpu().numpy())
                else:
                    # Fallback: use last hidden states or feature projection if available
                    print("Warning: No predicted_features in model output, using alternative method")
                    if hasattr(trainer.model, 'feature_projection') and hasattr(outputs, 'last_hidden_state'):
                        # Use feature projection head if available
                        feature_tokens = outputs.last_hidden_state[:, :args.num_spectral_features, :]  # Get feature tokens
                        pred_feat = trainer.model.feature_projection(feature_tokens.mean(dim=1))  # Average and project
                        predicted_features.append(pred_feat.cpu().numpy())
                    else:
                        # Skip this batch if no feature prediction available
                        print(f"Skipping batch {i}: no feature prediction capability found")
                        true_features.pop()  # Remove the true features we added
                        continue
                
                # Track sample indices for debugging
                sample_indices.extend(range(i * test_loader.batch_size, 
                                          min((i + 1) * test_loader.batch_size, len(test_loader.dataset))))
                
                # Limit to reasonable number for visualization
                # if len(true_features) * test_loader.batch_size >= 1000:  # Set reasonable default
                #     break
        
        if not true_features:
            print(f"Error: No feature predictions collected for {condition_name} dataset.")
            return None
        
        # Concatenate all features
        return {
            'true_features': np.concatenate(true_features, axis=0),
            'predicted_features': np.concatenate(predicted_features, axis=0),
            'true_params': np.concatenate(true_params, axis=0),
            'true_snr': np.concatenate(true_snr, axis=0),
            'giant': np.concatenate(giant, axis=0),
            'condition_name': condition_name
        }
    
    print("Running feature prediction evaluation...")
    
    # Collect predictions for both conditioned and non-conditioned datasets
    conditioned_data = collect_predictions(test_loader, None, "Conditioned")
    if conditioned_data is None:
        print("Error: Failed to collect conditioned predictions.")
        return
    
    non_conditioned_data = collect_predictions(test_loader, create_non_conditioned_batch, "Non-Conditioned")
    if non_conditioned_data is None:
        print("Warning: Failed to collect non-conditioned predictions, creating plots only for conditioned data.")
        datasets = [conditioned_data]
    else:
        datasets = [conditioned_data, non_conditioned_data]
    
    def create_plots_for_datasets(datasets, plots_dir):
        """Create paired plots comparing conditioned vs non-conditioned datasets."""
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
        import json
        
        # Determine method name for dimensionality reduction
        method_name = "UMAP" if use_umap else "t-SNE"
        
        # Process each dataset and compute embeddings
        processed_datasets = []
        for data in datasets:
            # Calculate distances in feature space
            feature_distances = np.linalg.norm(data['true_features'] - data['predicted_features'], axis=1)
            avg_feature_distance = np.mean(feature_distances)
            std_feature_distance = np.std(feature_distances)
            
            print(f"{data['condition_name']} - Average feature space distance: {avg_feature_distance:.4f} ± {std_feature_distance:.4f}")
            
            # Prepare data for UMAP/t-SNE
            all_features = np.vstack([data['true_features'], data['predicted_features']])
            
            # Apply dimensionality reduction
            if use_umap:
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
                embedding = reducer.fit_transform(all_features)
            else:
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features)//4))
                embedding = reducer.fit_transform(all_features)
            
            # Split embeddings
            n_samples = len(data['true_features'])
            true_embedding = embedding[:n_samples]
            pred_embedding = embedding[n_samples:]
            
            # Calculate distances in embedding space
            embedding_distances = np.linalg.norm(true_embedding - pred_embedding, axis=1)
            avg_embedding_distance = np.mean(embedding_distances)
            std_embedding_distance = np.std(embedding_distances)
            
            print(f"{data['condition_name']} - Average {method_name} space distance: {avg_embedding_distance:.4f} ± {std_embedding_distance:.4f}")
            
            processed_data = data.copy()
            processed_data.update({
                'true_embedding': true_embedding,
                'pred_embedding': pred_embedding,
                'feature_distances': feature_distances,
                'embedding_distances': embedding_distances,
                'avg_feature_distance': avg_feature_distance,
                'std_feature_distance': std_feature_distance,
                'avg_embedding_distance': avg_embedding_distance,
                'std_embedding_distance': std_embedding_distance,
                'method_name': method_name
            })
            processed_datasets.append(processed_data)
        
        # Plot 1: Separate plots for true and predicted features (side-by-side comparison)
        n_datasets = len(processed_datasets)
        fig_height = 7 if n_datasets == 1 else 14
        fig, axes = plt.subplots(n_datasets, 2, figsize=(16, fig_height))
        if n_datasets == 1:
            axes = axes.reshape(1, -1)
        
        for i, data in enumerate(processed_datasets):
            # True features
            scatter1 = axes[i, 0].scatter(data['true_embedding'][:, 0], data['true_embedding'][:, 1], 
                              alpha=0.7, s=30, c=data['true_params'][:, 0], label='True Features')
            axes[i, 0].set_title(f'{data["condition_name"]} - True Features ({method_name})')
            axes[i, 0].set_xlabel(f'{method_name} 1')
            axes[i, 0].set_ylabel(f'{method_name} 2')
            axes[i, 0].grid(True, alpha=0.3)
            cbar1 = fig.colorbar(scatter1, ax=axes[i, 0])
            cbar1.set_label('Teff (K)')
            axes[i, 0].legend()
            
            # Predicted features
            scatter2 = axes[i, 1].scatter(data['pred_embedding'][:, 0], data['pred_embedding'][:, 1], 
                              alpha=0.7, s=30, c=data['true_params'][:, 0], label='Predicted Features')
            axes[i, 1].set_title(f'{data["condition_name"]} - Predicted Features ({method_name})')
            axes[i, 1].set_xlabel(f'{method_name} 1')
            axes[i, 1].set_ylabel(f'{method_name} 2')
            axes[i, 1].grid(True, alpha=0.3)
            cbar2 = fig.colorbar(scatter2, ax=axes[i, 1])
            cbar2.set_label('Teff (K)')
            axes[i, 1].legend()
        
        plt.tight_layout()
        plot1_path = os.path.join(plots_dir, 'feature_prediction_separate_comparison.png')
        plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Separate feature comparison plot saved to {plot1_path}")
        
        # Plot 2: Paired visualization (side-by-side comparison)
        fig, axes = plt.subplots(1, n_datasets, figsize=(12*n_datasets, 9))
        if n_datasets == 1:
            axes = [axes]
        
        for i, data in enumerate(processed_datasets):
            # Background: all true features in light gray
            axes[i].scatter(data['true_embedding'][:, 0], data['true_embedding'][:, 1], 
                          alpha=0.3, s=20, c='lightgray', label='True Features (background)')
            
            # Show first 50 pairs with lines and colors based on SNR values
            n_pairs_to_show = min(50, len(data['true_features']))
            color_arr = data['true_snr'][:n_pairs_to_show]
            color_label = 'SNRG'
            
            # Create a colormap for the values
            norm = Normalize(vmin=color_arr.min(), vmax=color_arr.max())
            cmap = cm.viridis
            
            for j in range(n_pairs_to_show):
                color = cmap(norm(color_arr[j]))
                
                # Draw line connecting true and predicted
                axes[i].plot([data['true_embedding'][j, 0], data['pred_embedding'][j, 0]], 
                           [data['true_embedding'][j, 1], data['pred_embedding'][j, 1]], 
                           color=color, alpha=0.6, linewidth=1)
                
                # True feature point
                axes[i].scatter(data['true_embedding'][j, 0], data['true_embedding'][j, 1], 
                              s=60, c=[color], marker='o', edgecolors='black', linewidths=0.5,
                              alpha=0.8)
                
                # Predicted feature point
                axes[i].scatter(data['pred_embedding'][j, 0], data['pred_embedding'][j, 1], 
                              s=60, c=[color], marker='s', edgecolors='black', linewidths=0.5,
                              alpha=0.8)
            
            # Add colorbar
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes[i])
            cbar.set_label(color_label)
            
            # Custom legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                           markersize=8, markeredgecolor='black', label='True Features'),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                           markersize=8, markeredgecolor='black', label='Predicted Features'),
                plt.Line2D([0], [0], color='gray', alpha=0.6, label=f'Prediction Links (first {n_pairs_to_show})')
            ]
            axes[i].legend(handles=legend_elements, loc='best')
            
            axes[i].set_title(f'{data["condition_name"]} - Feature Prediction Consistency ({method_name})\n'
                            f'Avg Feature Distance: {data["avg_feature_distance"]:.4f} ± {data["std_feature_distance"]:.4f}, '
                            f'Avg {method_name} Distance: {data["avg_embedding_distance"]:.4f} ± {data["std_embedding_distance"]:.4f}')
            axes[i].set_xlabel(f'{method_name} 1')
            axes[i].set_ylabel(f'{method_name} 2')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot2_path = os.path.join(plots_dir, 'feature_prediction_paired_comparison.png')
        plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Paired feature comparison plot saved to {plot2_path}")
        
        # Plot 3: Distance distribution histograms (side-by-side comparison)
        fig, axes = plt.subplots(2, n_datasets, figsize=(7*n_datasets, 10))
        if n_datasets == 1:
            axes = axes.reshape(-1, 1)
        
        for i, data in enumerate(processed_datasets):
            # Feature space distances
            axes[0, i].hist(data['feature_distances'], bins=30, alpha=0.7, color='blue', edgecolor='black')
            axes[0, i].axvline(data['avg_feature_distance'], color='red', linestyle='--', 
                        label=f'Mean: {data["avg_feature_distance"]:.4f}')
            axes[0, i].set_xlabel('Feature Space Distance')
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].set_title(f'{data["condition_name"]} - Feature Space Distances')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Embedding space distances
            axes[1, i].hist(data['embedding_distances'], bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[1, i].axvline(data['avg_embedding_distance'], color='red', linestyle='--', 
                        label=f'Mean: {data["avg_embedding_distance"]:.4f}')
            axes[1, i].set_xlabel(f'{method_name} Space Distance')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].set_title(f'{data["condition_name"]} - {method_name} Space Distances')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot3_path = os.path.join(plots_dir, 'feature_prediction_distances_comparison.png')
        plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Distance distribution comparison plot saved to {plot3_path}")
        
        # Save metrics to file
        metrics = {}
        for data in processed_datasets:
            condition_key = data['condition_name'].lower().replace(' ', '_').replace('-', '_')
            metrics[condition_key] = {
                'avg_feature_distance': float(data['avg_feature_distance']),
                'std_feature_distance': float(data['std_feature_distance']),
                'avg_embedding_distance': float(data['avg_embedding_distance']),
                'std_embedding_distance': float(data['std_embedding_distance']),
                'n_samples': int(len(data['true_features'])),
                'feature_dim': int(data['true_features'].shape[1]),
                'embedding_method': method_name
            }
        
        metrics_path = os.path.join(plots_dir, 'feature_prediction_metrics_comparison.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Comparison metrics saved to {metrics_path}")
    
    # Create visualizations
    os.makedirs(plots_dir, exist_ok=True)
    create_plots_for_datasets(datasets, plots_dir)


def main():
    """Main inference function"""
    
    # Parse arguments
    inference_args = parse_inference_args()
    
    # Load training config
    if inference_args.config_path:
        with open(inference_args.config_path, 'r') as f:
            config = json.load(f)
    else:
        config = load_config_from_checkpoint_dir(inference_args.checkpoint_path)
    
    # Create full args from config and inference args
    args = create_args_from_config(config, inference_args)
    # Set up device
    device, world_size, gpus_per_node = setup()
    print(f"Using device: {device}")

    
    # Create output directory relative to checkpoint path
    checkpoint_parent = os.path.dirname(inference_args.checkpoint_path)
    output_dir = os.path.join(checkpoint_parent, inference_args.output_dir)
    inference_args.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_model(inference_args.checkpoint_path, args, device)

    model.eval()
    
    # Create dataloaders (we only need test_loader)
    print("Creating test dataloader...")
    print("single_sample_prob: ", args.single_sample_prob)
    # args.single_sample_prob = 1.0
    # args.mode = "single_star"
    backend_config = ensure_backend_config(args)
    train_loader, val_loader, test_loader = create_datasets_and_loaders(args, device, backend_config)

    optimizer, scheduler, scaler = create_optimizer_and_scheduler(model, args, train_loader)

    tuned_cfg_path = os.path.join(ROOT_DIR, 'src', 'llm_config_tuned.json')
    if os.path.isfile(tuned_cfg_path):
        with open(tuned_cfg_path, 'r') as f:
            tuned_cfg = json.load(f)
        lora_params = tuned_cfg.get('lora_params', {})
    else:
        # Fallback to base config if tuned not found
        base_cfg_path = os.path.join(ROOT_DIR, 'src', 'llm_config.json')
        with open(base_cfg_path, 'r') as f:
            base_cfg = json.load(f)
        lora_params = base_cfg.get('lora_params', {})
    
    # Run predictions
    trainer = LLMTrainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        world_size=world_size,
        output_dim=1,
        scheduler=None,
        max_iter=args.max_iter,
        log_path=args.output_dir,
        exp_name=args.exp_name,
        lora_params=lora_params,
        scaler=None,
        use_amp=args.use_amp,
        max_grad_norm=args.max_grad_norm,
        mode=args.mode,
        curriculum_decay_steps=args.curriculum_decay_steps,
        quantiles=args.quantiles,
    )
    
    trainer.combined_mode = (args.mode == "combined")

    # Get quantiles from args for CQR calibration
    quantiles = getattr(args, 'quantiles', [0.159, 0.5, 0.841])  # Default: ~1-sigma + median
    print(f"Using quantiles for CQR calibration: {quantiles}")
    
    plots_dir = os.path.join(inference_args.output_dir, 'plots')

    run_prediction_evaluation(
        trainer=trainer,
        test_loader=test_loader,
        device=device,
        args=args,
        inference_args=inference_args,
        quantiles=quantiles,
        plots_dir=plots_dir,
        backend_config=backend_config,
    )

    # Run feature prediction evaluation if requested
    if getattr(inference_args, 'predict_features', False):
        print("\n" + "="*80)
        print("FEATURE PREDICTION EVALUATION")
        print("="*80)
        run_feature_prediction_evaluation(
            trainer=trainer,
            test_loader=test_loader,
            device=device,
            args=args,
            inference_args=inference_args,
            plots_dir=plots_dir,
        )
        print("\n" + "="*80)
        print("FEATURE SIMILARITY TRAJECTORY ANALYSIS")
        print("="*80)
        run_multiple_feature_trajectories(
            trainer=trainer,
            args=args,
            dataset=test_loader.dataset,
            collate_fn=getattr(test_loader, 'collate_fn', None),
            device=device,
            num_trajectories=5,
            max_trajectory_length=10,
            output_dir=plots_dir
        )

    interpolation_results, interpolation_pairs, background_points = run_interpolation(
        trainer=trainer,
        test_loader=test_loader,
        device=device,
        args=args,
        inference_args=inference_args,
        quantiles=quantiles,
        plots_dir=plots_dir,
    )

    baseline_ckpt_arg = getattr(inference_args, 'baseline_mlp_checkpoint_path', None)
    if baseline_ckpt_arg:
        baseline_ckpt_path = baseline_ckpt_arg if os.path.isabs(baseline_ckpt_arg) else os.path.join(ROOT_DIR, baseline_ckpt_arg)
    else:
        baseline_ckpt_path = os.path.join(ROOT_DIR, 'logs', 'baseline_mlp', 'baseline_mlp.pt')
    baseline_available = os.path.isfile(baseline_ckpt_path)

    comparison_requested = getattr(inference_args, 'comparison_checkpoint_paths', None)
    should_compare = (
        interpolation_results and interpolation_pairs and
        (comparison_requested or baseline_available)
    )

    if should_compare:
        compare_models_on_interpolations(
            base_results=interpolation_results,
            interpolation_pairs=interpolation_pairs,
        background_points=background_points,
        base_trainer=trainer,
        inference_args=inference_args,
        base_args=args,
        device=device,
        quantiles=quantiles,
        plots_dir=plots_dir,
        dataset=test_loader.dataset,
        collate_fn=getattr(test_loader, 'collate_fn', None),
        )

    print("[OK] Inference completed successfully!")


if __name__ == '__main__':
    main()
