#!/usr/bin/env python3
"""
Inference script for the multimodal stellar model.
Loads a trained model, runs predictions on test set, and saves results.
"""

import os
import sys
import json
import argparse
import random
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional, Union
from copy import deepcopy

# Add root directory to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.simple_questions import (
    parse_args as base_parse_args,
    _load_llm_model,
    _load_spectra_model,
    get_model_path,
    create_optimizer_and_scheduler,
    setup
)
from src.simple_questions_multitok import (
    create_datasets_and_loaders,
    build_model_multitok
)
from nn.llm_multi import MultimodalLlamaModelMultiTokens
from llama3.llama.tokenizer import Tokenizer
from nn.train import LLMTrainer
from nn.optim import CQR
from data.transforms import GeneralSpectrumPreprocessor, ToTensor, Compose
import torch.nn.functional as F
import re


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


def run_feature_interpolation(trainer: LLMTrainer,
                             args,
                               dataset,
                               collate_fn,
                               device: Union[int, torch.device],
                               pairs: List[Tuple[int, int]],
                               alphas: List[float],
                               output_dir: str,
                               quantiles: List[float],
                               min_teff_diff: float = 0.0) -> List[Dict[str, Any]]:
    """Interpolate latent features between sample pairs and record predicted stellar parameters."""
    if not pairs:
        return []

    _, tokenizer_path = get_model_path(args)
    tokenizer = Tokenizer(model_path=tokenizer_path)

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
        tokens = sample_dict.get('tokens')
        if tokens is None or not torch.is_tensor(tokens):
            return
        metadata = sample_dict.get('metadata') or {}
        feature_start = int(metadata.get('feature_start_idx', 0) or 0)
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

        sample_dict['tokens'] = new_tokens
        sample_dict['target_ids'] = torch.full_like(new_tokens, -100)

        span = sample_dict.get('span_indices')
        if torch.is_tensor(span) and span.numel() >= 2:
            span[0] = end
            span[1] = end

        metadata['input_text'] = prompt
        metadata['target_text'] = ''
        metadata['question_start_idx'] = start
        metadata['answer_start_idx'] = end
        metadata['target_length'] = 0
        if 'raw' in metadata and isinstance(metadata['raw'], dict):
            metadata['raw']['input_text'] = prompt
            metadata['raw']['target_text'] = ''
            metadata['raw']['question_start_idx'] = start
            metadata['raw']['answer_start_idx'] = end
            metadata['raw']['target_length'] = 0
        sample_dict['metadata'] = metadata

    prompt_text = 'Describe this star.'

    target_dim = get_single_star_latent_dim(trainer)

    os.makedirs(output_dir, exist_ok=True)

    bounds = {
        'Teff': (3000.0, 7500.0),
        'logg': (0.0, 5.0),
        'FeH': (-3.0, 0.5),
    }
    median_idx = len(quantiles) // 2 if quantiles else 0
    results: List[Dict[str, Any]] = []

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
        if sample_dict is None:
            return None

        teff_norm: Optional[float] = None
        y_numeric = sample_dict.get('y_numeric')
        if y_numeric is not None:
            if torch.is_tensor(y_numeric):
                if y_numeric.numel() > 0:
                    teff_norm = float(y_numeric.view(-1)[0].item())
            else:
                try:
                    arr = np.asarray(y_numeric, dtype=float).reshape(-1)
                    if arr.size > 0:
                        teff_norm = float(arr[0])
                except Exception:
                    teff_norm = None

        if teff_norm is not None and not np.isnan(teff_norm):
            low, high = bounds['Teff']
            return teff_norm * (high - low) + low

        meta = sample_dict.get('metadata')
        candidate_dicts = []
        if isinstance(meta, dict):
            raw_meta = meta.get('raw')
            if isinstance(raw_meta, dict):
                candidate_dicts.extend([
                    raw_meta.get('stellar_data'),
                    raw_meta.get('star_a'),
                    raw_meta.get('star_a_params'),
                    raw_meta
                ])
        for data in candidate_dicts:
            if not isinstance(data, dict):
                continue
            for key in ('Teff', 'teff_k', 'teff', 'effective_temperature'):
                val = data.get(key) if data else None
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        continue
        return None

    for idx_a, idx_b in pairs:
        try:
            sample_a = deepcopy(dataset[idx_a])
            sample_b = deepcopy(dataset[idx_b])
        except Exception as exc:
            print(f"Warning: Failed to load samples {idx_a}:{idx_b} for interpolation ({exc})")
            continue


        teff_a = _extract_teff(sample_a)
        teff_b = _extract_teff(sample_b)
        if min_teff_diff > 0.0:
            if teff_a is None or teff_b is None:
                print(f"Warning: Samples {idx_a}:{idx_b} missing Teff data; skipping pair.")
                continue
            if abs(teff_a - teff_b) < min_teff_diff:
                diff = abs(teff_a - teff_b)
                print(f"Warning: Teff difference {diff:.1f} K for samples {idx_a}:{idx_b} below threshold {min_teff_diff:.1f} K; skipping pair.")
                continue

        pair_record = {
            'pair': (idx_a, idx_b),
            'alpha_points': [],
            'teff_values': [teff_a, teff_b]
        }
        tensor_a = get_tensor_from_sample(sample_a, 'masked_spectra')
        tensor_b = get_tensor_from_sample(sample_b, 'masked_spectra')
        if tensor_a is None or tensor_b is None:
            print(f"Warning: Samples {idx_a}:{idx_b} missing masked spectra; skipping pair.")
            continue

        tensor_a = tensor_a.clone()
        tensor_b = tensor_b.clone()

        for alpha in alphas:
            alpha = float(max(0.0, min(1.0, alpha)))

            synthetic_sample = deepcopy(sample_a)
            synthetic_sample['mode'] = 'single'

            interpolated_core = ((1.0 - alpha) * tensor_a + alpha * tensor_b).to(dtype=torch.float32)
            synthetic_sample['masked_spectra'] = interpolated_core.clone()

            _retokenize_single_prompt(synthetic_sample, prompt_text)

            # Ensure metadata carries the interpolated tensors so collate functions pick them up
            meta = synthetic_sample.get('metadata')
            if isinstance(meta, dict):
                meta['masked_spectra'] = synthetic_sample['masked_spectra']

            batch = collate([synthetic_sample])
            batch_device: Dict[str, Any] = {}
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch_device[key] = value.to(device)
                else:
                    batch_device[key] = value
            with torch.no_grad():
                outputs = trainer.get_logits(batch_device, device, val=True)

                generated_text, input_text, target_text, _ = trainer.model.generate_response_from_batch(
                                batch_data=batch_device,
                                batch_idx=0,
                                tokenizer=tokenizer,
                                max_new_tokens=50,
                                temperature=0.0,
                                top_p=1.0
                            )
                            
            stellar_preds = outputs.get('stellar_predictions')

            # input_text = batch_device.get('input_texts', [''])[batch_idx]
            # target_text = batch_device.get('target_texts', [''])[batch_idx]
            # gen_ids = generate_response(outputs)
            # generated_text = tokenizer.decode(torch.tensor(gen_ids).cpu().numpy()) if tokenizer is not None else ''


            norm_preds = _extract_norm_predictions(stellar_preds)
            teff_pred = float(_convert_norm_to_physical('Teff', norm_preds.get('Teff', float('nan'))))
            logg_pred = float(_convert_norm_to_physical('logg', norm_preds.get('logg', float('nan'))))
            feh_pred = float(_convert_norm_to_physical('FeH', norm_preds.get('FeH', float('nan'))))

            # print("stellar preds: ", teff_pred, logg_pred, feh_pred)
            # print("input text: ", input_text)
            # print("generated_text: ", generated_text)
            # print("target_text: ", target_text)

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
                'text_teff': text_physical.get('Teff'),
                'text_logg': text_physical.get('logg'),
                'text_feh': text_physical.get('FeH'),
            })

        results.append(pair_record)

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
            'alpha_points': []
        }
        for point in record.get('alpha_points', []):
            cleaned_point = {key: _sanitize(val) for key, val in point.items()}
            cleaned_record['alpha_points'].append(cleaned_point)
        serializable_results.append(cleaned_record)

    with open(output_json, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"✓ Interpolation results saved to {output_json}")

    return results


def select_random_interpolation_pairs(dataset,
                                      trainer: LLMTrainer,
                                      num_pairs: int,
                                      seed: int = 42,
                                      min_teff_diff: float = 0.0) -> List[Tuple[int, int]]:
    """Randomly choose interpolation pairs that have valid single-star features."""
    if num_pairs <= 0:
        return []

    teff_bounds = (3000.0, 7500.0)

    def _extract_teff(sample_dict: Dict[str, Any]) -> Optional[float]:
        if sample_dict is None:
            return None
        y_numeric = sample_dict.get('y_numeric')
        teff_norm: Optional[float] = None
        if y_numeric is not None:
            if torch.is_tensor(y_numeric):
                if y_numeric.numel() > 0:
                    teff_norm = float(y_numeric.view(-1)[0].item())
            else:
                try:
                    arr = np.asarray(y_numeric, dtype=float).reshape(-1)
                    if arr.size > 0:
                        teff_norm = float(arr[0])
                except Exception:
                    teff_norm = None
        if teff_norm is not None and not np.isnan(teff_norm):
            low, high = teff_bounds
            return teff_norm * (high - low) + low

        meta = sample_dict.get('metadata')
        if isinstance(meta, dict):
            raw_meta = meta.get('raw')
            if isinstance(raw_meta, dict):
                for data in (raw_meta.get('stellar_data'), raw_meta.get('star_a'), raw_meta.get('star_a_params'), raw_meta):
                    if not isinstance(data, dict):
                        continue
                    for key in ('Teff', 'teff_k', 'teff', 'effective_temperature'):
                        val = data.get(key)
                        if val is not None:
                            try:
                                return float(val)
                            except (TypeError, ValueError):
                                continue
        return None

    target_dim = get_single_star_latent_dim(trainer)
    total_samples = len(dataset)
    indices = list(range(total_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    valid: List[Tuple[int, Optional[float]]] = []
    for idx in indices:
        try:
            sample = dataset[idx]
        except Exception as exc:
            print(f"Warning: Failed to load sample {idx} while searching for interpolation pairs ({exc})")
            continue
        if find_interpolatable_tensor(sample, target_dim) is None:
            continue
        teff = _extract_teff(sample)
        if min_teff_diff > 0.0 and teff is None:
            print(f"Warning: Sample {idx} missing Teff data; skipping for interpolation pair selection.")
            continue
        valid.append((idx, teff))
        if len(valid) >= num_pairs * 4:
            break

    if len(valid) < 2:
        print("Warning: Not enough valid samples found for interpolation.")
        return []

    pairs: List[Tuple[int, int]] = []
    used_indices = set()
    for i, (idx_a, teff_a) in enumerate(valid):
        if len(pairs) >= num_pairs:
            break
        if i in used_indices:
            continue
        best_match = None
        for j in range(i + 1, len(valid)):
            if j in used_indices:
                continue
            idx_b, teff_b = valid[j]
            if min_teff_diff > 0.0:
                if teff_a is None or teff_b is None:
                    continue
                if abs(teff_a - teff_b) < min_teff_diff:
                    continue
            best_match = j
            break
        if best_match is not None:
            used_indices.add(i)
            used_indices.add(best_match)
            pairs.append((idx_a, valid[best_match][0]))

    if len(pairs) < num_pairs:
        print(f"Warning: Only found {len(pairs)} interpolation pairs out of requested {num_pairs}.")

    return pairs


def plot_interpolation_kiel(pair_results: List[Dict[str, Any]], output_dir: str) -> None:
    """Plot Kiel diagrams (Teff-logg) for interpolation trajectories."""
    if not pair_results:
        print("No interpolation results to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

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
        ax.set_xlim(8000, 4000)
        ax.set_ylim(5.5, 2)
        scatter = ax.scatter(teff_valid, logg_valid, c=alphas_valid, cmap='viridis', s=60, edgecolors='k')
        line_color = 'tab:blue'
        ax.plot(teff_valid, logg_valid, linestyle='-', color=line_color, linewidth=2.0, alpha=0.9, label='Model (numerical)')

        # if text_teff_valid.size > 0:
            # ax.plot(text_teff_valid, text_logg_valid, linestyle='--', color=line_color, linewidth=2.0, alpha=0.9, label='Text-derived')
            # ax.scatter(text_teff_valid, text_logg_valid, color=line_color, s=40, alpha=0.75)

        ax.scatter([teff_valid[0]], [logg_valid[0]], color='red', s=80, marker='^', label='alpha=0')
        ax.scatter([teff_valid[-1]], [logg_valid[-1]], color='blue', s=80, marker='s', label='alpha=1')

        ax.set_xlabel('Effective Temperature Teff (K)')
        ax.set_ylabel('Surface Gravity log g')
        ax.set_title(f'Latent Interpolation Pair {record['pair'][0]} → {record['pair'][1]}')
        ax.invert_yaxis()
        ax.invert_xaxis()

        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Interpolation alpha')

        filename = os.path.join(output_dir, f'interpolation_pair_{record['pair'][0]}_{record['pair'][1]}.png')
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Interpolation plot saved to {filename}")



def plot_interpolation_kiel_combined(pair_results: List[Dict[str, Any]], output_dir: str) -> None:
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

    cmap = plt.get_cmap('tab20')
    labels_used_num: set[str] = set()
    labels_used_text: set[str] = set()

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
        color = cmap(idx % cmap.N)
        label = f"{record['pair'][0]}→{record['pair'][1]}"

        line_label = label if label not in labels_used_num else None
        if line_label is not None:
            labels_used_num.add(line_label)
        ax_num.plot(teff_valid, logg_valid, color=color, linewidth=2.0, alpha=0.85, label=line_label)
        ax_num.scatter(teff_valid, logg_valid, color=color, s=35, alpha=0.85)
        ax_num.scatter([teff_valid[0]], [logg_valid[0]], color=color, marker='^', s=70)
        ax_num.scatter([teff_valid[-1]], [logg_valid[-1]], color=color, marker='s', s=70)

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
    print(f"✓ Numeric interpolation plot saved to {numeric_path}")
    print(f"✓ Text interpolation plot saved to {text_path}")

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
        plt.text(0.05, 0.95, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r_squared:.3f}\nN: {len(stellar_vals)}',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{param}_consistency.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Consistency plot saved for {param}")
    
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
            
            axes[i].text(0.05, 0.95, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r_squared:.3f}\nN: {len(stellar_vals)}',
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'stellar_text_consistency_combined.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Combined consistency plot saved")


def llm_predict_stellar_bulk(trainer, data_loader, device, quantiles, max_iter=np.inf):
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
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
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
                    
                    # Reshape to [batch_size, num_params, num_quantiles] 
                    preds_reshaped = stellar_preds_tensor.view(batch_size, num_params, num_quantiles)
                    
                    # Append to predictions array
                    preds = np.concatenate([preds, preds_reshaped.cpu().numpy()], axis=0)
                    
                    # Track per-sample modes
                    batch_modes = infer_batch_modes(batch, batch_size)
                    modes.extend(batch_modes)

                    # Extract obsids for this batch
                    batch_obsids = list(batch.get('obsids', []))
                    if len(batch_obsids) < batch_size:
                        batch_obsids.extend([None] * (batch_size - len(batch_obsids)))
                    obsids.extend(batch_obsids[:batch_size])
                
                    # Extract targets following the same logic as training
                    batch_targets = np.full((batch_size, num_params), np.nan)
                    
                    # Single-star mode
                    if 'y_numeric' in batch and batch['y_numeric'] is not None:
                        gt_params = batch['y_numeric']
                        gt_mask = batch['y_numeric_present']
                        
                        if gt_mask.any():
                            valid_indices = gt_mask.cpu().numpy()
                            valid_gt = gt_params[gt_mask].cpu().numpy()
                            batch_targets[valid_indices] = valid_gt
                    
                    # Two-star mode (use star A)
                    elif 'y_numeric_a' in batch and batch['y_numeric_a'] is not None:
                        gt_params_a = batch['y_numeric_a']
                        gt_mask_a = batch['y_numeric_a_present']
                        
                        if gt_mask_a.any():
                            valid_indices_a = gt_mask_a.cpu().numpy()
                            valid_gt_a = gt_params_a[gt_mask_a].cpu().numpy()
                            batch_targets[valid_indices_a] = valid_gt_a
                    
                    targets = np.concatenate([targets, batch_targets], axis=0)
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(data_loader)}")
                
            if batch_idx > max_iter:
                break
    
    print(f"✓ Bulk prediction completed: preds shape {preds.shape}, targets shape {targets.shape}, obsids: {len(obsids)}")
    return preds, targets, obsids, modes


def llm_predict_with_text_generation(trainer, data_loader, device, max_iter=np.inf, 
                                   generate_text=True, max_new_tokens=128, args=None):
    """
    Enhanced prediction that collects both stellar predictions and generated text
    """
    trainer.model.eval()
    
    # Explicitly load tokenizer using get_model_path
    tokenizer = None
    if generate_text and args is not None:
        try:
            _, tokenizer_path = get_model_path(args)
            tokenizer = Tokenizer(model_path=tokenizer_path)
            print(f"✓ Loaded tokenizer from {tokenizer_path}")
        except Exception as e:
            print(f"Warning: Failed to load tokenizer: {e}")
            generate_text = False
    elif generate_text:
        print("Warning: No args provided for tokenizer loading")
        generate_text = False
    
    # Collect predictions
    all_text_predictions = []
    all_text_targets = []
    all_questions = []
    all_modes = []
    text_obsids = []  # Track obsids for verification
    
    print(f"Running predictions with text generation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
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
            if generate_text:
                print('batch modes: ', batch['mode'])
                for sample_idx in range(batch_size):
                    try:
                        # Get the model (handle DataParallel/DistributedDataParallel)
                        model = trainer.model
                        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                            model = model.module
                        
                        # Generate response for this sample
                        generated_text, input_text, target_text, _ = model.generate_response_from_batch(
                            batch_data=batch,
                            batch_idx=sample_idx,
                            tokenizer=tokenizer,
                            max_new_tokens=max_new_tokens,
                            temperature=0.2,
                            top_p=0.8
                        )
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
            
            if batch_idx % 1 == 0:
                print(f"Processed batch {batch_idx}/{len(data_loader)}")
                
            if batch_idx > max_iter:
                break
    
    print(f"✓ Text generation completed: {len(all_text_predictions)} samples, obsids: {len(text_obsids)}")
    return all_text_predictions, all_text_targets, text_obsids, all_questions, all_modes


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
    parser.add_argument('--llm_root', type=str, default=os.environ.get('LLM_ROOT', '/data/.llama'),
                       help='Root directory containing LLaMA models (or set env LLM_ROOT)')
    parser.add_argument('--llm_model', type=str, default='Llama3.1-8B',)
    
    # Data arguments (can be overridden from config)
    parser.add_argument('--json_file', type=str, 
                        default='/data/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json',
                        help='Path to main dataset JSON file')
    parser.add_argument('--comparative_json_file', type=str,
                        default='/data/TalkingLatents/data/dataset/comparative_dataset.json',
                        help='Path to comparative dataset JSON file')
    parser.add_argument('--features_file', type=str,
                        default='/data/TalkingLatents/logs/2025-07-29/features.npy',
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

    parser.add_argument('--interpolation_num_pairs', type=int, default=0,
                        help='Randomly choose this many interpolation pairs if explicit pairs are not provided')
    parser.add_argument('--interpolation_seed', type=int, default=42,
                        help='Random seed for interpolation pair selection')

    
    return parser.parse_args()


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
    """Create args namespace from training config, overriding with inference args"""
    
    # Start with inference_args as the base
    args = argparse.Namespace(**vars(inference_args))
    
    # Update with all config values (config takes precedence for any overlaps)
    for key, value in config.items():
        setattr(args, key, value)
    
    return args


def load_model(checkpoint_path: str, args: argparse.Namespace, device: Union[int, torch.device]) -> torch.nn.Module:
    """Load the trained model from checkpoint"""
    
    print(f"Loading model from {checkpoint_path}")

    if not hasattr(args, 'llm_precision'):
        args.llm_precision = 'fp16'
    if not hasattr(args, 'gradient_checkpointing'):
        args.gradient_checkpointing = False
    
    # Convert device to torch.device if it's an int (local rank)
    if isinstance(device, int):
        device = torch.device(f'cuda:{device}')
    
    # Build the model architecture
    model = build_model_multitok(args, device, world_size=1)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel/DistributedDataParallel prefixes
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    # Load state dict
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    print("✓ Model loaded successfully")
    return model




def save_predictions(text_predictions: List[str],
                     text_targets: List[str],
                     questions: List[str],
                     text_modes: List[str],
                     stellar_preds_quantiles: Dict,
                     stellar_targets: Dict,
                     stellar_modes: List[str],
                     output_path: str):
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

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Predictions saved to {output_path}")
    if max_text_items:
        print(f"  - {max_text_items} text entries saved")
    if stellar_preds_quantiles:
        num_quantiles = results['metadata']['num_quantiles']
        print(f"  - {results['metadata']['num_stellar_samples']} stellar parameter predictions with {num_quantiles} quantiles each")

def plot_stellar_parameters(stellar_preds_quantiles: Dict,
                            stellar_targets: Dict,
                            output_dir: str,
                            modes: Optional[List[str]] = None):
    """Plot true vs predicted stellar parameters with confidence intervals, color-coded by mode"""
    
    if not stellar_preds_quantiles or not stellar_targets:
        print("No stellar parameters to plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
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
            plt.title(f'True vs Predicted {param} (CQR Calibrated)')
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
            plt.text(0.05, 0.95, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r_squared:.3f}\nCoverage: {coverage:.1%}\nN: {len(valid_targets)}',
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{param}_true_vs_predicted_cqr_calibrated.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Plot with CQR calibrated confidence intervals saved for {param}")
    
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
            axes[i].set_title(f'{param}')
            axes[i].grid(True, alpha=0.3)
            
            # Calculate and display metrics
            mae = np.mean(np.abs(valid_median_preds - valid_targets))
            rmse = np.sqrt(np.mean((valid_median_preds - valid_targets)**2))
            r_squared = np.corrcoef(valid_targets, valid_median_preds)[0, 1]**2
            coverage = np.mean((valid_targets >= valid_lower_preds) & (valid_targets <= valid_upper_preds))
            
            axes[i].text(0.05, 0.95, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r_squared:.3f}\nCov: {coverage:.1%}\nN: {len(valid_targets)}',
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            handles = [interval_handle, line_handle] + scatter_handles
            labels = ['1-sigma Confidence Interval', 'Perfect Prediction'] + scatter_labels
            axes[i].legend(handles, labels)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'stellar_parameters_combined_cqr_calibrated.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Combined plot with CQR calibrated confidence intervals saved")


def plot_training_fit_results(checkpoint_path: str, output_dir: str) -> None:
    """Plot aggregated training losses stored in fit_res.json."""
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
    print(f"✓ Training loss plot saved to {output_path}")


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
    args.single_sample_prob = 0.5
    args.mode = "combined"
    train_loader, val_loader, test_loader = create_datasets_and_loaders(args, device)

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
    # Generate text predictions separately
    print("\n" + "="*60)
    print("GENERATING TEXT PREDICTIONS")
    print("="*60)
    
    text_preds, text_targets, obsids_text, questions, text_modes = llm_predict_with_text_generation(
        trainer, test_loader, device, max_iter=3, generate_text=True, args=args
    )

    
    # Create CQR loss function (same as used in training)
    loss_fn = CQR(quantiles=quantiles, reduction='none')
    
    print("\n" + "="*60)
    print("BULK PREDICTIONS")
    print("="*60)
    print("len val/test/train dataloader: ", len(val_loader), len(test_loader), "N/A")

    
    # Step 1: Get validation predictions for calibration
    preds_val, targets_val, obsids_val, _ = llm_predict_stellar_bulk(trainer, val_loader, device, quantiles, max_iter=100)
    print(f"Validation predictions shape: {preds_val.shape}, targets: {targets_val.shape}")
    
    # Step 2: Get test predictions  
    preds, targets, obsids_stellar, stellar_modes = llm_predict_stellar_bulk(trainer, test_loader, device, quantiles)
    print(f"Test predictions shape: {preds.shape}, targets: {targets.shape}")
    
    # Step 3: Check coverage before calibration
    low_q = preds[:, :, 0]  # First quantile
    high_q = preds[:, :, -1]  # Last quantile
    print("before calibration - first sample predictions: ", low_q[0,0], high_q[0,0])
    coverage = np.mean((targets >= low_q) & (targets <= high_q))
    print('coverage before calibration: ', coverage)
    
    # Step 4: CQR Calibration (following APOGEE script exactly)
    print("\nApplying CQR calibration...")
    cqr_errs = loss_fn.calibrate(preds_val, targets_val)
    print(f"CQR errors shape: {cqr_errs.shape}")
    print(f"Targets shape: {targets.shape}, Preds shape: {preds.shape}")
    preds_cqr = loss_fn.predict(preds, cqr_errs)
    
    # Step 5: Check coverage after calibration
    low_q = preds_cqr[:, :, 0]
    high_q = preds_cqr[:, :, -1]
    print("after calibration shape: ", preds_cqr.shape)
    print("after calibration - first 10 samples low_q[:10,0]: ", low_q[:10,0])
    print("after calibration - first 10 samples high_q[:10,0]: ", high_q[:10,0])
    coverage = np.mean((targets >= low_q) & (targets <= high_q))
    print('coverage after calibration: ', coverage)
    
    
    # Verify sample alignment by checking obsids
    print(f"\n" + "="*60)
    print("VERIFYING SAMPLE ALIGNMENT")
    print("="*60)
    print(f"Stellar predictions: {len(obsids_stellar)} samples")
    print(f"Text predictions: {len(obsids_text)} samples")
    
    if len(obsids_stellar) == len(obsids_text):
        # Check if obsids match exactly
        mismatches = []
        for i, (stellar_obsid, text_obsid) in enumerate(zip(obsids_stellar, obsids_text)):
            if stellar_obsid != text_obsid:
                mismatches.append((i, stellar_obsid, text_obsid))
        
        if len(mismatches) == 0:
            print("✓ All obsids match perfectly - samples are aligned")
        else:
            print(f"⚠ Found {len(mismatches)} mismatched obsids:")
            for i, stellar_obsid, text_obsid in mismatches[:10]:  # Show first 10
                print(f"  Index {i}: stellar={stellar_obsid}, text={text_obsid}")
            if len(mismatches) > 10:
                print(f"  ... and {len(mismatches) - 10} more")
    else:
        print(f"⚠ Sample count mismatch: {len(obsids_stellar)} vs {len(obsids_text)}")
        
    print(f"First 5 stellar obsids: {obsids_stellar[:5]}")
    print(f"First 5 text obsids: {obsids_text[:5]}")

    if stellar_modes:
        print("Stellar mode distribution:", Counter(stellar_modes))
    if text_modes:
        print("Text mode distribution:", Counter(text_modes))
    
    # Step 7: Convert to format expected by plotting/saving functions
    # Unnormalize the calibrated predictions and convert to dict format
    stellar_preds_unnormalized = {}
    stellar_targets_unnormalized = {}
    
    # Unnormalize predictions (calibrated quantiles)
    for param_idx, param in enumerate(['Teff', 'logg', 'FeH']):
        if param == 'Teff':
            stellar_preds_unnormalized[param] = preds_cqr[:, param_idx, :] * (7500 - 3000) + 3000
            stellar_targets_unnormalized[param] = targets[:, param_idx] * (7500 - 3000) + 3000
        elif param == 'logg':
            stellar_preds_unnormalized[param] = preds_cqr[:, param_idx, :] * (5.0 - 0) + 0
            stellar_targets_unnormalized[param] = targets[:, param_idx] * (5.0 - 0) + 0
        elif param == 'FeH':
            stellar_preds_unnormalized[param] = preds_cqr[:, param_idx, :] * (0.5 - (-3)) + (-3)
            stellar_targets_unnormalized[param] = targets[:, param_idx] * (0.5 - (-3)) + (-3)
    
    # Use the calibrated predictions
    stellar_preds = stellar_preds_unnormalized
    stellar_targets = stellar_targets_unnormalized
    
    print("✓ CQR calibration completed! Confidence intervals are now properly calibrated.")
    print("Final shapes: ", {param: arr.shape for param, arr in stellar_preds.items()})
    # Save predictions
    if inference_args.save_predictions and stellar_preds is not None:
        print("saving predictions...")
        predictions_path = os.path.join(inference_args.output_dir, 'test_predictions.json')
        save_predictions(text_preds, text_targets, questions, text_modes, stellar_preds, stellar_targets, stellar_modes, predictions_path)
    
    # Generate plots
    if inference_args.plot_results:
        if stellar_preds is not None:
            plot_stellar_parameters(stellar_preds, stellar_targets, plots_dir, stellar_modes)

            # Generate consistency plots between stellar and text predictions
            if text_preds:
                consistency_dir = os.path.join(inference_args.output_dir, 'consistency_plots')
                plot_stellar_vs_text_consistency(stellar_preds, text_preds, consistency_dir)

        plot_training_fit_results(inference_args.checkpoint_path, plots_dir)

    interpolation_pairs = parse_interpolation_pairs(inference_args.interpolation_pairs)
    if not interpolation_pairs and getattr(inference_args, 'interpolation_num_pairs', 0) > 0:
        interpolation_pairs = select_random_interpolation_pairs(
            test_loader.dataset,
            trainer,
            inference_args.interpolation_num_pairs,
            inference_args.interpolation_seed,
            inference_args.interpolation_min_teff_diff,
        )
        if interpolation_pairs:
            print(f"Selected {len(interpolation_pairs)} random interpolation pairs (seed {inference_args.interpolation_seed}).")
        else:
            print("Warning: Unable to select interpolation pairs automatically.")
    if interpolation_pairs:
        interpolation_alphas = inference_args.interpolation_alphas or [i / 9 for i in range(10)]
        interpolation_dir = os.path.join(plots_dir, 'interpolation')
        interpolation_results = run_feature_interpolation(
            trainer=trainer,
            args=args,
            dataset=test_loader.dataset,
            collate_fn=getattr(test_loader, 'collate_fn', None),
            device=device,
            pairs=interpolation_pairs,
            alphas=interpolation_alphas,
            output_dir=interpolation_dir,
            quantiles=quantiles,
            min_teff_diff=inference_args.interpolation_min_teff_diff,
        )
        if inference_args.plot_results and interpolation_results:
            plot_interpolation_kiel(interpolation_results, interpolation_dir)
            plot_interpolation_kiel_combined(interpolation_results, interpolation_dir)

    print("✓ Inference completed successfully!")


if __name__ == '__main__':
    main()
