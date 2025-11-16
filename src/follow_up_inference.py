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
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from llama3.llama.tokenizer import Tokenizer  # noqa: E402
from src.follow_up_templates import (  # noqa: E402
    PHYSICAL_BOUNDS,
    PARAM_KEY_ALIASES,
    create_follow_up_specs,
)


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
    parser.add_argument('--max_new_tokens', type=int, default=64, help='Max tokens to sample per generation')
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
    parser.add_argument('--hf_model_name', type=str, default=None,
                        help='Override HF model name when llm_backend=hf/qwen.')
    parser.add_argument('--hf_revision', type=str, default=None,
                        help='Override HF revision.')
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
    return parser.parse_args()


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
        'hf_auth_token', 'llm_precision', 'gradient_checkpointing'
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
        model = build_model_multitok(args, device, world_size=1, backend_config=backend_config)
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


def create_follow_up_questions(params: Dict[str, Optional[float]], rng: random.Random) -> List[Dict[str, str]]:
    specs = create_follow_up_specs(params, rng, include_answers=False)
    return [{'type': spec['type'], 'question': spec['question']} for spec in specs]


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
            'input_spectra': batch_data['masked_spectra'][batch_idx:batch_idx+1],
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
    })
    return context, input_text, target_text


def append_text_to_prompt(context: Dict[str, Any], tokenizer: Tokenizer, text: str) -> None:
    """Append arbitrary text (tokenized) after the current model response."""
    if not text:
        return
    tokens = tokenizer.encode(text, bos=False, eos=False)
    if not tokens:
        return
    tensor = torch.tensor([tokens], dtype=context['prompt'].dtype, device=context['prompt'].device)
    context['prompt'] = torch.cat([context['prompt'], tensor], dim=1)


def _sample_top_p(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        return 0
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


def _decode_generation_requests(model: torch.nn.Module,
                                requests: List[Dict[str, Any]],
                                tokenizer: Tokenizer,
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
                               tokenizer: Tokenizer,
                               max_new_tokens: int,
                               temperature: float,
                               top_p: float) -> str:
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

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = forward_chunk(prompt, start_pos=0)
            logits = outputs['logits'][:, -1, :]
            next_token = _sample_top_p(logits.squeeze(0), temperature=temperature, top_p=top_p)
            gen_ids.append(next_token)
            next_tensor = torch.tensor([[next_token]], device=device, dtype=prompt.dtype)
            prompt = torch.cat([prompt, next_tensor], dim=1)
            if tokenizer is not None and hasattr(tokenizer, 'eos_id') and next_token == tokenizer.eos_id:
                break

    context['prompt'] = prompt
    _profile_record(f"generate_text_{mode}", profile_start, len(gen_ids))
    return tokenizer.decode(gen_ids) if tokenizer is not None else ''


def generate_text_for_group(model: torch.nn.Module,
                            contexts: Sequence[Dict[str, Any]],
                            tokenizer: Tokenizer,
                            max_new_tokens: int,
                            temperature: float,
                            top_p: float) -> List[str]:
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

    def _forward_chunk(chunk_tokens: torch.Tensor, start_pos: int) -> Dict[str, torch.Tensor]:
        if mode == "two_star":
            return base_model._forward_two_star_mode(
                input_ids=chunk_tokens,
                star_a_features=star_a_features,
                star_b_features=star_b_features,
                star_a_indices=star_a_indices,
                star_b_indices=star_b_indices,
                start_pos=start_pos,
                cache_rows=None,
                use_cache=False,
            )
        return base_model._forward_single_mode(
            input_ids=chunk_tokens,
            latent_features=latent_features,
            feature_start_indices=feature_start_idx,
            start_pos=start_pos,
            cache_rows=None,
            use_cache=False,
        )

    generated_tokens: List[List[int]] = [[] for _ in contexts]
    alive_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    eos_id = getattr(tokenizer, 'eos_id', None)
    pad_id = getattr(tokenizer, 'pad_id', 0)

    for _ in range(max_new_tokens):
        if not alive_mask.any():
            break
        outputs = _forward_chunk(prompt, start_pos=0)
        logits = outputs['logits'][:, -1, :]
        next_tokens = []
        for idx in range(batch_size):
            if not alive_mask[idx]:
                next_tokens.append(pad_id)
                continue
            sampled = _sample_top_p(logits[idx], temperature=temperature, top_p=top_p)
            next_tokens.append(sampled)
            generated_tokens[idx].append(sampled)
            if eos_id is not None and sampled == eos_id:
                alive_mask[idx] = False
        next_tensor = torch.tensor(next_tokens, device=device, dtype=prompt.dtype).unsqueeze(1)
        prompt = torch.cat([prompt, next_tensor], dim=1)

    decoded = [tokenizer.decode(tokens) if tokenizer is not None else '' for tokens in generated_tokens]

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
    _, tokenizer_path = get_model_path(args)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    print(f"[OK] Loaded tokenizer from {tokenizer_path}")

    _, _, test_loader = create_datasets_and_loaders(args, device, backend_config)
    print(f"Test loader has {len(test_loader.dataset)} samples")

    results: List[Dict[str, Any]] = []
    processed = 0

    for batch in tqdm(test_loader, desc="Running follow-up inference"):
        move_batch_to_device(batch, device)
        batch_size = batch['input_ids'].shape[0]
        batch_modes = infer_batch_modes(batch, batch_size)
        sample_entries: List[Dict[str, Any]] = []
        obsids = batch.get('obsids', [])
        target_texts = batch.get('target_texts', [])

        for sample_idx in range(batch_size):
            if cli_args.max_samples is not None and processed + len(sample_entries) >= cli_args.max_samples:
                break

            context, dataset_question, dataset_target = build_generation_context(
                batch_data=batch,
                batch_idx=sample_idx,
            )
            stellar_params = get_stellar_params(batch, sample_idx)
            follow_up_specs = create_follow_up_questions(stellar_params, seed_rng)
            obsid_value = obsids[sample_idx] if sample_idx < len(obsids) else None
            target_value = target_texts[sample_idx] if sample_idx < len(target_texts) else dataset_target

            sample_entries.append({
                'context': context,
                'dataset_question': dataset_question,
                'dataset_target': target_value,
                'stellar_params': stellar_params,
                'follow_up_specs': follow_up_specs,
                'obsid': obsid_value,
                'mode_label': batch_modes[sample_idx] if sample_idx < len(batch_modes) else 'single_star',
                'batch_ref': batch,
                'batch_sample_idx': sample_idx,
            })

        if not sample_entries:
            if cli_args.max_samples is not None and processed >= cli_args.max_samples:
                break
            continue

        for entry in sample_entries:
            base_text, _, _, _ = model.generate_response_from_batch(
                batch_data=entry['batch_ref'],
                batch_idx=entry['batch_sample_idx'],
                tokenizer=tokenizer,
                max_new_tokens=cli_args.max_new_tokens,
                temperature=cli_args.temperature,
                top_p=cli_args.top_p,
            )
            entry['base_answer'] = base_text
            entry.pop('batch_ref', None)
            entry.pop('batch_sample_idx', None)
            if tokenizer is not None:
                gen_tokens = tokenizer.encode(base_text, bos=False, eos=False)
                if gen_tokens:
                    tensor = torch.tensor([gen_tokens],
                                          dtype=entry['context']['prompt'].dtype,
                                          device=entry['context']['prompt'].device)
                    entry['context']['prompt'] = torch.cat([entry['context']['prompt'], tensor], dim=1)

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
                        entry['follow_up_answers'].append({
                            'type': spec.get('type', 'follow_up'),
                            'question': question_text,
                            'answer': answer,
                        })
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

        stop_processing = False
        for entry_idx, entry in enumerate(sample_entries):
            entry['context'].pop('prepared_inputs', None)

            results.append({
                'sample_index': processed,
                'obsid': entry['obsid'],
                'mode': entry['mode_label'],
                'dataset_question': entry['dataset_question'],
                'dataset_target_answer': entry['dataset_target'],
                'model_answer': entry['base_answer'],
                'stellar_params': {k: sanitize_for_json(v) for k, v in entry['stellar_params'].items()},
                'follow_up_answers': entry['follow_up_answers'],
            })
            processed += 1
            if cli_args.max_samples is not None and processed >= cli_args.max_samples:
                stop_processing = True
                remaining = sample_entries[entry_idx + 1:]
                for pending in remaining:
                    pending['context'].pop('prepared_inputs', None)
                break

        if stop_processing:
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
