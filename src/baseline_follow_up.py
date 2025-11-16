#!/usr/bin/env python3
"""Baseline follow-up generation using the pretrained (un-finetuned) multimodal model."""

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import sys

import torch
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.simple_questions import get_model_path, setup  # noqa: E402
from src.simple_questions_multitok import (  # noqa: E402
    create_datasets_and_loaders,
    parse_args as training_parse_args,
    ensure_backend_config,
    build_model_multitok,
)
from src.tokenizer_adapter import load_tokenizer_adapter  # noqa: E402
import src.follow_up_inference as follow_up  # noqa: E402


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate follow-up questions with the baseline pretrained LLM (no fine-tuned checkpoint)."
    )
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Optional trained checkpoint path (used only to locate training_config.json).')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Explicit training config JSON. Required if --checkpoint_path is not provided.')
    parser.add_argument('--output_dir', type=str, default='baseline_follow_up_outputs',
                        help='Directory (relative to checkpoint dir when provided) for JSON output')
    parser.add_argument('--json_filename', type=str, default='baseline_follow_up_answers.json',
                        help='Filename for the saved JSON payload')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit the number of test samples to process')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for dataloader (overrides config)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible follow-up prompts')
    parser.add_argument('--max_new_tokens', type=int, default=64, help='Max tokens to sample per generation')
    parser.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature for text generation')
    parser.add_argument('--top_p', type=float, default=0.8, help='Top-p nucleus sampling value')
    parser.add_argument('--profile_timings', action='store_true', help='Enable lightweight CPU timing profiler')
    parser.add_argument('--profile_output', type=str, default=None, help='Optional JSON file to dump timing stats')
    parser.add_argument('--max_decode_group', type=int, default=2,
                        help='Maximum number of samples to decode in parallel')
    parser.add_argument('--max_decode_tokens_per_group', type=int, default=4096,
                        help='Upper bound on prompt_length * group_size before forcing sequential decoding')
    parser.add_argument('--max_iter', type=int, default=None,
                        help='Truncate generation after this many dataloader iterations (batches)')
    parser.add_argument('--predict_features', action=argparse.BooleanOptionalAction, default=None,
                        help='Use feature-prediction dataset/mode instead of QA dataset')
    parser.add_argument('--llm_backend', choices=['llama', 'hf', 'qwen'], default=None,
                        help='Override the LLM backend (defaults to the training config).')
    parser.add_argument('--hf_model_name', '--qwen_model_name', dest='hf_model_name', type=str, default=None,
                        help='HF/Qwen model identifier (only when llm_backend=hf/qwen).')
    parser.add_argument('--hf_quantization', '--qwen_quantization', dest='hf_quantization',
                        choices=['none', '8bit', '4bit', 'fp16'], default=None,
                        help='Quantization mode for the HF/Qwen backbone.')
    parser.add_argument('--hf_device_map', '--qwen_device_map', dest='hf_device_map', type=str, default=None,
                        help='Device map passed to AutoModelForCausalLM.from_pretrained')
    parser.add_argument('--hf_max_memory_gb', '--qwen_max_memory_gb', dest='hf_max_memory_gb',
                        type=float, default=None,
                        help='Per-device memory cap forwarded to the HF loader')
    parser.add_argument('--hf_trust_remote_code', '--qwen_trust_remote_code', dest='hf_trust_remote_code',
                        action=argparse.BooleanOptionalAction, default=None,
                        help='Set/clear --trust-remote-code when loading HF/Qwen models')
    parser.add_argument('--hf_revision', '--qwen_revision', dest='hf_revision', type=str, default=None,
                        help='Specific HF revision/commit to fetch (optional)')
    return parser.parse_args()


def resolve_config(cli_args: argparse.Namespace) -> Tuple[Dict[str, Any], str]:
    if cli_args.config_path:
        with open(cli_args.config_path, 'r') as f:
            return json.load(f), cli_args.config_path
    if not cli_args.checkpoint_path:
        default_training_args = training_parse_args([])
        print("[INFO] No checkpoint/config provided; using default training arguments.")
        return vars(default_training_args), '__defaults_simple_questions_multitok__'
    config = follow_up.load_config_from_checkpoint_dir(cli_args.checkpoint_path)
    config_path = os.path.join(os.path.dirname(cli_args.checkpoint_path), 'training_config.json')
    return config, config_path


def resolve_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda', local_rank)
    return torch.device('cpu')


def determine_output_path(cli_args: argparse.Namespace) -> Tuple[str, str]:
    base = os.path.dirname(cli_args.checkpoint_path) if cli_args.checkpoint_path else os.getcwd()
    output_dir = cli_args.output_dir if os.path.isabs(cli_args.output_dir) else os.path.join(base, cli_args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, os.path.join(output_dir, cli_args.json_filename)


def apply_backend_overrides(base_args: argparse.Namespace, cli_args: argparse.Namespace) -> None:
    override_fields = [
        'llm_backend', 'hf_model_name', 'hf_quantization', 'hf_device_map',
        'hf_max_memory_gb', 'hf_trust_remote_code', 'hf_revision'
    ]
    for field in override_fields:
        value = getattr(cli_args, field, None)
        if value is not None:
            setattr(base_args, field, value)
    if getattr(base_args, 'hf_quantization', None) == 'fp16':
        base_args.hf_quantization = 'none'


def build_baseline_model(args: argparse.Namespace, device: torch.device):
    backend_config = ensure_backend_config(args)
    args.gradient_checkpointing = getattr(args, 'gradient_checkpointing', False)
    model = build_model_multitok(args, device, world_size=1, backend_config=backend_config)
    model.eval()
    return model, backend_config


def evaluate_follow_up_baseline(model: torch.nn.Module,
                                tokenizer,
                                test_loader,
                                args: argparse.Namespace,
                                cli_args: argparse.Namespace,
                                output_path: str,
                                config_path: str,
                                backend_info: Dict[str, Any]) -> None:
    seed_rng = follow_up.seed_everything(cli_args.seed)
    follow_up.ACTIVE_PROFILER = follow_up.InferenceProfiler() if cli_args.profile_timings else None

    results: List[Dict[str, Any]] = []
    processed = 0
    model_device = next(model.parameters()).device
    print(f"Evaluating on device: {model_device}")

    iter_limit = cli_args.max_iter
    iter_count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running baseline follow-up inference"):
            if iter_limit is not None and iter_count >= iter_limit:
                print(f"[INFO] Reached max_iter={iter_limit}; stopping generation early.")
                break
            follow_up.move_batch_to_device(batch, model_device)
            batch_size = batch['input_ids'].shape[0]
            batch_modes = follow_up.infer_batch_modes(batch, batch_size)
            sample_entries: List[Dict[str, Any]] = []
            obsids = batch.get('obsids', [])
            target_texts = batch.get('target_texts', [])

            for sample_idx in range(batch_size):
                if cli_args.max_samples is not None and processed + len(sample_entries) >= cli_args.max_samples:
                    break

                context, dataset_question, dataset_target = follow_up.build_generation_context(
                    batch_data=batch,
                    batch_idx=sample_idx,
                )
                stellar_params = follow_up.get_stellar_params(batch, sample_idx)
                follow_up_specs = follow_up.create_follow_up_questions(stellar_params, seed_rng)
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
                        tensor = torch.tensor(
                            [gen_tokens],
                            dtype=entry['context']['prompt'].dtype,
                            device=entry['context']['prompt'].device,
                        )
                        entry['context']['prompt'] = torch.cat([entry['context']['prompt'], tensor], dim=1)

            for entry in sample_entries:
                entry['follow_up_answers'] = []
                follow_up.append_text_to_prompt(entry['context'], tokenizer, "\n")

            max_follow_ups = max((len(entry['follow_up_specs']) for entry in sample_entries), default=0)
            for round_idx in range(max_follow_ups):
                round_requests: List[Dict[str, Any]] = []
                for entry in sample_entries:
                    if round_idx >= len(entry['follow_up_specs']):
                        continue
                    spec = entry['follow_up_specs'][round_idx]
                    question_text = spec['question'].strip()
                    follow_up.append_text_to_prompt(
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
                            follow_up.append_text_to_prompt(entry['context'], tokenizer, "\n")
                        return _assign

                    round_requests.append({
                        'context': entry['context'],
                        'on_answer': _make_on_answer(),
                    })

                follow_up._decode_generation_requests(
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
                    'stellar_params': {k: follow_up.sanitize_for_json(v) for k, v in entry['stellar_params'].items()},
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

            iter_count += 1

    metadata = {
        'timestamp': datetime.utcnow().isoformat(),
        'reference_checkpoint_path': cli_args.checkpoint_path,
        'config_path': config_path,
        'num_samples': processed,
        'max_samples': cli_args.max_samples,
        'generation': {
            'max_new_tokens': cli_args.max_new_tokens,
            'temperature': cli_args.temperature,
            'top_p': cli_args.top_p,
        },
        'backend': backend_info,
    }

    payload = {
        'metadata': metadata,
        'samples': follow_up.sanitize_for_json(results),
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved baseline follow-up generations for {processed} samples to {output_path}")

    if follow_up.ACTIVE_PROFILER is not None:
        summary = follow_up.ACTIVE_PROFILER.summary()
        print("[PROFILE] Timing summary:")
        for label, data in summary.items():
            print(f"  - {label}: {data['seconds']:.2f}s over {data['calls']} calls "
                  f"(avg {data['avg_seconds']:.3f}s, {data['tokens_per_second']:.1f} tok/s)")
        if cli_args.profile_output:
            profile_path = cli_args.profile_output
            if not os.path.isabs(profile_path):
                profile_path = os.path.join(os.path.dirname(output_path), profile_path)
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            print(f"[PROFILE] Saved timing summary to {profile_path}")


def main() -> None:
    cli_args = parse_cli_args()
    config, config_path = resolve_config(cli_args)
    args = follow_up.create_args_from_config(config, cli_args)
    apply_backend_overrides(args, cli_args)

    local_rank, _, _ = setup()
    device = resolve_device(local_rank)
    print(f"Using device: {device}")

    output_dir, output_path = determine_output_path(cli_args)

    model, backend_config = build_baseline_model(args, device)
    tokenizer_backend = backend_config.tokenizer_backend
    tokenizer_path = backend_config.tokenizer_path
    if tokenizer_backend == 'llama' and not tokenizer_path:
        _, tokenizer_path = get_model_path(args)
    padding_side = 'left' if tokenizer_backend != 'llama' else 'right'
    tokenizer = load_tokenizer_adapter(
        backend=tokenizer_backend,
        tokenizer_path=tokenizer_path,
        hf_model_name=backend_config.model_name_or_path if tokenizer_backend != 'llama' else None,
        trust_remote_code=backend_config.trust_remote_code,
        hf_revision=backend_config.revision,
        padding_side=padding_side,
    )
    print(f"[OK] Loaded tokenizer for backend '{tokenizer_backend}'")

    _, _, test_loader = create_datasets_and_loaders(args, device, backend_config)
    print(f"Test loader has {len(test_loader.dataset)} samples")

    evaluate_follow_up_baseline(
        model=model,
        tokenizer=tokenizer,
        test_loader=test_loader,
        args=args,
        cli_args=cli_args,
        output_path=output_path,
        config_path=config_path,
        backend_info=backend_config.to_dict(),
    )


if __name__ == '__main__':
    main()
