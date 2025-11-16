#!/usr/bin/env python3
"""
Evaluate a trained LateFusionModel by running base descriptions and synthetic
follow-up questions, recording the spectral reconstructions produced at each turn.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from data.dataset_late_fusion import create_late_fusion_dataloaders
from llama3.llama.tokenizer import Tokenizer
from nn.late_fusion import LateFusionModel
from src.follow_up_templates import (
    PARAM_KEY_ALIASES,
    create_follow_up_specs,
)
from src.late_fusion import (
    apply_precision,
    build_llm_args,
    deep_update,
    load_model_config,
    load_yaml_config,
    maybe_load_spectral_model,
    resolve_path,
    seed_everything,
)
from src.simple_questions import _load_llm_model


def ensure_distributed_environment() -> None:
    """
    Ensure torch.distributed and fairscale model-parallel groups are initialized.
    Required because LLaMA uses VocabParallelEmbedding even for single-GPU runs.
    """
    if not dist.is_available():
        return

    if not dist.is_initialized():
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        backend = "nccl" if torch.cuda.is_available() else "gloo"

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "12910")

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.cuda.set_device(local_rank % torch.cuda.device_count())

        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
        if rank == 0:
            print(f"✓ Initialized torch.distributed (backend={backend}, world_size={world_size})")

    try:
        import fairscale.nn.model_parallel.initialize as fs_init
    except ImportError:
        return

    if not fs_init.model_parallel_is_initialized():
        fs_init.initialize_model_parallel(1)
        if dist.get_rank() == 0:
            print("✓ Initialized fairscale model-parallel group (size=1)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Late-fusion follow-up inference")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config used for training (e.g., configs/late_fusion.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a checkpoint produced by LateFusionTrainer",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="late_fusion_followups.json",
        help="Where to store the JSON report",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="Maximum number of test samples to evaluate",
    )
    parser.add_argument(
        "--max_followups",
        type=int,
        default=2,
        help="Maximum follow-up turns to synthesize per sample",
    )
    parser.add_argument(
        "--followup_seed",
        type=int,
        default=42,
        help="Seed controlling random follow-up selection",
    )
    parser.add_argument(
        "--append_answer_prompt",
        action="store_true",
        help="Append '\\nAnswer:' after each question before re-running LateFusion",
    )
    return parser.parse_args()


def load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading checkpoint: {missing[:5]}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint: {unexpected[:5]}")


def build_tokenizer(data_cfg: Dict[str, Any]) -> Tokenizer:
    tok_path = data_cfg.get("tokenizer_path")
    if not tok_path:
        raise ValueError("data.tokenizer_path must be specified in the config for inference.")
    tok_path = resolve_path(tok_path)
    if tok_path is None or not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer path '{data_cfg.get('tokenizer_path')}' not found.")
    print(f"✓ Loaded tokenizer from {tok_path}")
    return Tokenizer(model_path=str(tok_path))


def encode_text(
    tokenizer: Tokenizer,
    text: str,
    max_length: int,
) -> torch.Tensor:
    pad_id = tokenizer.pad_id if hasattr(tokenizer, "pad_id") else 0
    token_ids = tokenizer.encode(text, bos=True, eos=False)
    token_ids = token_ids[:max_length]
    if len(token_ids) < max_length:
        token_ids += [pad_id] * (max_length - len(token_ids))
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)


def extract_params(stellar_data: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    if not isinstance(stellar_data, dict):
        return {}
    params: Dict[str, Optional[float]] = {}
    for param, aliases in PARAM_KEY_ALIASES.items():
        value = None
        for key in aliases:
            raw_val = stellar_data.get(key)
            if raw_val is None:
                continue
            try:
                value = float(raw_val)
                break
            except (TypeError, ValueError):
                value = None
        params[param] = value
    return params


def run_late_fusion(
    model: LateFusionModel,
    input_ids: torch.Tensor,
    spectral_data: torch.Tensor,
    masked_spectra: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    with torch.no_grad():
        batch = {
            "input_ids": input_ids.to(device),
            "attention_mask": (input_ids != 0).long().to(device),
            "spectral_data": spectral_data.to(device),
            "masked_spectra": masked_spectra.to(device),
        }
        outputs = model(batch)
    return {
        "spectral_prediction": outputs["spectral_reconstruction"][0].detach().cpu().tolist(),
        "prefix_embeddings": outputs.get("prefix_embeddings", torch.zeros(1)).detach().cpu().tolist(),
    }


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config file '{args.config}' not found.")
    config = load_yaml_config(config_path)

    exp_cfg = config.get("experiment", {})
    seed = exp_cfg.get("seed", 42)
    seed_everything(seed)
    print(f"✓ Seeded everything with {seed}")

    data_cfg = config.get("data", {})
    train_loader, val_loader, test_loader = create_late_fusion_dataloaders(
        json_file=data_cfg["json_file"],
        batch_size=data_cfg.get("batch_size", 4),
        train_ratio=data_cfg.get("train_ratio", 0.7),
        val_ratio=data_cfg.get("val_ratio", 0.15),
        test_ratio=data_cfg.get("test_ratio", 0.15),
        random_state=data_cfg.get("random_state", 42),
        num_workers=data_cfg.get("num_workers", 0),
        cache_dir=data_cfg.get("cache_dir"),
        world_size=1,
        tokenizer_path=data_cfg.get("tokenizer_path"),
        max_length=data_cfg.get("max_length", 512),
    )
    del train_loader, val_loader  # Only test loader is needed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ensure_distributed_environment()

    llm_args = build_llm_args(config.get("llm", {}), data_cfg)
    llm_model = _load_llm_model(llm_args)
    apply_precision(llm_model, config.get("llm", {}).get("llm_precision", "fp16"))
    llm_model = llm_model.to(device)

    spectral_model = maybe_load_spectral_model(config.get("spectral_model", {}))
    if spectral_model is not None:
        spectral_model = spectral_model.to(device)

    perceiver_cfg = load_model_config(config.get("model", {}))
    model = LateFusionModel(llm_model, spectral_model, perceiver_cfg).to(device)
    checkpoint_path = resolve_path(args.checkpoint)
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint '{args.checkpoint}' not found.")
    load_checkpoint_weights(model, checkpoint_path)
    model.eval()

    tokenizer = build_tokenizer(data_cfg)
    max_length = data_cfg.get("max_length", 512)

    rng = random.Random(args.followup_seed)

    results: List[Dict[str, Any]] = []
    processed = 0
    max_samples = args.max_samples if args.max_samples is not None else math.inf
    for batch in tqdm(test_loader, desc="Running late-fusion follow-ups"):
        batch_size = batch["input_ids"].size(0)
        for idx in range(batch_size):
            if processed >= max_samples:
                break

            base_text = batch["texts"][idx]
            spectral_data = batch["spectral_data"][idx : idx + 1]
            masked_spectra = batch["masked_spectra"][idx : idx + 1]
            stellar_data = batch["stellar_data"][idx]

            base_ids = batch["input_ids"][idx : idx + 1]
            base_prediction = run_late_fusion(
                model,
                base_ids,
                spectral_data,
                masked_spectra,
                device,
            )

            star_params = extract_params(stellar_data)
            followup_specs = create_follow_up_specs(
                star_params,
                rng,
                max_pairs=args.max_followups,
                include_answers=True,
            )

            followup_turns: List[Dict[str, Any]] = []
            conversation_text = base_text
            for spec in followup_specs:
                question = spec["question"]
                appended = f"{conversation_text}\nFollow-up question: {question}"
                if args.append_answer_prompt:
                    appended += "\nAnswer:"
                follow_ids = encode_text(tokenizer, appended, max_length)
                follow_pred = run_late_fusion(
                    model,
                    follow_ids,
                    spectral_data,
                    masked_spectra,
                    device,
                )
                followup_turns.append(
                    {
                        "question": question,
                        "template_answer": spec.get("answer"),
                        "spectral_prediction": follow_pred["spectral_prediction"],
                    }
                )
                conversation_text = appended

            sample_result = {
                "sample_index": processed,
                "obsid": batch["obsids"][idx],
                "base_text": base_text,
                "base_prediction": base_prediction["spectral_prediction"],
                "follow_up_turns": followup_turns,
            }
            results.append(sample_result)
            processed += 1
        if processed >= max_samples:
            break

    output_payload = {
        "metadata": {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "max_samples": args.max_samples,
            "max_followups": args.max_followups,
        },
        "samples": results,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2)
    print(f"✓ Saved follow-up spectral predictions to {output_path}")


if __name__ == "__main__":
    main()
