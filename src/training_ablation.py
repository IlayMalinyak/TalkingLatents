from __future__ import annotations

import os
import sys
import json
import copy
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Reuse the project’s existing pipeline pieces
from src.simple_questions_multitok import (
    ensure_backend_config,
    create_datasets_and_loaders,
    build_model_multitok,
    prepare_training_with_resume,
)
from src.simple_questions_multitok import parse_args as base_parse_args
from torch.nn.parallel import DistributedDataParallel as DDP
from src.simple_questions import setup


@dataclass
class AblationConfig:
    label: str
    json_variant: str  # 'long' | 'short'
    followup: bool
    feature_pred: str  # 'none' | 'pred_pair_random' | 'pred_pair_nn'


def make_base_args(output_dir: str) -> Any:
    """Create a minimal args object based on the main script’s parser defaults, then tweak small-run settings."""
    args = base_parse_args([])

    # Small/fast run defaults (override here for ablation)
    args.mode = 'single_star'                # Ablation focuses on single-star description QA
    args.batch_size = 8
    args.num_epochs = 2
    args.max_iter = 5000                      # Constrain iterations per epoch (trainer reshuffles each epoch)
    args.random_seed = 123
    args.num_spectral_features = 4
    args.max_seq_length = 128

    # Keep training stable but light
    args.use_amp = True
    args.gradient_checkpointing = True
    args.max_grad_norm = 1.0

    # Do not use classification/comparative in this ablation
    args.enable_classification = False
    args.disable_classification = True

    # Aux options default off; they will be configured per-run
    args.predict_stellar_params = False
    args.predict_features = False
    args.random_pairing = False
    args.followup_prob = 1.0   # use when followups enabled
    args.max_followup_turns = 1

    # Backend defaults (reuse existing infra; llama backend assumed available in this environment)
    args.llm_backend = 'llama'
    args.hf_model_name = None
    args.hf_quantization = 'none'

    # Paths
    args.output_dir = output_dir
    # Feature file (required if predict_features=True)
    args.features_file = '/data/TalkingLatents/logs/2025-07-29/features.npy'

    # Data json defaults; switched per ablation config
    args.json_file = '/data/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json'
    args.advanced_json_file = '/data/TalkingLatents/data/dataset/caption_advanced_100k.json' 

    return args


def apply_ablation_to_args(args: Any, cfg: AblationConfig) -> Any:
    args = copy.deepcopy(args)

    # only short targets with followup from long targets
    
    args.json_file = '/data/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json'
    args.followup_json_file = '/data/TalkingLatents/data/dataset/stellar_descriptions_questions.json'

    # Follow-up augmentation flag (applies during dataset creation for train/val/test)
    args.enable_followup_augmentation = bool(cfg.followup)

    # Feature prediction mode
    if cfg.feature_pred == 'none':
        args.predict_features = False
        args.random_pairing = False
    elif cfg.feature_pred == 'pred_pair_random':
        args.predict_features = True
        args.random_pairing = True
    elif cfg.feature_pred == 'pred_pair_nn':
        args.predict_features = True
        args.random_pairing = False
    else:
        raise ValueError(f"Unknown feature_pred mode: {cfg.feature_pred}")

    # Ensure features file is available when feature prediction is on
    if args.predict_features and (not args.features_file or not os.path.exists(args.features_file)):
        raise FileNotFoundError(
            f"Feature prediction requires features_file. Not found: {args.features_file}"
        )

    # Naming
    args.exp_name = f"ablation_{cfg.json_variant}_fup{int(cfg.followup)}_{cfg.feature_pred}"

    return args


def evaluate_with_followups_toggle(trainer, base_args: Any, device: int, epoch: int) -> Dict[str, Any]:
    """Run evaluation twice: (1) description-only, (2) with follow-up turns enabled.
    Returns metrics for both settings.
    """
    results: Dict[str, Any] = {}

    # Helper to rebuild val loader with a flag
    def _rebuild_val_loader(enable_followups: bool):
        tmp = copy.deepcopy(base_args)
        tmp.enable_followup_augmentation = enable_followups
        backend_cfg = ensure_backend_config(tmp)
        # Recreate only val loader (but function returns all three)
        train_dl, val_dl, _ = create_datasets_and_loaders(tmp, device=device, backend_config=backend_cfg)
        return val_dl

    # 1) Description-only evaluation (no followups)
    val_dl_desc = _rebuild_val_loader(False)
    trainer.val_dl = val_dl_desc
    try:
        trainer.evaluate_validation_samples(device, epoch, num_samples=5, max_new_tokens=50, temperature=0.2, top_p=0.8)
    except Exception as e:
        print(f"Warning: description-only evaluation failed: {e}")
    results['description_eval_done'] = True

    # 2) Follow-up evaluation (followups enabled)
    val_dl_fup = _rebuild_val_loader(True)
    trainer.val_dl = val_dl_fup
    try:
        trainer.evaluate_validation_samples(device, epoch, num_samples=5, max_new_tokens=50, temperature=0.2, top_p=0.8)
    except Exception as e:
        print(f"Warning: follow-up evaluation failed: {e}")
    results['followup_eval_done'] = True

    return results


def run_single_ablation(cfg: AblationConfig, base_args: Any, device: int = 0) -> Dict[str, Any]:
    print(f"\n=== Running ablation: {cfg.label} ===")
    args = apply_ablation_to_args(base_args, cfg)

    # Ensure output dir exists per run
    os.makedirs(args.output_dir, exist_ok=True)

    # Backend, datasets, model
    backend_config = ensure_backend_config(args)
    train_loader, val_loader, _ = create_datasets_and_loaders(args, device=device, backend_config=backend_config)
    model = build_model_multitok(args, device, world_size=1, backend_config=backend_config)

    # Resolve tokenizer for trainer (from loaders)
    if hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'tokenizer'):
        tokenizer = train_loader.dataset.tokenizer
    else:
        tokenizer = None

    # Load LoRA config via main script’s logic inside prepare_training_with_resume
    # and attach trainer; use resume disabled for ablations
    tuned_cfg_path = '/data/TalkingLatents/src/llm_config_tuned.json'
    with open(tuned_cfg_path, 'r') as f:
        tuned_cfg = json.load(f)
    lora_params = tuned_cfg.get('lora_params', {})
    optimizer, scheduler, scaler, trainer, start_epoch, initial_min_loss, initial_best_acc = prepare_training_with_resume(
        args=args,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        lora_params=lora_params,
        local_rank=device,
        world_size=1,
        backend_config=backend_config,
    )

    # Short training (max_iter + reshuffle handled inside Trainer)
    fit_res = trainer.fit(
        num_epochs=args.num_epochs,
        device=device,
        early_stopping=2,
        best='loss',
        start_epoch=start_epoch,
        initial_min_loss=initial_min_loss,
        initial_best_acc=initial_best_acc,
    )

    # Evaluate both on descriptions and follow-ups
    eval_res = evaluate_with_followups_toggle(trainer, args, device, epoch=args.num_epochs)

    # Gather summary
    summary = {
        'config': asdict(cfg),
        'fit_res_keys': list(fit_res.keys()) if isinstance(fit_res, dict) else [],
        'eval': eval_res,
        'exp_name': args.exp_name,
        'output_dir': args.output_dir,
    }
    return summary


def main():
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
    root_out = os.path.join('/data/TalkingLatents/logs', f'ablation_{timestamp}')
    os.makedirs(root_out, exist_ok=True)

    # Base args (small-run config)
    base_args = make_base_args(output_dir=root_out)

    # Define ablation grid
    grid: List[AblationConfig] = []
    for json_variant in ('short'):
        for followup in (False, True):
            for feature_pred in ('none', 'pred_pair_random', 'pred_pair_nn'):
                label = f"{json_variant}-fup{int(followup)}-{feature_pred}"
                grid.append(AblationConfig(label=label, json_variant=json_variant, followup=followup, feature_pred=feature_pred))

    all_results: List[Dict[str, Any]] = []

    local_rank, world_size, _ = setup()

    # Single-device ablations
    device = 0 if torch.cuda.is_available() else 'cpu'

    for cfg in grid:
        # try:
        res = run_single_ablation(cfg, base_args, device=local_rank if isinstance(local_rank, int) else 0)
        # except Exception as e:
        #     print(f"Ablation '{cfg.label}' failed: {e}")
        #     res = {'config': asdict(cfg), 'error': str(e)}
        all_results.append(res)

        # Save intermediate results
        with open(os.path.join(root_out, 'ablation_results_partial.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

    # Final save
    out_path = os.path.join(root_out, 'ablation_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved ablation summary to {out_path}")


if __name__ == '__main__':
    main()

