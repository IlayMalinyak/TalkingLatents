#!/usr/bin/env python3
"""
Entry point for running a LateFusionModel experiment.
Most knobs are configured via a YAML file; only a few essentials stay on the CLI.
"""

import argparse
import json
import os
import random
import sys
import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, List

import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

# Ensure project root on path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
os.system('pip install tiktoken fairscale fire blobfile torchdiffeq torchcfm transformers bitsandbytes accelerate')


from data.dataset_diverse import create_diverse_dataloaders
from data.dataset_hybrid import create_hybrid_dataloaders
from nn.late_fusion import LateFusionModel
from nn.train import LateFusionTrainer
from src.simple_questions import setup, _load_llm_model 
from src.tokenizer_adapter import load_tokenizer_adapter
from nn.spectra_model import MultiTaskRegressor 
from data.transforms import GeneralSpectrumPreprocessor, ToTensor, Compose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a late-fusion experiment.")
    parser.add_argument("--config", type=str, default="configs/late_fusion.yaml", help="Path to YAML config.")
    parser.add_argument("--exp-name", type=str, default=None, help="Override experiment name from config.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override base output directory defined in the config.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint created by LateFusionTrainer._save_all_checkpoints (optional).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip training after building all components.")
    return parser.parse_args()


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def resolve_path(path_like: Optional[str]) -> Optional[Path]:
    if path_like is None:
        return None
    candidate = Path(path_like)
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    return candidate


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


def prepare_output_directory(base_dir: Path, exp_name: str) -> Path:
    exp_dir = base_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def build_llm_args(llm_cfg: Dict[str, Any], data_cfg: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        llm_root=llm_cfg.get("llm_root"),
        llm_model=llm_cfg.get("llm_model"),
        llm_path=llm_cfg.get("llm_path"),
        batch_size=data_cfg.get("batch_size", 4),
        max_seq_length=data_cfg.get("max_length", 512),
    )


def maybe_load_spectral_model(cfg: Dict[str, Any]):
    if not cfg.get("enabled", False):
        return None
    
    print("Loading spectral model (local implementation)...")
    # Constants from src/simple_questions.py
    SPECTRA_CONFIG_PATH = "/home/ilay.kamai/work/DESA/logs/spec_decode2_2025-02-16/MultiTaskRegressor_spectra__decode_4_complete_config.yaml"
    SPECTRA_WEIGHTS_PATH = "/home/ilay.kamai/work/DESA/logs/spec_decode2_2025-02-16/MultiTaskRegressor_spectra_decode_4.pth"
    
    with open(SPECTRA_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        
    spec_model_args = SimpleNamespace(**config['model_args'])
    conformer_args = SimpleNamespace(**config['conformer_args'])
    
    model = MultiTaskRegressor(spec_model_args, conformer_args)
    
    if os.path.exists(SPECTRA_WEIGHTS_PATH):
        print(f"Loading pretrained weights from {SPECTRA_WEIGHTS_PATH}")
        checkpoint = torch.load(SPECTRA_WEIGHTS_PATH, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        # Remove 'module.' prefix from DDP checkpoints
        state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
    else:
        print(f"Warning: Spectral weights not found at {SPECTRA_WEIGHTS_PATH}")
        
    return model


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


def build_optimizer(model: torch.nn.Module, optim_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    opt_type = (optim_cfg.get("type") or "adamw").lower()
    lr = optim_cfg.get("lr", 1e-4)
    weight_decay = optim_cfg.get("weight_decay", 0.0)
    params = [p for p in model.parameters() if p.requires_grad]
    if opt_type == "adamw":
        betas = tuple(optim_cfg.get("betas", (0.9, 0.999)))
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if opt_type == "adam":
        betas = tuple(optim_cfg.get("betas", (0.9, 0.999)))
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if opt_type == "sgd":
        momentum = optim_cfg.get("momentum", 0.9)
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if opt_type in ["adamw8bit", "paged_adamw_8bit"]:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit optimizers.")
        betas = tuple(optim_cfg.get("betas", (0.9, 0.999)))
        print(f"Using {opt_type} optimizer from bitsandbytes...")
        if opt_type == "paged_adamw_8bit":
             return bnb.optim.PagedAdamW8bit(params, lr=lr, betas=betas, weight_decay=weight_decay)
        return bnb.optim.AdamW8bit(params, lr=lr, betas=betas, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer type: {opt_type}")


def build_scheduler(optimizer: torch.optim.Optimizer, sched_cfg: Optional[Dict[str, Any]]):
    if not sched_cfg:
        return None
    sched_type = (sched_cfg.get("type") or "").lower()
    if sched_type == "cosine":
        t_max = sched_cfg.get("t_max", 10)
        eta_min = sched_cfg.get("min_lr", 0.0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    if sched_type == "step":
        step_size = sched_cfg.get("step_size", 1)
        gamma = sched_cfg.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if sched_type == "constant":
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=sched_cfg.get("factor", 1.0), total_iters=sched_cfg.get("total_iters", 1)
        )
    raise ValueError(f"Unsupported scheduler type: {sched_type}")


def build_dataloaders(data_cfg: Dict[str, Any], llm_cfg: Dict[str, Any], world_size: int):
    # Setup transforms for raw spectral data
    spectral_transforms = Compose([
        GeneralSpectrumPreprocessor(),
        ToTensor() 
    ])

    dataset_kwargs = dict(data_cfg.get("dataset_kwargs") or {})
    dataset_kwargs["spectral_transforms"] = spectral_transforms
    
    cache_dir = resolve_path(data_cfg.get("cache_dir"))
    if cache_dir:
        dataset_kwargs["cache_dir"] = str(cache_dir)

    # Cycle-fusion follow-up controls (optional)
    if data_cfg.get("use_followups") is not None:
        dataset_kwargs["enable_followup"] = bool(data_cfg.get("use_followups"))
    if data_cfg.get("followup_prob") is not None:
        dataset_kwargs["followup_prob"] = float(data_cfg.get("followup_prob"))
    if data_cfg.get("max_followups") is not None:
        dataset_kwargs["max_followup_turns"] = int(data_cfg.get("max_followups"))
    
    # Construct tokenizer args from LLM config and Data Config
    llm_backend = llm_cfg.get("backend", "llama")
    # Prefer tokenizer path given in data config, fallback to llm config
    tokenizer_path = resolve_path(data_cfg.get("tokenizer_path") or llm_cfg.get("tokenizer_path"))
    
    tokenizer_adapter = load_tokenizer_adapter(
        backend=llm_backend,
        tokenizer_path=str(tokenizer_path) if tokenizer_path else None,
        hf_model_name=llm_cfg.get("llm_model") if llm_backend != 'llama' else None,
    )
    
    # Features loading - not using precomputed features with GeneralSpectrumPreprocessor
    features_file = resolve_path(data_cfg.get("features_file"))
    if features_file and features_file.exists():
        print(f"Warning: 'features_file' is specified ({features_file}), but GeneralSpectrumPreprocessor is used. "
              "Pre-computed features will be ignored in favor of raw spectra processing.")

    print(f"DEBUG: dataset_kwargs passed to loader: {dataset_kwargs}")
    dataset_type = data_cfg.get("dataset_type", "diverse")
    print(f"DEBUG: Dataset type: {dataset_type}")
    
    
    loader_fn = create_hybrid_dataloaders
        
    train_loader, val_loader, test_loader = loader_fn(
        json_file=str(resolve_path(data_cfg["json_file"])),
        features_array=None, # Not using precomputed features
        batch_size=data_cfg.get("batch_size", 4),
        train_ratio=data_cfg.get("train_ratio", 0.7),
        val_ratio=data_cfg.get("val_ratio", 0.15),
        test_ratio=data_cfg.get("test_ratio", 0.15),
        random_state=data_cfg.get("random_state", 42),
        num_workers=data_cfg.get("num_workers", 2),
        world_size=world_size,
        max_length=data_cfg.get("max_length", 256),
        tokenizer=tokenizer_adapter,
        tokenizer_backend=llm_cfg.get("llm_model", "Llama3.1-8B"), # Basic string or inferred
        **dataset_kwargs
    )
    return train_loader, val_loader, test_loader


def save_runtime_config(exp_dir: Path, config: Dict[str, Any], args: argparse.Namespace) -> None:
    snapshot = {
        "config": config,
        "cli": vars(args),
    }
    output_path = exp_dir / "config_runtime.yaml"
    with open(output_path, "w") as handle:
        yaml.safe_dump(snapshot, handle)
    print(f"✓ Saved runtime config to {output_path}")


def resume_from_checkpoint(trainer: LateFusionTrainer, resume_path: str, device: torch.device) -> Dict[str, Any]:
    print(f"Loading checkpoint from {resume_path} to CPU with mmap...")
    try:
        # Load with mmap to reduce RAM usage, map to CPU
        ckpt = torch.load(resume_path, map_location='cpu', mmap=True)
    except (TypeError, AttributeError):
        print("mmap=True not supported/failed, falling back to standard CPU load")
        ckpt = torch.load(resume_path, map_location='cpu')

    state = ckpt.get("model_state_dict", ckpt)
    trainer._unwrap_model().load_state_dict(state, strict=False)
    del state  # Free reference to state code
    
    if trainer.optimizer is not None and ckpt.get("optimizer"):
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
    if trainer.scheduler is not None and ckpt.get("scheduler"):
        trainer.scheduler.load_state_dict(ckpt["scheduler"])
    if trainer.scaler is not None and ckpt.get("scaler"):
        trainer.scaler.load_state_dict(ckpt["scaler"])
        # IMPORTANT: Reset the scaler's loss scale to a safe value
        # A very large scale from checkpoint can cause numerical instability
        current_scale = trainer.scaler.get_scale()
        if current_scale > 2**16:  # Scale is too large, reset to safe value
            print(f"Warning: GradScaler scale was {current_scale:.0f}, resetting to 2^16=65536 for stability")
            trainer.scaler._scale.fill_(2**16)
        elif current_scale < 1.0:  # Scale is too small
            print(f"Warning: GradScaler scale was {current_scale:.6f}, resetting to 2^16=65536")
            trainer.scaler._scale.fill_(2**16)
    
    print(f"✓ Resumed weights from {resume_path}")
    
    # Extract metadata to return, then delete ckpt to free memory
    resume_info = {
        "epoch": ckpt.get("epoch", 0),
        "min_loss": ckpt.get("min_loss"),
        "best_acc": ckpt.get("best_acc"),
    }
    del ckpt
    return resume_info


def main():
    # Turning off anomaly detection to allow GradScaler to handle Infs/NaNs naturally
    torch.autograd.set_detect_anomaly(False)
    args = parse_args()
    config_path = resolve_path(args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config file '{args.config}' not found.")
    config = load_yaml_config(config_path)

    exp_cfg = config.get("experiment", {})
    exp_name = args.exp_name or exp_cfg.get("exp_name", "late_fusion")

    if not args.resume:
        date_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        exp_name = f"{exp_name}_{date_str}"
    base_output = resolve_path(args.output_dir or exp_cfg.get("output_dir", "logs/late_fusion"))
    if base_output is None:
        raise ValueError("Unable to resolve output directory.")
    exp_dir = prepare_output_directory(base_output, exp_name)

    seed = exp_cfg.get("seed", 42)
    seed_everything(seed)
    print(f"✓ Seeded everything with {seed}")

    training_cfg = config.get("training", {})
    distributed = bool(training_cfg.get("distributed", False) and torch.cuda.is_available())
    if distributed:
        local_rank, world_size, _ = setup()
        device = torch.device(f"cuda:{local_rank}")
    else:
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(device.index or 0)

    print(f"Using device {device}, world_size={world_size}")

    data_cfg = config.get("data", {})
    llm_cfg = config.get("llm", {})
    print(data_cfg.keys())
    train_loader, val_loader, _ = build_dataloaders(data_cfg, llm_cfg, world_size)

    # Print sample follow-up questions before training
    tokenizer = None
    if (not distributed) or local_rank == 0:
        print("\n" + "="*80)
        print("SAMPLE FOLLOW-UP QUESTIONS AND ANSWERS")
        print("="*80)

        # Load tokenizer for decoding - use the one from the dataset adapter
        tokenizer = train_loader.dataset.tokenizer
        if tokenizer:
             print("✓ Tokenizer available for decoding debug")

        # Sample a few batches
        sample_count = 0
        max_samples = 3

        for batch_idx, batch in enumerate(train_loader):
            print(f"Debug Batch {batch_idx}: max_samples={max_samples}, current_count={sample_count}")
            
            # Get batch size from input_ids (main turn)
            if "input_ids" not in batch:
                continue
            curr_batch_size = batch["input_ids"].size(0)

            for i in range(curr_batch_size):
                if sample_count >= max_samples:
                    break

                # Check for followup content in this specific sample
                has_active_followup = False
                # Check text-based followups (list of lists)
                if "followup_turns" in batch:
                    # 'followup_turns' is a list of length batch_size
                    if i < len(batch["followup_turns"]) and batch["followup_turns"][i]:
                        has_active_followup = True
                
                # If no text, check if we have tensors (less reliable for individual sample check if sparse, 
                # but valid for hybrid which is dense).
                # Only trust followup_turns for printing text.
                
                if not has_active_followup:
                    continue

                sample_count += 1
                print(f"\n--- Sample {sample_count} ---")

                # Print Main Turn
                if "input_texts" in batch:
                    print(f"Main Question: {batch['input_texts'][i]}")
                if "target_texts" in batch:
                    print(f"Main Answer: {batch['target_texts'][i]}")

                # Print Follow-up Turns
                if "followup_turns" in batch:
                    turns = batch["followup_turns"][i]
                    for t_idx, turn in enumerate(turns):
                        # turn is (q, a) tuple
                        if len(turn) == 2:
                            q, a = turn
                            print(f"Follow-up {t_idx+1} Question: {q}")
                            print(f"Follow-up {t_idx+1} Answer: {a}")
                        else:
                            print(f"Follow-up {t_idx+1}: {turn}")

                # Print stellar data if available
                if "stellar_data" in batch:
                    stellar_info = batch["stellar_data"][i]
                    print(f"Stellar Data: {stellar_info}")

            if sample_count >= max_samples:
                break

        if sample_count == 0:
            print("⚠ No follow-up questions found in training data (checked first few batches)!")

        print("="*80 + "\n")
    
    # Removed exit() to allow training to proceed

    llm_args = build_llm_args(config.get("llm", {}), data_cfg)
    llm_model = _load_llm_model(llm_args)
    apply_precision(llm_model, config.get("llm", {}).get("llm_precision", "fp16"))
    llm_model = llm_model.to(device)

    spectral_model = None
    # If using pre-computed features (feature mode), we don't load the spectral model
    # The dataset loader handles feature loading, and LateFusionModel will use input as features
    features_file = resolve_path(data_cfg.get("features_file"))
    if features_file and features_file.exists():
        print(f"Using pre-computed features from {features_file}; skipping spectral model load.")
        spectral_model = None
    else:
        spectral_model = maybe_load_spectral_model(config.get("spectral_model", {}))
        if spectral_model is not None:
            spectral_model = spectral_model.to(device)

    perceiver_config = load_model_config(config.get("model", {}))
    model = LateFusionModel(llm_model, spectral_model, perceiver_config).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    # DDP Wrapping moved to after Trainer initialization (Layer Freezing)

    # Load LoRA parameters from config file
    lora_params = None
    lora_config_path = training_cfg.get("lora_config_path", "/home/ilay.kamai/work/TalkingLatents/src/llm_config_tuned.json")
    if lora_config_path:
        lora_config_path_resolved = resolve_path(lora_config_path)
        if lora_config_path_resolved and lora_config_path_resolved.exists():
            try:
                with open(lora_config_path_resolved, "r") as f:
                    lora_config = json.load(f)
                    lora_params = lora_config.get("lora_params")
                    if lora_params:
                        print(f"✓ Loaded LoRA config from {lora_config_path_resolved}")
                        print(f"  LoRA rank: {lora_params.get('lora_rank')}")
                        print(f"  LoRA alpha: {lora_params.get('lora_alpha')}")
                        print(f"  Freeze strategy: {lora_params.get('freeze_strategy')}")
                        print(f"  LoRA start epoch: {lora_params.get('lora_start_epoch')}")
                    else:
                        print(f"⚠ No lora_params found in {lora_config_path_resolved}")
            except Exception as e:
                print(f"⚠ Error loading LoRA config from {lora_config_path_resolved}: {e}")
        else:
            print(f"⚠ LoRA config file not found: {lora_config_path}, continuing without LoRA")

    # Define scaler before trainer
    use_amp = bool(training_cfg.get("use_amp", True) and torch.cuda.is_available())
    scaler = GradScaler(enabled=use_amp) if use_amp else None

    # Create trainer first (this applies LoRA and freeze strategy!)
    trainer = LateFusionTrainer(
        lora_params=lora_params,
        model=model,
        optimizer=None, # Will set later
        criterion=None,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        world_size=world_size,
        output_dim=1,
        scheduler=None, # Will set later
        max_iter=training_cfg.get("max_iter", -1),
        scaler=scaler,
        use_amp=use_amp,
        grad_clip=training_cfg.get("grad_clip", True),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        log_path=str(exp_dir),
        exp_name=exp_name,
        accumulation_step=training_cfg.get("accumulation_steps", 1),
        save_full_every_epoch=training_cfg.get("save_full_every_epoch", True),
        log_loss_every=training_cfg.get("log_losses_every"),
    )
    
    
    # Wrap in DDP AFTER applying freeze/LoRA strategies
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        trainer.model = model

    # Re-verify trainable parameters after trainer applied strategies
    real_model = model.module if hasattr(model, "module") else model
    llm_trainable = sum(p.numel() for p in real_model.llm_model.parameters() if p.requires_grad)
    print(f"DEBUG: Post-Trainer LLM trainable params: {llm_trainable}")
    
    # NOW build optimizer with strictly trainable parameters
    optimizer = build_optimizer(model, training_cfg.get("optimizer", {}))
    scheduler = build_scheduler(optimizer, training_cfg.get("scheduler"))
    
    # Attach to trainer
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.tokenizer = tokenizer

    # Optional CE warmup schedule for cycle-fusion
    ce_cfg = training_cfg.get("ce", {}) or {}
    trainer.ce_schedule = {
        "enable": bool(ce_cfg.get("enable", perceiver_config.get("enable_cycle_ce", False))),
        "target_weight": float(ce_cfg.get("target_weight", 1.0)),
        "warmup_epochs": int(ce_cfg.get("warmup_epochs", 0)),
        "start_epoch": int(ce_cfg.get("start_epoch", 0)),
    }

    resume_info = None
    if args.resume:
        resume_path = resolve_path(args.resume)
        if resume_path is None or not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint '{args.resume}' not found.")
        resume_info = resume_from_checkpoint(trainer, str(resume_path), device)

    save_runtime_config(exp_dir, config, args)

    if args.dry_run:
        print("Dry run flag set; exiting before training.")
        return

    # Run initial evaluation before training starts
    # if (not distributed) or local_rank == 0:
    #     print("Running initial evaluation before training...")
    #     try:
    #         trainer.evaluate_validation_samples(device, epoch=-1)
    #     except Exception as e:
    #         print(f"Warning: Initial evaluation failed: {e}")

    results = trainer.fit(
        num_epochs=training_cfg.get("num_epochs", 1),
        device=device,
        early_stopping=training_cfg.get("early_stopping"),
        best=training_cfg.get("best_metric", "loss"),
        start_epoch=resume_info.get("epoch", 0) + 1 if resume_info else 0,
        initial_min_loss=resume_info.get("min_loss") if resume_info else None,
        initial_best_acc=resume_info.get("best_acc") if resume_info else None,
    )

    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        results_path = exp_dir / f"{exp_name}_final_results.json"
        with open(results_path, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"✓ Training finished. Metrics saved to {results_path}")


if __name__ == "__main__":
    main()
