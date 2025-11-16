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
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

# Ensure project root on path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from data.dataset_late_fusion import create_late_fusion_dataloaders
from nn.late_fusion import LateFusionModel
from nn.train import LateFusionTrainer
from src.simple_questions import setup, _load_llm_model, _load_spectra_model


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


def build_dataloaders(data_cfg: Dict[str, Any], world_size: int):
    dataset_kwargs = dict(data_cfg.get("dataset_kwargs", {}))
    cache_dir = resolve_path(data_cfg.get("cache_dir"))
    tokenizer_path = resolve_path(data_cfg.get("tokenizer_path"))
    if tokenizer_path:
        dataset_kwargs["tokenizer_path"] = str(tokenizer_path)
    if cache_dir:
        dataset_kwargs["cache_dir"] = str(cache_dir)

    train_loader, val_loader, test_loader = create_late_fusion_dataloaders(
        json_file=str(resolve_path(data_cfg["json_file"])),
        batch_size=data_cfg.get("batch_size", 4),
        train_ratio=data_cfg.get("train_ratio", 0.7),
        val_ratio=data_cfg.get("val_ratio", 0.15),
        test_ratio=data_cfg.get("test_ratio", 0.15),
        random_state=data_cfg.get("random_state", 42),
        num_workers=data_cfg.get("num_workers", 0),
        world_size=world_size,
        max_length=data_cfg.get("max_length", 512),
        **dataset_kwargs,
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
    ckpt = torch.load(resume_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    trainer._unwrap_model().load_state_dict(state, strict=False)
    if trainer.optimizer is not None and ckpt.get("optimizer"):
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
    if trainer.scheduler is not None and ckpt.get("scheduler"):
        trainer.scheduler.load_state_dict(ckpt["scheduler"])
    if trainer.scaler is not None and ckpt.get("scaler"):
        trainer.scaler.load_state_dict(ckpt["scaler"])
    print(f"✓ Resumed weights from {resume_path}")
    return ckpt


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config file '{args.config}' not found.")
    config = load_yaml_config(config_path)

    exp_cfg = config.get("experiment", {})
    exp_name = args.exp_name or exp_cfg.get("exp_name", "late_fusion")
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
    train_loader, val_loader, _ = build_dataloaders(data_cfg, world_size)

    llm_args = build_llm_args(config.get("llm", {}), data_cfg)
    llm_model = _load_llm_model(llm_args)
    apply_precision(llm_model, config.get("llm", {}).get("llm_precision", "fp16"))
    llm_model = llm_model.to(device)

    spectral_model = maybe_load_spectral_model(config.get("spectral_model", {}))
    if spectral_model is not None:
        spectral_model = spectral_model.to(device)

    perceiver_config = load_model_config(config.get("model", {}))
    model = LateFusionModel(llm_model, spectral_model, perceiver_config).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    optimizer = build_optimizer(model, training_cfg.get("optimizer", {}))
    scheduler = build_scheduler(optimizer, training_cfg.get("scheduler"))
    use_amp = bool(training_cfg.get("use_amp", True) and torch.cuda.is_available())
    scaler = GradScaler(enabled=use_amp) if use_amp else None

    trainer = LateFusionTrainer(
        model=model,
        optimizer=optimizer,
        criterion=None,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        world_size=world_size,
        output_dim=1,
        scheduler=scheduler,
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
