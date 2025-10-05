import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.system('pip install tiktoken fairscale fire blobfile')

from src.simple_questions import (
    setup,
    parse_args as base_parse_args,
    create_optimizer_and_scheduler,
    save_config,
    _load_llm_model,
    _load_spectra_model,
    print_detailed_memory,
    get_model_path
)
from nn.llm_multi import MultimodalLlamaModelMultiTokens
from nn.train import LLMTrainer
from data.dataset_interpert import create_stellar_dataloaders
from data.dataset_comparative import create_comparative_dataloaders
from data.transforms import GeneralSpectrumPreprocessor, ToTensor, Compose
import numpy as np
import torch.distributed as dist


class InterleavedDataLoader:
    """Alternate batches from multiple dataloaders while tagging their mode."""

    def __init__(self, loaders: Dict[str, DataLoader], mode_order=None):
        self.loaders = loaders
        self.mode_order = mode_order or list(loaders.keys())

    def __len__(self):
        return sum(len(loader) for loader in self.loaders.values())

    def set_epoch(self, epoch: int):
        for loader in self.loaders.values():
            sampler = getattr(loader, 'sampler', None)
            if sampler is not None and hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)

    def __iter__(self):
        iterators = {mode: iter(loader) for mode, loader in self.loaders.items()}
        finished = {mode: False for mode in self.mode_order}

        while not all(finished.values()):
            for mode in self.mode_order:
                if finished[mode]:
                    continue
                try:
                    batch = next(iterators[mode])
                except StopIteration:
                    finished[mode] = True
                    continue

                if isinstance(batch, dict):
                    tagged = dict(batch)
                    tagged['mode'] = mode
                    yield tagged
                else:
                    yield {'mode': mode, 'batch': batch}

def parse_args():
    """Parse command line arguments with multitoken defaults."""
    args = base_parse_args()
    return args


def create_datasets_and_loaders(args, device):
    """Create datasets and dataloaders with mode support"""
    
    spectral_features = None
    if args.features_file and os.path.exists(args.features_file):
        print(f"Loading spectral features from {args.features_file}")
        spectral_features = np.load(args.features_file)
        print(f"Spectral features shape: {spectral_features.shape}")
    else:
        print("No spectral features file provided or file not found. Will use raw spectra on-the-fly.")
    
    model_path, tokenizer_path = get_model_path(args)
    transf = Compose([GeneralSpectrumPreprocessor(rv_norm=True), ToTensor()])
    
    split_root = args.split_cache_root or os.path.join(ROOT_DIR, 'split_cache')
    os.makedirs(split_root, exist_ok=True)

    def build_split_dir(subfolder: str, json_path: str) -> str:
        dataset_key = Path(json_path).stem
        target_dir = os.path.join(split_root, subfolder, dataset_key)
        if args.allow_new_splits and not os.path.isdir(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        return target_dir

    single_split_dir = build_split_dir('single', args.json_file)
    comparative_split_dir = build_split_dir('comparative', args.comparative_json_file)

    allow_new_splits = bool(args.allow_new_splits)
    
    if args.mode == "two_star":
        print(f"Creating two-star comparative datasets from {args.comparative_json_file}...")
        
        train_loader, val_loader, test_loader = create_comparative_dataloaders(
            json_file=args.comparative_json_file,
            features_array=spectral_features,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_seed,
            num_workers=args.num_workers,
            cache_dir=comparative_split_dir,
            allow_new_splits=allow_new_splits,
            tokenizer_path=tokenizer_path,
            max_length=args.max_seq_length,
            num_stellar_features=args.num_spectral_features,
        )
        
    elif args.mode == "combined":
        print(f"Creating combined datasets - single_star from {args.json_file} and two_star from {args.comparative_json_file}...")
        print("Will interleave single_star and two_star batches within each epoch")
        
        # Create both single-star and two-star dataloaders
        print("Creating single-star dataloaders...")
        single_train_loader, single_val_loader, single_test_loader = create_stellar_dataloaders(
            json_file=args.json_file,
            features_array=spectral_features,
            spectral_transforms=transf,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_seed,
            num_spectral_features=args.num_spectral_features,
            cache_dir=single_split_dir,
            allow_new_splits=allow_new_splits,
            tokenizer_path=tokenizer_path,  
            max_length=args.max_seq_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        
        print("Creating two-star dataloaders...")
        two_train_loader, two_val_loader, two_test_loader = create_comparative_dataloaders(
            json_file=args.comparative_json_file,
            features_array=spectral_features,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_seed,
            num_workers=args.num_workers,
            cache_dir=comparative_split_dir,
            allow_new_splits=allow_new_splits,
            tokenizer_path=tokenizer_path,
            max_length=args.max_seq_length,
            num_stellar_features=args.num_spectral_features,
        )
        
        interleaved_train = InterleavedDataLoader(
            {'single_star': single_train_loader, 'two_star': two_train_loader},
            mode_order=['single_star', 'two_star']
        )
        interleaved_val = InterleavedDataLoader(
            {'single_star': single_val_loader, 'two_star': two_val_loader},
            mode_order=['single_star', 'two_star']
        )
        interleaved_test = InterleavedDataLoader(
            {'single_star': single_test_loader, 'two_star': two_test_loader},
            mode_order=['single_star', 'two_star']
        )

        train_loader = {
            'mixed': interleaved_train,
            'single_star': single_train_loader,
            'two_star': two_train_loader,
        }
        val_loader = {
            'mixed': interleaved_val,
            'single_star': single_val_loader,
            'two_star': two_val_loader,
        }
        test_loader = {
            'mixed': interleaved_test,
            'single_star': single_test_loader,
            'two_star': two_test_loader,
        }
        
    else:
        print(f"Creating single-star datasets from {args.json_file}...")

        train_loader, val_loader, test_loader = create_stellar_dataloaders(
            json_file=args.json_file,
            features_array=spectral_features,
            spectral_transforms=transf,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_seed,
            num_spectral_features=args.num_spectral_features,
            cache_dir=single_split_dir,
            allow_new_splits=allow_new_splits,
            tokenizer_path=tokenizer_path,  
            max_length=args.max_seq_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    return train_loader, val_loader, test_loader


def build_model_multitok(args, device):
    print("Loading LLM model...")
    llm = _load_llm_model(args)
    if args.llm_precision == 'fp16':
        llm.half(); print("✓ LLM weights cast to float16")
    elif args.llm_precision == 'bf16' and torch.cuda.is_bf16_supported():
        llm.to(dtype=torch.bfloat16); print("✓ LLM weights cast to bfloat16")
    llm = llm.to(device)

    fm = None if args.features_file else _load_spectra_model()
    if fm is not None:
        fm = fm.to(device)

    model = MultimodalLlamaModelMultiTokens(
        base_model=llm,
        fm_model=fm,
        latent_dim=args.spectral_embedding_dim,
        hidden_dim=args.hidden_dim,
        num_spectral_features=args.num_spectral_features,
        mode=args.mode,
    ).to(device)
    # Keep projector in float32 for numeric stability with GradScaler
    # (base model runs in fp16/bf16; features are cast to projector dtype inside model)
    return model


def main():
    args = parse_args()
    # allow override from env for quick tests
    date = __import__('datetime').datetime.now().strftime('%Y-%m-%d-%H-%M')
    args.output_dir = os.path.join(args.output_dir, date)
    os.makedirs(args.output_dir, exist_ok=True)

    local_rank, world_size, _ = setup()
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    print("Creating datasets and dataloaders...")
    train_loader, val_loader, test_loader = create_datasets_and_loaders(args, local_rank)
    
    # Handle tokenizer extraction for combined mode
    if args.mode == "combined":
        tokenizer = train_loader['single_star'].dataset.tokenizer
    else:
        tokenizer = train_loader.dataset.tokenizer

    print("Creating multitoken multimodal model...")
    model = build_model_multitok(args, local_rank)

    print(f"Model mode after creation: {model.mode}")
    if hasattr(model, 'module'):
        print(f"Model.module mode: {model.module.mode}")

    # Freeze large submodules BEFORE wrapping with DDP so the reducer
    # only tracks truly trainable parameters (avoids unused-grad errors).
    base = model
    if isinstance(base, DDP):
        base = base.module
    if hasattr(base, 'base_model') and base.base_model is not None:
        for p in base.base_model.parameters():
            p.requires_grad = False
        print("✓ Frozen base LLaMA parameters")
    if hasattr(base, 'fm_model') and base.fm_model is not None:
        for p in base.fm_model.parameters():
            p.requires_grad = False
        print("✓ Frozen spectral FM parameters")
    print_detailed_memory()

    if world_size > 1:
        print(f"Wrapping with DDP (world_size={world_size})")
        # All large modules are frozen; but LoRA may be applied later and
        # not participate immediately. Use find_unused_parameters=True to
        # avoid reducer errors in early epochs.
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
            broadcast_buffers=False,
            bucket_cap_mb=25,
            gradient_as_bucket_view=True,
        )
    else:
        print("Single GPU - no DDP")

    print("Creating trainer (tuned LoRA config)...")
    # Load tuned LoRA config (attention-only by default)
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

    # Handle dataloader for trainer initialization
    if args.mode == "combined":
        initial_train_loader = train_loader['mixed']
        initial_val_loader = val_loader['mixed']
        initial_mode = "single_star"
    else:
        initial_train_loader = train_loader
        initial_val_loader = val_loader
        initial_mode = args.mode

    print("Creating optimizer and scheduler...")
    optimizer, scheduler, scaler = create_optimizer_and_scheduler(model, args, initial_train_loader)

    trainer = LLMTrainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        train_dataloader=initial_train_loader,
        val_dataloader=initial_val_loader,
        device=local_rank,
        world_size=world_size,
        output_dim=1,
        scheduler=None,
        max_iter=args.max_iter,
        log_path=args.output_dir,
        exp_name=args.exp_name,
        lora_params=lora_params,
        scaler=scaler,
        use_amp=args.use_amp,
        max_grad_norm=args.max_grad_norm,
        mode=initial_mode,
        validation_interval=args.validation_interval,
    )
    
    # Store combined mode information in trainer for mode-aware batching
    if args.mode == "combined":
        trainer.combined_mode = True
        trainer.single_star_loaders = {'train': train_loader['single_star'], 'val': val_loader['single_star']}
        trainer.two_star_loaders = {'train': train_loader['two_star'], 'val': val_loader['two_star']}
        trainer.mixed_train_loader = train_loader['mixed']
        trainer.mixed_val_loader = val_loader['mixed']
        trainer.original_mode = args.mode
    else:
        trainer.combined_mode = False
    trainer.scheduler = scheduler
    trainer.tokenizer = tokenizer

    # Resume handling (model + optimizer + scheduler + scaler + epoch)
    start_epoch = 0
    initial_min_loss = None
    initial_best_acc = None

    def _load_state_dict(target, state):
        if state is None:
            return
        clean_state = {}
        for k, v in state.items():
            nk = k[7:] if k.startswith('module.') else k
            clean_state[nk] = v
        target.load_state_dict(clean_state, strict=False)

    if getattr(args, 'resume_path', None) and os.path.exists(args.resume_path):
        print(f"Resuming from checkpoint: {args.resume_path}")
        try:
            ckpt = torch.load(args.resume_path, map_location='cpu')
        except Exception as err:
            print(f"Failed to load resume checkpoint ({err}); proceeding without exact resume")
            ckpt = None
        if ckpt is not None:
            model_state = ckpt.get('model', ckpt)
            if isinstance(model_state, dict) and 'state_dict' in model_state:
                model_state = model_state['state_dict']
            _load_state_dict(model.module if isinstance(model, DDP) else model, model_state)

            opt_state = ckpt.get('optimizer')
            if opt_state and optimizer is not None:
                try:
                    optimizer.load_state_dict(opt_state)
                    print("✓ Optimizer state loaded")
                except Exception as err:
                    print(f"Warning: optimizer state not loaded ({err})")

            sched_state = ckpt.get('scheduler')
            if sched_state and scheduler is not None:
                try:
                    scheduler.load_state_dict(sched_state)
                    print("✓ Scheduler state loaded")
                except Exception as err:
                    print(f"Warning: scheduler state not loaded ({err})")

            scaler_state = ckpt.get('scaler')
            if scaler_state and scaler is not None:
                try:
                    scaler.load_state_dict(scaler_state)
                    print("✓ AMP scaler state loaded")
                except Exception as err:
                    print(f"Warning: scaler state not loaded ({err})")

            start_epoch = int(ckpt.get('epoch', -1)) + 1
            initial_min_loss = ckpt.get('min_loss')
            initial_best_acc = ckpt.get('best_acc')
    elif getattr(args, 'checkpoint_dir', None):
        ckpt_path = os.path.join(args.checkpoint_dir, f"{args.exp_name}.pth")
        if os.path.isfile(ckpt_path):
            print(f"Warm-starting weights from {ckpt_path}")
            state = torch.load(ckpt_path, map_location='cpu')
            _load_state_dict(model.module if isinstance(model, DDP) else model, state)
        else:
            print(f"Checkpoint directory provided but file not found: {ckpt_path}")

    if local_rank == 0:
        save_config(args, args.output_dir)

    if args.train:
        if start_epoch == 0:
            trainer.evaluate_validation_samples(local_rank, 0)
        print(f"Starting training from epoch {start_epoch}...")
        _ = trainer.fit(
            num_epochs=args.num_epochs,
            device=local_rank,
            early_stopping=args.early_stopping,
            best='loss',
            start_epoch=start_epoch,
            initial_min_loss=initial_min_loss,
            initial_best_acc=initial_best_acc,
        )


if __name__ == '__main__':
    print("="*80)
    print("MULTIMODAL STELLAR MODEL TRAINING (Multi spectral tokens)")
    print("Supports both single-star and two-star comparative modes")
    print("="*80)
    main()
