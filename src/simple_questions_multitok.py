import os
import sys
import json
import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.system('pip install tiktoken fairscale fire blobfile torchdiffeq torchcfm')

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
from data.dataset_mixed import create_mixed_dataloaders
from data.transforms import GeneralSpectrumPreprocessor, ToTensor, Compose
import numpy as np
import torch.distributed as dist
import gc


def parse_args():
    """Parse command line arguments with mode support"""
    # Base parser now includes all needed arguments
    args = base_parse_args()
    
    # Override default mode for this script to 'combined'
    if args.mode == 'single_star':  # Only change if it's the default
        import sys
        if '--mode' not in sys.argv:  # User didn't specify mode explicitly
            args.mode = 'combined'
    
    return args


def create_datasets_and_loaders(args, device):
    """Create datasets and dataloaders with mode support - only on rank 0 for memory efficiency"""
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    # Only rank 0 loads spectral features to avoid OOM
    spectral_features = None
    if rank == 0:
        if args.features_file and os.path.exists(args.features_file):
            print(f"Loading spectral features from {args.features_file}")
            spectral_features = np.load(args.features_file)
            print(f"Spectral features shape: {spectral_features.shape}")
        else:
            print("No spectral features file provided or file not found. Will use raw spectra on-the-fly.")
    
    # Synchronize before proceeding
    if dist.is_initialized():
        dist.barrier()
    
    model_path, tokenizer_path = get_model_path(args)
    transf = Compose([GeneralSpectrumPreprocessor(rv_norm=True), ToTensor()])
    
    # Create cache directory for split consistency
    cache_dir_base = os.path.join(args.output_dir, 'cache')
    cache_dir = cache_dir_base if rank == 0 else f"{cache_dir_base}_r{rank}"
    os.makedirs(cache_dir, exist_ok=True)
    
    if args.mode == "two_star":
        if rank == 0:
            print(f"Creating two-star comparative datasets from {args.comparative_json_file}...")
        
        train_loader, val_loader, test_loader = create_comparative_dataloaders(
            json_file=args.comparative_json_file,
            features_array=spectral_features,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_seed,
            num_workers=args.num_workers // world_size if world_size > 1 else args.num_workers,
            cache_dir=cache_dir,
            tokenizer_path=tokenizer_path,
            max_length=args.max_seq_length,
            num_spectral_features=args.num_spectral_features,
        )
        
    elif args.mode == "combined":
        if rank == 0:
            print(f"Creating mixed dataset from {args.json_file} and {args.comparative_json_file}...")
            print(f"Single-sample probability: {args.single_sample_prob}")

        single_kwargs = dict(
            json_file=args.json_file,
            features_array=spectral_features,
            spectral_transforms=transf,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_seed,
            num_spectral_features=args.num_spectral_features,
            cache_dir=cache_dir + "_single",
            tokenizer_path=tokenizer_path,
            max_length=args.max_seq_length,
        )

        comparative_kwargs = None
        if args.comparative_json_file:
            comparative_kwargs = dict(
                json_file=args.comparative_json_file,
                features_array=spectral_features,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                random_state=args.random_seed,
                cache_dir=cache_dir + "_two",
                tokenizer_path=tokenizer_path,
                max_length=args.max_seq_length,
                num_spectral_features=args.num_spectral_features,
                spectral_transforms=transf,
            )

        num_workers_per_rank = args.num_workers // world_size if world_size > 1 else args.num_workers
        train_loader, val_loader, test_loader = create_mixed_dataloaders(
            single_kwargs=single_kwargs,
            comparative_kwargs=comparative_kwargs,
            batch_size=args.batch_size,
            single_sample_prob=args.single_sample_prob,
            seed=args.random_seed,
            length_strategy='max',
            num_workers=num_workers_per_rank,
            persistent_workers=num_workers_per_rank > 0,
            pin_memory=torch.cuda.is_available(),
            numeric_keys=('Teff', 'logg', 'FeH'),
        )
        
    else:
        if rank == 0:
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
            cache_dir=cache_dir,
            tokenizer_path=tokenizer_path,  
            max_length=args.max_seq_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers // world_size if world_size > 1 else args.num_workers,
        )

    # Synchronize all processes after dataset creation
    if dist.is_initialized():
        dist.barrier()

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
        use_cfm=args.use_cfm,  # Add CFM flag
        cfm_weight=args.cfm_weight,  # CFM loss weight
        predict_stellar_params=True,  # Enable stellar parameter prediction
        stellar_params=['Teff', 'logg', 'FeH'],  # Predict these parameters
        quantiles=args.quantiles  # CQR quantiles
    ).to(device)
    
    # Keep stellar predictor and transformer in FP32 for numerical stability
    if hasattr(model, 'stellar_predictor') and model.stellar_predictor is not None:
        model.stellar_predictor.float()
    if hasattr(model, 'stellar_transformer') and model.stellar_transformer is not None:
        for layer in model.stellar_transformer:
            layer.float()
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

    # Clear GPU memory and force garbage collection before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("Creating datasets and dataloaders...")
    train_loader, val_loader, test_loader = create_datasets_and_loaders(args, local_rank)
    
    # Clear memory after dataset creation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Handle tokenizer extraction for combined mode
    if args.mode == "combined":
        # For mixed datasets, get tokenizer from the single dataset inside the mixed dataset
        tokenizer = train_loader.dataset.single_dataset.tokenizer
    else:
        tokenizer = train_loader.dataset.tokenizer

    print("Creating multitoken multimodal model...")
    model = build_model_multitok(args, local_rank)

    # Clear memory after model creation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    
    # Clear memory after freezing parameters
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
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

    print("Creating optimizer and scheduler...")
    optimizer, scheduler, scaler = create_optimizer_and_scheduler(model, args, train_loader)

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
    initial_train_loader = train_loader
    initial_val_loader = val_loader
    initial_mode = args.mode

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
        curriculum_decay_steps=args.curriculum_decay_steps,
        quantiles=args.quantiles,
    )
    
    trainer.combined_mode = (args.mode == "combined")
    trainer.scheduler = scheduler
    trainer.tokenizer = tokenizer

    if local_rank == 0:
        save_config(args, args.output_dir)

    if args.train:
        trainer.evaluate_validation_samples(local_rank, 0)
        _ = trainer.fit(
            num_epochs=args.num_epochs,
            device=local_rank,
            early_stopping=args.early_stopping,
            best='loss'
        )


if __name__ == '__main__':
    print("="*80)
    print("MULTIMODAL STELLAR MODEL TRAINING (Multi spectral tokens)")
    print("Supports both single-star and two-star comparative modes")
    print("="*80)
    main()
