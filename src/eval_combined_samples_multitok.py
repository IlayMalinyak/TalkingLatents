import os
import sys
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from data.dataset_interpert import create_stellar_dataloaders, collate_fn as single_collate_fn
from data.dataset_comparative import create_comparative_dataloaders, collate_comparative_fn
from data.transforms import Compose, GeneralSpectrumPreprocessor, ToTensor
from nn.llm_multi import MultimodalLlamaModelMultiTokens
from src.simple_questions import _load_llm_model, _load_spectra_model, get_model_path


def setup_single_gpu():
    """Initialize single-process distributed + fairscale MP environment."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for evaluation")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        import fairscale.nn.model_parallel.initialize as fs_init
        if not fs_init.model_parallel_is_initialized():
            fs_init.initialize_model_parallel(1)
    except Exception:
        pass
    torch.cuda.set_device(0)
    return torch.device('cuda', 0)


def parse_args():
    parser = argparse.ArgumentParser("Evaluate multi-token model on combined single/comparative samples")
    parser.add_argument('--json_file', type=str, required=True,
                        help='Path to simple (single-star) questions JSON')
    parser.add_argument('--comparative_json_file', type=str, required=True,
                        help='Path to comparative questions JSON')
    parser.add_argument('--features_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='logs/eval_multitok_combined')
    parser.add_argument('--exp_name', type=str, default='eval_multitok_combined')
    parser.add_argument('--llm_root', type=str, default=os.environ.get('LLM_ROOT', '/data/.llama'))
    parser.add_argument('--llm_model', type=str, default='Llama3.1-8B')
    parser.add_argument('--llm_path', type=str, default=None)
    parser.add_argument('--resume_path', type=str, default=None,
                        help='Composite checkpoint from training run (used to locate cached splits)')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--llm_precision', type=str, default='fp16', choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('--spectral_embedding_dim', type=int, default=2048)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_spectral_features', type=int, default=8,
                        help='Number of spectral tokens per star')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Dummy batch size used for LLM loading helpers')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Total samples to evaluate (will be split across datasets)')
    parser.add_argument('--num_simple_samples', type=int, default=None,
                        help='Override number of single-star samples (defaults to half)')
    parser.add_argument('--num_comparative_samples', type=int, default=None,
                        help='Override number of comparative samples (defaults to remainder)')
    parser.add_argument('--max_new_tokens', type=int, default=150)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--split_cache_root', type=str, default=None,
                        help='Root directory containing cached dataset splits')
    parser.add_argument('--allow_new_splits', action='store_true',
                        help='Regenerate splits if cached indices are missing')
    return parser.parse_args()


def load_model(args, device):
    llm = _load_llm_model(args)
    if args.llm_precision == 'fp16':
        llm.half()
    elif args.llm_precision == 'bf16' and torch.cuda.is_bf16_supported():
        llm.to(dtype=torch.bfloat16)
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
        use_checkpoint=False,
    ).to(device)

    if args.resume_path and os.path.isfile(args.resume_path):
        ckpt = torch.load(args.resume_path, map_location='cpu')
        state = ckpt.get('model', ckpt)
        if 'state_dict' in state:
            state = state['state_dict']
        cleaned = { (k[7:] if k.startswith('module.') else k): v for k, v in state.items() }
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[WARN] Missing keys during resume load: {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys during resume load: {unexpected}")
    elif args.weights and os.path.isfile(args.weights):
        state = torch.load(args.weights, map_location='cpu')
        if isinstance(state, dict) and ('model' in state or 'state_dict' in state):
            state = state.get('model', state.get('state_dict'))
        model.load_state_dict(state, strict=False)

    model.eval()
    return model


def compute_tf_perplexity_single(model, batch, device):
    prev_mode = model.mode
    model.mode = "single_star"

    input_ids = batch['input_ids'].to(device)
    target_ids = batch['target_ids'].to(device)
    spectra = batch['masked_spectra'].to(device)
    feature_indices = batch['feature_start_indices'].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids,
                    input_spectra=spectra,
                    special_token_positions=feature_indices)
        logits = out['logits']

    model.mode = prev_mode

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target_ids[..., 1:].contiguous()
    mask = (shift_labels != -100)
    if not mask.any():
        return float('inf')
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    flat_mask = mask.view(-1)
    active_logits = flat_logits[flat_mask]
    active_labels = flat_labels[flat_mask]
    if active_logits.numel() == 0:
        return float('inf')
    loss = torch.nn.functional.cross_entropy(active_logits, active_labels, reduction='mean')
    return torch.exp(loss).item()


def compute_tf_perplexity_two_star(model, batch, device):
    prev_mode = model.mode
    model.mode = "two_star"

    input_ids = batch['input_ids'].to(device)
    target_ids = batch['target_ids'].to(device)
    star_a_spectra = batch['masked_spectra_a'].to(device)
    star_b_spectra = batch['masked_spectra_b'].to(device)
    star_a_indices = batch['star_a_feature_indices'].to(device)
    star_b_indices = batch['star_b_feature_indices'].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids,
                    input_spectra=None,
                    star_a_spectra=star_a_spectra,
                    star_b_spectra=star_b_spectra,
                    star_a_indices=star_a_indices,
                    star_b_indices=star_b_indices)
        logits = out['logits']

    model.mode = prev_mode

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target_ids[..., 1:].contiguous()
    mask = (shift_labels != -100)
    if not mask.any():
        return float('inf')
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    flat_mask = mask.view(-1)
    active_logits = flat_logits[flat_mask]
    active_labels = flat_labels[flat_mask]
    if active_logits.numel() == 0:
        return float('inf')
    loss = torch.nn.functional.cross_entropy(active_logits, active_labels, reduction='mean')
    return torch.exp(loss).item()


def sample_dataset_indices(total, count):
    if total <= count:
        return list(range(total))
    return random.sample(range(total), count)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    device = setup_single_gpu()

    spectral_features = None
    if args.features_file and os.path.exists(args.features_file):
        spectral_features = np.load(args.features_file)

    _, tokenizer_path = get_model_path(args)
    transforms = Compose([GeneralSpectrumPreprocessor(rv_norm=True), ToTensor()])

    split_root = args.split_cache_root or os.path.join(ROOT_DIR, 'split_cache')
    allow_new_splits = bool(args.allow_new_splits)

    def resolve_split_dir(subfolder: str, json_path: str) -> str:
        dataset_key = Path(json_path).stem
        target = os.path.join(split_root, subfolder, dataset_key)
        if not os.path.isdir(target):
            if allow_new_splits:
                os.makedirs(target, exist_ok=True)
            else:
                raise FileNotFoundError(
                    f"Cached splits not found at {target}. Provide valid --split_cache_root "
                    "or rerun with --allow_new_splits to generate them.")
        return target

    single_cache_dir = resolve_split_dir('single', args.json_file)
    comp_cache_dir = resolve_split_dir('comparative', args.comparative_json_file)

    single_train, single_val, single_test = create_stellar_dataloaders(
        json_file=args.json_file,
        features_array=spectral_features,
        spectral_transforms=transforms,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed,
        num_spectral_features=args.num_spectral_features,
        cache_dir=single_cache_dir,
        allow_new_splits=allow_new_splits,
        tokenizer_path=tokenizer_path,
        max_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    comp_train, comp_val, comp_test = create_comparative_dataloaders(
        json_file=args.comparative_json_file,
        features_array=spectral_features,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed,
        num_workers=args.num_workers,
        cache_dir=comp_cache_dir,
        allow_new_splits=allow_new_splits,
        tokenizer_path=tokenizer_path,
        max_length=args.max_seq_length,
        num_stellar_features=args.num_spectral_features,
    )

    single_dataset = single_test.dataset
    comp_dataset = comp_test.dataset
    tokenizer = getattr(single_dataset, 'tokenizer', None)

    total_samples = max(args.num_samples, 0)
    if args.num_simple_samples is not None:
        num_single = args.num_simple_samples
    else:
        num_single = total_samples // 2
    if args.num_comparative_samples is not None:
        num_comp = args.num_comparative_samples
    else:
        num_comp = total_samples - num_single

    num_single = min(num_single, len(single_dataset))
    num_comp = min(num_comp, len(comp_dataset))

    if num_single + num_comp == 0:
        print("No samples selected; nothing to evaluate.")
        return

    model = load_model(args, device)

    results = {
        'epoch': None,
        'samples': [],
        'avg_teacher_forcing_perplexity': None,
        'avg_generation_perplexity': None,
    }

    tf_values = []
    gen_values = []

    single_indices = sample_dataset_indices(len(single_dataset), num_single)
    comp_indices = sample_dataset_indices(len(comp_dataset), num_comp)

    for idx in single_indices:
        sample = single_dataset[idx]
        batch = single_collate_fn([sample])

        tf_perp = compute_tf_perplexity_single(model, batch, device)

        prev_mode = model.mode
        model.mode = "single_star"
        gen_text, input_text, target_text, logps = model.generate_response_from_batch(
            batch_data=batch,
            batch_idx=0,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        model.mode = prev_mode

        if logps:
            avg_logp = float(np.mean(logps))
            gen_perp = float(np.exp(-avg_logp))
        else:
            gen_perp = float('inf')

        results['samples'].append({
            'mode': 'single_star',
            'obsid': sample.get('obsid', 'Unknown') if isinstance(sample, dict) else 'Unknown',
            'question': input_text,
            'true_answer': target_text,
            'generated_answer': gen_text,
            'teacher_forcing_perplexity': None if np.isinf(tf_perp) else tf_perp,
            'generation_perplexity': None if np.isinf(gen_perp) else gen_perp,
            'num_generated_tokens': len(logps),
        })

        if not np.isinf(tf_perp):
            tf_values.append(tf_perp)
        if not np.isinf(gen_perp):
            gen_values.append(gen_perp)

    for idx in comp_indices:
        sample = comp_dataset[idx]
        batch = collate_comparative_fn([sample])

        tf_perp = compute_tf_perplexity_two_star(model, batch, device)

        prev_mode = model.mode
        model.mode = "two_star"
        gen_text, input_text, target_text, logps = model.generate_response_from_batch(
            batch_data=batch,
            batch_idx=0,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        model.mode = prev_mode

        if logps:
            avg_logp = float(np.mean(logps))
            gen_perp = float(np.exp(-avg_logp))
        else:
            gen_perp = float('inf')

        results['samples'].append({
            'mode': 'two_star',
            'pair_id': sample.get('pair_id', 'Unknown') if isinstance(sample, dict) else 'Unknown',
            'question': sample.get('full_question_text', input_text),
            'true_answer': target_text,
            'generated_answer': gen_text,
            'teacher_forcing_perplexity': None if np.isinf(tf_perp) else tf_perp,
            'generation_perplexity': None if np.isinf(gen_perp) else gen_perp,
            'num_generated_tokens': len(logps),
        })

        if not np.isinf(tf_perp):
            tf_values.append(tf_perp)
        if not np.isinf(gen_perp):
            gen_values.append(gen_perp)

    if tf_values:
        results['avg_teacher_forcing_perplexity'] = float(np.mean(tf_values))
    if gen_values:
        results['avg_generation_perplexity'] = float(np.mean(gen_values))

    out_file = os.path.join(args.output_dir, f"{args.exp_name}_combined_samples.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved combined samples to: {out_file}")


if __name__ == '__main__':
    main()
