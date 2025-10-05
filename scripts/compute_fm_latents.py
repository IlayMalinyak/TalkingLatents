import json
import os
from pathlib import Path
from typing import Optional

import torch


def load_fm_model(config_yaml: Path, weights_path: Path, device: torch.device):
    import yaml
    from nn.models import MultiTaskRegressor
    from util.utils import Container

    cfg = yaml.safe_load(open(config_yaml, 'r'))
    cfg['model_args']['avg_output'] = False
    model = MultiTaskRegressor(Container(**cfg['model_args']), Container(**cfg['conformer_args']))

    ckpt = torch.load(weights_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    # strip 'module.' if present
    state = { (k[7:] if k.startswith('module.') else k): v for k, v in state.items() }
    model.load_state_dict(state)
    model.eval().to(device)
    return model, cfg['conformer_args']['encoder_dim']


def iter_all_splits(json_file: Path,
                    batch_size: int,
                    num_workers: int,
                    random_seed: int,
                    num_spectral_features: int,
                    max_length: int):
    from data.dataset_interpert import create_stellar_dataloaders
    from data.transforms import Compose, GeneralSpectrumPreprocessor, ToTensor
    import numpy as np

    transforms = Compose([GeneralSpectrumPreprocessor(rv_norm=True), ToTensor()])
    train_loader, val_loader, test_loader = create_stellar_dataloaders(
        json_file=str(json_file),
        features_array=None,
        spectral_transforms=transforms,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=random_seed,
        num_spectral_features=num_spectral_features,
        cache_dir=None,
        tokenizer_path=None,
        max_length=max_length,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return [train_loader, val_loader, test_loader]


def main():
    import argparse
    import numpy as np

    p = argparse.ArgumentParser("Compute FM latents for all items in a JSON and save as .npy aligned by 'index'")
    p.add_argument('--json_file', type=Path, required=True)
    p.add_argument('--config_yaml', type=Path, required=True, help='FM model config YAML')
    p.add_argument('--weights', type=Path, required=True, help='FM model weights .pth')
    p.add_argument('--out', type=Path, required=True, help='Output .npy path')
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_length', type=int, default=128)
    p.add_argument('--num_spectral_features', type=int, default=4)
    args = p.parse_args()

    # Load JSON to determine N
    raw = json.load(open(args.json_file, 'r'))
    N = len(raw)

    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    fm, D = load_fm_model(args.config_yaml, args.weights, device)

    latents = np.zeros((N, D), dtype=np.float32)
    seen = np.zeros(N, dtype=bool)

    loaders = iter_all_splits(args.json_file, args.batch, args.num_workers, args.seed, args.num_spectral_features, args.max_length)

    with torch.no_grad():
        for loader in loaders:
            for batch in loader:
                x = batch['masked_spectra'].to(device)
                df_idx = batch['df_indices'] if 'df_indices' in batch else batch.get('df_index', None)
                if isinstance(df_idx, list):
                    idxs = np.array([int(i) for i in df_idx])
                else:
                    idxs = batch['df_indices'].cpu().numpy()
                # Forward FM
                _, _, x_enc = fm(x)
                # mean over stages (S)
                if x_enc.dim() == 3:
                    feat = x_enc.mean(dim=1)
                else:
                    feat = x_enc
                feat = feat.detach().cpu().numpy().astype(np.float32)
                # Assign
                latents[idxs] = feat
                seen[idxs] = True

    missing = int((~seen).sum())
    if missing > 0:
        print(f"Warning: {missing} items not covered by current splits; leaving zeros for them.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, latents)
    print(f"Saved latents: shape={latents.shape} to {args.out}")


if __name__ == '__main__':
    main()

