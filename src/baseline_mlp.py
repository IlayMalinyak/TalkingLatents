import argparse
import json
import math
import os
import sys
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Import updated data pipeline
from data.dataset_interpert import create_stellar_dataloaders


DEFAULT_FEATURES_PATH = os.path.join("logs", "2025-07-29", "features.npy")
DEFAULT_TARGET_COLUMNS = ("Teff", "logg", "FeH")


@dataclass(frozen=True)
class DenormalizationSpec:
    scale: float = 1.0
    shift: float = 0.0

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        result = values
        if self.scale is not None:
            result = result * self.scale
        if self.shift is not None:
            result = result + self.shift
        return result


DEFAULT_TARGET_DENORMALIZATION: Dict[str, DenormalizationSpec] = {
    "Teff": DenormalizationSpec(scale=5778.0, shift=0.0),
}


def build_target_denormalizers(
    target_names: Sequence[str],
    config_path: Optional[str],
) -> Dict[str, DenormalizationSpec]:
    mapping: Dict[str, DenormalizationSpec] = {}

    if config_path:
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as handle:
                    raw_config = json.load(handle)
                if not isinstance(raw_config, dict):
                    print(f"Warning: Expected a JSON object in {config_path}; ignoring file.")
                else:
                    for name in target_names:
                        entry = raw_config.get(name)
                        if entry is None:
                            continue

                        if isinstance(entry, dict):
                            scale = entry.get("scale", entry.get("std"))
                            shift = entry.get("shift", entry.get("mean"))
                            scale = 1.0 if scale is None else float(scale)
                            shift = 0.0 if shift is None else float(shift)
                            mapping[name] = DenormalizationSpec(scale=scale, shift=shift)
                        elif isinstance(entry, (int, float)):
                            mapping[name] = DenormalizationSpec(scale=float(entry), shift=0.0)
                        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                            scale, shift = entry
                            mapping[name] = DenormalizationSpec(scale=float(scale), shift=float(shift))
                        else:
                            print(
                                f"Warning: Unsupported denormalization format for target '{name}' in {config_path}; skipping."
                            )
            except (OSError, json.JSONDecodeError) as exc:
                print(f"Warning: Could not load target denormalization config from {config_path} ({exc}).")
        else:
            print(f"Warning: Target denormalization config not found at {config_path}; ignoring.")

    for name in target_names:
        if name not in mapping:
            default_spec = DEFAULT_TARGET_DENORMALIZATION.get(name)
            if default_spec is not None:
                mapping[name] = default_spec

    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a simple MLP baseline on pre-computed spectral features to predict "
            "stellar parameters."
        )
    )
    # Data source options (mutually exclusive groups)
    parser.add_argument(
        "--json_file",
        type=str,
        default=None,
        help="Path to JSON file (new pipeline). If provided, uses create_stellar_dataloaders.",
    )
    parser.add_argument(
        "--features_file",
        type=str,
        default=DEFAULT_FEATURES_PATH,
        help="Path to the .npy file containing spectral features.",
    )
    parser.add_argument(
        "--info_file",
        type=str,
        default=None,
        help="Path to the CSV file with target parameters (legacy). Defaults to info.csv next to features file.",
    )
    parser.add_argument(
        "--feature_stats_file",
        type=str,
        default=None,
        help="Path to .npz file with pre-computed feature normalization stats (mean/std).",
    )
    parser.add_argument(
        "--index_df_file",
        type=str,
        default=None,
        help="Path to CSV file mapping obsid to feature indices.",
    )
    parser.add_argument(
        "--num_spectral_features",
        type=int,
        default=8,
        help="Number of spectral feature tokens (for new pipeline).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Maximum sequence length (for new pipeline).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_TARGET_COLUMNS),
        help="Names of the target columns to predict.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Mini-batch size for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Width of hidden layers.",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=3,
        help="Number of hidden layers.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability applied after each hidden layer.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of data used for training.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of data used for validation.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of data used for testing.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of data loader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for splits and initialization.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g. cuda, cuda:1, cpu). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples to use (useful for quick debugging).",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=10,
        help="Number of validation epochs without improvement before stopping training.",
    )
    parser.add_argument(
        "--skip_normalization",
        action="store_true",
        help="Disable feature normalization computed from the training split.",
    )
    parser.add_argument(
        "--target_denorm_config",
        type=str,
        default=None,
        help=(
            "Optional path to a JSON file describing how to de-normalize targets for reporting/plots. "
            "Values can be specified as {'scale': <float>, 'shift': <float>} or {'std': ..., 'mean': ...}."
        ),
    )
    parser.add_argument(
        "--save_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist the best validation checkpoint to disk (default: True).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("logs", "baseline_mlp"),
        help="Directory used when saving model checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional path to a saved checkpoint for resuming training or running inference.",
    )
    parser.add_argument(
        "--inference_only",
        action="store_true",
        help="Skip training and run evaluation/plots using the checkpoint or freshly initialised model.",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Disable generation of diagnostic plots.",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default=None,
        help="Directory where inference plots will be written (defaults to <output_dir>/plots).",
    )
    parser.add_argument(
        "--num_interpolation_pairs",
        type=int,
        default=25,
        help="Number of sample pairs used in the interpolation analysis.",
    )
    parser.add_argument(
        "--interpolation_alphas",
        type=float,
        nargs="+",
        default=None,
        help="Alpha values (between 0 and 1) for feature interpolation. Defaults to 0-1 with 0.1 steps",
    )
    parser.add_argument(
        "--interpolation_pair_strategy",
        choices=("random", "closest_teff"),
        default="random",
        help="Strategy used when selecting sample pairs for interpolation analysis.",
    )
    parser.add_argument(
        "--interpolation_min_teff_diff",
        type=float,
        default=0.0,
        help=(
            "Minimum difference in Teff (after de-normalization) required between samples in an interpolation pair. "
            "Applies to both random and closest_teff strategies."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class StellarFeatureDataset(Dataset):
    """Simple dataset that returns (features, targets) pairs for stellar parameters."""

    def __init__(
        self,
        features_path: str,
        info_path: str,
        target_columns: Sequence[str],
        max_samples: Optional[int] = None,
    ) -> None:
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")

        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Info CSV not found: {info_path}")

        self._features = np.load(features_path, mmap_mode="r")
        df = pd.read_csv(info_path)

        if len(df) != len(self._features):
            raise ValueError(
                f"Feature array length ({len(self._features)}) does not match CSV rows ({len(df)})."
            )

        for column in target_columns:
            if column not in df.columns:
                raise ValueError(f"Target column '{column}' missing from {info_path}.")

        available_indices = np.arange(len(df))
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive when provided.")
            max_samples = min(max_samples, len(df))
            available_indices = available_indices[:max_samples]

        self._indices = available_indices
        self._targets = df.loc[available_indices, list(target_columns)].to_numpy(dtype=np.float32)
        self._target_columns: List[str] = list(target_columns)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        actual_index = self._indices[index]
        features_np = self._features[actual_index].astype(np.float32, copy=False)
        features = torch.tensor(features_np, dtype=torch.float32)
        targets = torch.from_numpy(self._targets[index])
        return features, targets

    @property
    def input_dim(self) -> int:
        return self._features.shape[1]

    @property
    def output_dim(self) -> int:
        return self._targets.shape[1]

    @property
    def target_columns(self) -> Sequence[str]:
        return self._target_columns

    def get_all_targets(self) -> np.ndarray:
        return self._targets.copy()


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, hidden_layers: int, dropout: float) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def split_dataset(
    dataset: Dataset,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[Subset, Optional[Subset], Optional[Subset]]:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-3):
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    subsets = torch.utils.data.random_split(
        dataset,
        lengths=[train_size, val_size, test_size],
        generator=generator,
    )
    return subsets[0], subsets[1] if val_size > 0 else None, subsets[2] if test_size > 0 else None


@dataclass
class EvalResult:
    loss: float
    mae_per_target: torch.Tensor
    overall_mae: float


def compute_feature_stats(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    running_mean = None
    running_sq_mean = None
    total_samples = 0

    for features, _ in loader:
        features = features.float()
        batch_size = features.shape[0]
        total_samples += batch_size
        batch_mean = features.mean(dim=0)
        batch_sq_mean = (features ** 2).mean(dim=0)

        if running_mean is None:
            running_mean = batch_mean
            running_sq_mean = batch_sq_mean
        else:
            running_mean = (running_mean * (total_samples - batch_size) + batch_mean * batch_size) / total_samples
            running_sq_mean = (running_sq_mean * (total_samples - batch_size) + batch_sq_mean * batch_size) / total_samples

    if running_mean is None:
        raise RuntimeError("Failed to compute feature statistics – training dataset is empty.")

    variance = torch.clamp(running_sq_mean - running_mean ** 2, min=0.0)
    std = torch.sqrt(variance + 1e-6)
    return running_mean, std


def normalize_batch(
    features: torch.Tensor,
    mean: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
) -> torch.Tensor:
    if mean is None or std is None:
        return features
    return (features - mean) / std


def extract_batch_data(batch, use_new_pipeline: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract features and targets from batch, handling both pipeline formats."""
    if use_new_pipeline:
        # New pipeline: batch is a dict
        features = batch['features']  # [B, feature_dim]
        
        # Extract targets from y_numeric [B, 3] for Teff, logg, FeH
        # y_numeric is already normalized in the dataset
        targets = batch['y_numeric']  # [B, 3]
        
        return features, targets
    else:
        # Legacy pipeline: batch could be tuple or list
        if isinstance(batch, dict):
            # Shouldn't happen for legacy, but handle it
            raise ValueError("Legacy pipeline received dict batch - this shouldn't happen")
        
        # Handle tuple/list with variable length
        if len(batch) == 2:
            features, targets = batch
            return features, targets
        else:
            # Debug: print what we got
            raise ValueError(
                f"Legacy pipeline batch has {len(batch)} elements (expected 2). "
                f"Batch type: {type(batch)}, "
                f"Element types: {[type(x) for x in batch]}"
            )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    feature_mean: Optional[torch.Tensor],
    feature_std: Optional[torch.Tensor],
    loss_fn: nn.Module,
    use_new_pipeline: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch in loader:
        features, targets = extract_batch_data(batch, use_new_pipeline)
        
        features = normalize_batch(features.to(device, non_blocking=True), feature_mean, feature_std)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        predictions = model(features)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = features.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        return float("nan")
    return total_loss / total_samples


def evaluate(
    model: nn.Module,
    loader: Optional[DataLoader],
    device: torch.device,
    feature_mean: Optional[torch.Tensor],
    feature_std: Optional[torch.Tensor],
    loss_fn: nn.Module,
    use_new_pipeline: bool = False,
) -> Optional[EvalResult]:
    if loader is None:
        return None

    model.eval()
    total_loss = 0.0
    total_mae = None
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            features, targets = extract_batch_data(batch, use_new_pipeline)
            
            features = normalize_batch(features.to(device, non_blocking=True), feature_mean, feature_std)
            targets = targets.to(device, non_blocking=True)

            predictions = model(features)
            loss = loss_fn(predictions, targets)

            abs_error = torch.abs(predictions - targets)
            batch_size = features.shape[0]

            total_loss += loss.item() * batch_size
            total_samples += batch_size
            if total_mae is None:
                total_mae = abs_error.sum(dim=0)
            else:
                total_mae += abs_error.sum(dim=0)

    if total_samples == 0:
        return EvalResult(loss=float("nan"), mae_per_target=torch.zeros(1), overall_mae=float("nan"))

    mean_loss = total_loss / total_samples
    mean_mae_per_target = total_mae / total_samples
    overall_mae = mean_mae_per_target.mean().item()

    return EvalResult(
        loss=mean_loss,
        mae_per_target=mean_mae_per_target.cpu(),
        overall_mae=overall_mae,
    )


def collect_predictions(
    model: nn.Module,
    loader: Optional[DataLoader],
    device: torch.device,
    feature_mean: Optional[torch.Tensor],
    feature_std: Optional[torch.Tensor],
    use_new_pipeline: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if loader is None:
        return None

    model.eval()
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            features, batch_targets = extract_batch_data(batch, use_new_pipeline)
            
            features = normalize_batch(features.to(device, non_blocking=True), feature_mean, feature_std)
            batch_targets = batch_targets.to(device, non_blocking=True)
            outputs = model(features)
            preds.append(outputs.detach().cpu())
            targets.append(batch_targets.detach().cpu())

    if not preds:
        return None

    predictions = torch.cat(preds, dim=0).numpy()
    target_values = torch.cat(targets, dim=0).numpy()
    return predictions, target_values


def plot_predictions_vs_true(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_names: Sequence[str],
    output_dir: str,
    target_denormalizers: Optional[Mapping[str, DenormalizationSpec]] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    num_targets = predictions.shape[1]
    columns = max(num_targets, 1)
    fig, axes = plt.subplots(1, columns, figsize=(5 * columns, 5), squeeze=False)

    for idx in range(num_targets):
        ax = axes[0, idx]
        pred_vals = predictions[:, idx]
        true_vals = targets[:, idx]

        valid_mask = np.isfinite(pred_vals) & np.isfinite(true_vals)
        if not np.any(valid_mask):
            print(f"Warning: No finite values for target '{target_names[idx]}' – skipping plot.")
            ax.set_visible(False)
            continue

        true_vals = true_vals[valid_mask]
        pred_vals = pred_vals[valid_mask]

        if target_denormalizers:
            denorm_spec = target_denormalizers.get(target_names[idx])
            if denorm_spec is not None:
                true_vals = denorm_spec.denormalize(true_vals)
                pred_vals = denorm_spec.denormalize(pred_vals)

        scatter = ax.scatter(true_vals, pred_vals, alpha=0.45, edgecolor="none", label="Predictions")

        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        line = ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1, label="Ideal")[0]

        residuals = pred_vals - true_vals
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        if np.var(true_vals) > 0 and np.var(pred_vals) > 0:
            corr = np.corrcoef(true_vals, pred_vals)[0, 1]
            r_squared = float(corr ** 2)
        else:
            r_squared = float("nan")
        metrics_label = f"MAE={mae:.3f}, RMSE={rmse:.3f}, R^2={r_squared:.3f}, N={len(true_vals)}"
        metrics_handle = Line2D([], [], color="none", marker="None", linestyle="None", label=metrics_label)

        ax.set_xlabel(f"True {target_names[idx]}")
        ax.set_ylabel(f"Predicted {target_names[idx]}")
        ax.grid(True, alpha=0.3)
        ax.legend(handles=[scatter, line, metrics_handle], loc="best", framealpha=0.95)

    fig.tight_layout()
    plot_path = os.path.join(output_dir, "pred_vs_true.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Prediction scatter saved to {plot_path}")
    return plot_path


def prepare_subset_for_interpolation(subset: Optional[Subset]) -> Optional[Subset]:
    if subset is None:
        return None
    if len(subset) < 2:
        print("Not enough samples for interpolation analysis; need at least two.")
        return None
    return subset


def run_interpolation_analysis(
    model: nn.Module,
    subset: Optional[Subset],
    device: torch.device,
    feature_mean: Optional[torch.Tensor],
    feature_std: Optional[torch.Tensor],
    target_names: Sequence[str],
    output_dir: str,
    num_pairs: int,
    alphas: Sequence[float],
    seed: int,
    pair_strategy: str,
    min_teff_diff: float,
    target_denormalizers: Optional[Mapping[str, DenormalizationSpec]] = None,
) -> Optional[str]:
    subset = prepare_subset_for_interpolation(subset)
    if subset is None:
        return None

    if "Teff" not in target_names or "logg" not in target_names:
        print("Skipping interpolation plot because Teff/logg are not part of the predicted targets.")
        return None

    teff_idx = target_names.index("Teff")
    logg_idx = target_names.index("logg")
    teff_denorm = target_denormalizers.get("Teff") if target_denormalizers else None
    logg_denorm = target_denormalizers.get("logg") if target_denormalizers else None
    feh_idx = target_names.index("FeH") if "FeH" in target_names else None
    feh_denorm = target_denormalizers.get("FeH") if target_denormalizers else None
    min_teff_diff = max(float(min_teff_diff), 0.0)

    if num_pairs <= 0:
        print("Interpolation analysis requested with non-positive number of pairs; skipping.")
        return None

    if not alphas:
        print("No alpha values provided for interpolation; skipping.")
        return None

    # Ensure boundary alphas are present
    alpha_values = sorted(set(float(alpha) for alpha in alphas if 0.0 <= alpha <= 1.0))
    if not alpha_values:
        print("Provided interpolation alphas are outside [0, 1]; skipping.")
        return None
    if alpha_values[0] != 0.0:
        alpha_values.insert(0, 0.0)
    if alpha_values[-1] != 1.0:
        alpha_values.append(1.0)

    total_samples = len(subset)
    if total_samples < 2:
        print("Not enough samples for interpolation analysis; need at least two.")
        return None

    teff_lookup: Dict[int, Optional[float]] = {}
    indices = list(range(total_samples))
    for idx in indices:
        try:
            _, targ = subset[idx]
        except Exception as exc:
            print(f"Warning: Failed to load sample {idx} while preparing interpolation pairs ({exc})")
            teff_lookup[idx] = None
            continue
        teff_value = float(targ[teff_idx].item())
        if math.isnan(teff_value):
            teff_lookup[idx] = None
            continue
        if teff_denorm is not None:
            teff_value = float(teff_denorm.denormalize(teff_value))
        teff_lookup[idx] = teff_value

    def _select_pairs_random() -> List[Tuple[int, int]]:
        rng = random.Random(seed)
        candidates: List[Tuple[int, int]] = []
        seen: set = set()
        attempts = 0
        max_attempts = max(num_pairs * 50, 100)
        while len(candidates) < num_pairs and attempts < max_attempts:
            attempts += 1
            i, j = rng.sample(indices, 2)
            key = tuple(sorted((i, j)))
            if key in seen:
                continue
            teff_i = teff_lookup.get(i)
            teff_j = teff_lookup.get(j)
            if min_teff_diff > 0.0:
                if teff_i is None or teff_j is None:
                    continue
                if abs(teff_i - teff_j) < min_teff_diff:
                    continue
            seen.add(key)
            candidates.append((i, j))
        if len(candidates) < num_pairs:
            print(f"Warning: Only found {len(candidates)} interpolation pairs out of requested {num_pairs}.")
        return candidates

    def _select_pairs_closest_teff() -> List[Tuple[int, int]]:
        valid_indices = [idx for idx in indices if teff_lookup.get(idx) is not None]
        if len(valid_indices) < 2:
            print("Warning: Not enough samples with finite Teff for closest-teff interpolation pairing.")
            return []
        sorted_by_teff = sorted(valid_indices, key=lambda idx: teff_lookup[idx])  # type: ignore[arg-type]
        position_lookup = {idx: pos for pos, idx in enumerate(sorted_by_teff)}
        used_positions: set = set()
        pairs: List[Tuple[int, int]] = []

        for pos, idx_a in enumerate(sorted_by_teff):
            if pos in used_positions:
                continue
            teff_a = teff_lookup[idx_a]
            if teff_a is None:
                continue

            best_idx: Optional[int] = None
            search_radius = 1
            while search_radius < len(sorted_by_teff):
                candidates: List[Tuple[float, int]] = []
                left_pos = pos - search_radius
                right_pos = pos + search_radius

                if left_pos >= 0 and left_pos not in used_positions:
                    idx_left = sorted_by_teff[left_pos]
                    teff_left = teff_lookup[idx_left]
                    if teff_left is not None:
                        diff_left = abs(teff_a - teff_left)
                        if diff_left >= min_teff_diff:
                            candidates.append((diff_left, idx_left))

                if right_pos < len(sorted_by_teff) and right_pos not in used_positions:
                    idx_right = sorted_by_teff[right_pos]
                    teff_right = teff_lookup[idx_right]
                    if teff_right is not None:
                        diff_right = abs(teff_a - teff_right)
                        if diff_right >= min_teff_diff:
                            candidates.append((diff_right, idx_right))

                if candidates:
                    diff, chosen_idx = min(candidates, key=lambda item: item[0])
                    best_idx = chosen_idx
                    break

                search_radius += 1

            if best_idx is None:
                continue

            b_pos = position_lookup.get(best_idx)
            if b_pos is None:
                continue
            used_positions.add(pos)
            used_positions.add(b_pos)
            pairs.append((idx_a, best_idx))
            if len(pairs) >= num_pairs:
                break

        if len(pairs) < num_pairs:
            print(f"Warning: Only found {len(pairs)} interpolation pairs out of requested {num_pairs}.")
        return pairs

    if pair_strategy == "closest_teff":
        pair_indices = _select_pairs_closest_teff()
    else:
        pair_indices = _select_pairs_random()

    if not pair_indices:
        print("Skipping interpolation plot because no interpolation pairs were selected.")
        return None

    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab20")
    legend_labels = set()
    interpolation_records: List[Dict[str, Any]] = []

    def _resolve_dataset_index(subset_obj: Subset, local_idx: int) -> int:
        idx = local_idx
        current: Any = subset_obj
        while isinstance(current, Subset):
            indices = current.indices
            if isinstance(indices, torch.Tensor):
                idx = int(indices[idx].item())
            else:
                idx = int(indices[idx])
            current = current.dataset
        return idx

    def _extract_background_targets(subset_obj: Subset) -> Optional[Tuple[np.ndarray, Sequence[str]]]:
        dataset: Any = subset_obj
        while isinstance(dataset, Subset):
            dataset = dataset.dataset
        if isinstance(dataset, StellarFeatureDataset):
            return dataset.get_all_targets(), dataset.target_columns
        return None

    background_payload = _extract_background_targets(subset)
    if background_payload is not None:
        background_targets, background_columns = background_payload
        if "Teff" in background_columns and "logg" in background_columns:
            bg_teff_idx = background_columns.index("Teff")
            bg_logg_idx = background_columns.index("logg")
            bg_teff_vals = background_targets[:, bg_teff_idx]
            bg_logg_vals = background_targets[:, bg_logg_idx]
            if teff_denorm is not None:
                bg_teff_vals = teff_denorm.denormalize(bg_teff_vals)
            if logg_denorm is not None:
                bg_logg_vals = logg_denorm.denormalize(bg_logg_vals)
            ax.scatter(
                bg_teff_vals,
                bg_logg_vals,
                color="gray",
                alpha=0.18,
                s=18,
                linewidths=0,
                label="_Background",
            )

    with torch.no_grad():
        for pair_idx, (idx_a, idx_b) in enumerate(pair_indices):
            features_a, targets_a = subset[idx_a]
            features_b, targets_b = subset[idx_b]

            features_a = features_a.to(device)
            features_b = features_b.to(device)

            predictions_for_pair = []
            for alpha in alpha_values:
                blended = (1.0 - alpha) * features_a + alpha * features_b
                blended = blended.unsqueeze(0)
                blended = normalize_batch(blended, feature_mean, feature_std)
                pred = model(blended).squeeze(0).detach().cpu().numpy()
                predictions_for_pair.append(pred)

            preds_array = np.stack(predictions_for_pair, axis=0)
            teff_vals = preds_array[:, teff_idx]
            logg_vals = preds_array[:, logg_idx]
            if feh_idx is not None:
                feh_vals = preds_array[:, feh_idx]
            else:
                feh_vals = np.full(len(alpha_values), np.nan, dtype=np.float64)

            if teff_denorm is not None:
                teff_vals = teff_denorm.denormalize(teff_vals)
            if logg_denorm is not None:
                logg_vals = logg_denorm.denormalize(logg_vals)
            if feh_idx is not None and feh_denorm is not None:
                feh_vals = feh_denorm.denormalize(feh_vals)

            color = cmap(pair_idx % cmap.N)
            label = f"{idx_a}→{idx_b}"
            line_label = label if len(legend_labels) < 10 else None
            if line_label is not None:
                legend_labels.add(line_label)

            ax.plot(teff_vals, logg_vals, color=color, linewidth=1.8, alpha=0.85, label=line_label)
            ax.scatter(teff_vals, logg_vals, color=color, s=30, alpha=0.85)
            ax.scatter([teff_vals[0]], [logg_vals[0]], color=color, marker="^", s=70)
            ax.scatter([teff_vals[-1]], [logg_vals[-1]], color=color, marker="s", s=70)

            true_a = targets_a.numpy()
            true_b = targets_b.numpy()
            if teff_denorm is not None:
                true_a_teff = teff_denorm.denormalize(true_a[teff_idx])
                true_b_teff = teff_denorm.denormalize(true_b[teff_idx])
            else:
                true_a_teff = true_a[teff_idx]
                true_b_teff = true_b[teff_idx]
            if logg_denorm is not None:
                true_a_logg = logg_denorm.denormalize(true_a[logg_idx])
                true_b_logg = logg_denorm.denormalize(true_b[logg_idx])
            else:
                true_a_logg = true_a[logg_idx]
                true_b_logg = true_b[logg_idx]
            ax.scatter([true_a_teff], [true_a_logg], facecolors="none", edgecolors=color, s=80, linewidths=1.2)
            ax.scatter([true_b_teff], [true_b_logg], facecolors="none", edgecolors=color, s=80, linewidths=1.2)

            dataset_idx_a = _resolve_dataset_index(subset, idx_a)
            dataset_idx_b = _resolve_dataset_index(subset, idx_b)

            teff_a_lookup = teff_lookup.get(idx_a)
            teff_b_lookup = teff_lookup.get(idx_b)
            record: Dict[str, Any] = {
                "pair": [int(dataset_idx_a), int(dataset_idx_b)],
                "teff_values": [
                    float(teff_a_lookup) if teff_a_lookup is not None and math.isfinite(teff_a_lookup) else None,
                    float(teff_b_lookup) if teff_b_lookup is not None and math.isfinite(teff_b_lookup) else None,
                ],
                "alpha_points": [],
            }

            for alpha, teff_val, logg_val, feh_val in zip(alpha_values, teff_vals, logg_vals, feh_vals):
                alpha_entry: Dict[str, Any] = {
                    "alpha": float(alpha),
                    "teff": float(teff_val) if np.isfinite(teff_val) else None,
                    "logg": float(logg_val) if np.isfinite(logg_val) else None,
                    "feh": float(feh_val) if np.isfinite(feh_val) else None,
                    "text_teff": None,
                    "text_logg": None,
                    "text_feh": None,
                }
                record["alpha_points"].append(alpha_entry)

            interpolation_records.append(record)

    ax.set_xlabel("Predicted Teff")
    ax.set_ylabel("Predicted logg")
    ax.set_title("Feature Interpolation Trajectories (Teff-logg)")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.invert_yaxis()
    if legend_labels:
        ax.legend(loc="best", fontsize=8, ncol=2)

    fig.tight_layout()
    plot_path = os.path.join(output_dir, "interpolation_teff_logg.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Interpolation plot saved to {plot_path}")
    if interpolation_records:
        json_path = os.path.join(output_dir, "interpolation_results.json")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(interpolation_records, handle, indent=2, allow_nan=True)
        print(f"✓ Interpolation JSON saved to {json_path}")
    return plot_path


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    interpolation_alphas = args.interpolation_alphas or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    args.interpolation_alphas = [float(alpha) for alpha in interpolation_alphas]

    if args.inference_only and not args.checkpoint_path:
        print("Warning: Inference-only mode requested without a checkpoint; using current model parameters.")

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Using device: {device}")

    target_denormalizers = build_target_denormalizers(args.targets, args.target_denorm_config)

    # Determine which data pipeline to use
    use_new_pipeline = args.json_file is not None
    
    if use_new_pipeline:
        print("Using NEW data pipeline (create_stellar_dataloaders)...")
        
        # Load features if provided
        features_array = None
        if args.features_file and os.path.exists(args.features_file):
            print(f"Loading features from {args.features_file}")
            features_array = np.load(args.features_file)
        
        # Load feature stats if provided
        feature_stats = None
        if args.feature_stats_file and os.path.exists(args.feature_stats_file):
            print(f"Loading feature stats from {args.feature_stats_file}")
            stats_data = np.load(args.feature_stats_file)
            feature_stats = {'mean': stats_data['mean'], 'std': stats_data['std']}
        
        # Load index_df if provided
        index_df = None
        if args.index_df_file and os.path.exists(args.index_df_file):
            print(f"Loading index_df from {args.index_df_file}")
            index_df = pd.read_csv(args.index_df_file)
        
        # Create dataloaders using new pipeline
        train_loader, val_loader, test_loader = create_stellar_dataloaders(
            json_file=args.json_file,
            features_array=features_array,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.seed,
            num_workers=args.num_workers,
            num_spectral_features=args.num_spectral_features,
            cache_dir=os.path.join(args.output_dir, 'cache'),
            feature_stats=feature_stats,
            index_df=index_df,
            device=device,
        )
        
        # Get input/output dims from first batch
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch['features'].shape[1] if sample_batch['features'] is not None else 2048
        output_dim = len(args.targets)
        
        print(f"Dataset loaded via new pipeline (input_dim={input_dim}, output_dim={output_dim})")
        
        # Feature normalization is handled by the dataset
        feature_mean = feature_std = None
        
    else:
        print("Using LEGACY data pipeline (StellarFeatureDataset)...")
        features_path = args.features_file
        info_path = args.info_file or os.path.join(os.path.dirname(features_path), "info.csv")

        dataset = StellarFeatureDataset(
            features_path=features_path,
            info_path=info_path,
            target_columns=args.targets,
            max_samples=args.max_samples,
        )

        print(
            f"Dataset loaded ({len(dataset)} samples, input_dim = {dataset.input_dim}, "
            f"targets = {list(args.targets)})"
        )

        train_set, val_set, test_set = split_dataset(
            dataset,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        
        input_dim = dataset.input_dim
        output_dim = dataset.output_dim

        pin_memory = device.type == "cuda"

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        ) if val_set is not None else None
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        ) if test_set is not None else None

        feature_mean = feature_std = None
        if not args.skip_normalization:
            print("Computing feature normalization statistics from the training split...")
            feature_mean, feature_std = compute_feature_stats(train_set, args.batch_size, args.num_workers)
            feature_mean = feature_mean.to(device)
            feature_std = feature_std.to(device)

    model = MLPRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    checkpoint_state = None
    if args.checkpoint_path:
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
        checkpoint_state = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_state["model_state"])
        print(f"✓ Loaded checkpoint from {args.checkpoint_path}")

        if not args.inference_only and "optimizer_state" in checkpoint_state:
            try:
                optimizer.load_state_dict(checkpoint_state["optimizer_state"])
                print("✓ Optimizer state restored from checkpoint")
            except Exception as exc:
                print(f"Warning: Could not restore optimizer state ({exc}); continuing without it.")

        loaded_mean = checkpoint_state.get("feature_mean")
        loaded_std = checkpoint_state.get("feature_std")
        if loaded_mean is not None:
            feature_mean = loaded_mean.to(device)
        if loaded_std is not None:
            feature_std = loaded_std.to(device)

    best_val_loss = float("inf")
    best_state = None

    if not args.inference_only and args.epochs > 0:
        non_improve_count = 0
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                feature_mean=feature_mean,
                feature_std=feature_std,
                loss_fn=loss_fn,
                use_new_pipeline=use_new_pipeline,
            )

            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                feature_mean=feature_mean,
                feature_std=feature_std,
                loss_fn=loss_fn,
                use_new_pipeline=use_new_pipeline,
            )

            if val_metrics is not None and val_metrics.loss < best_val_loss:
                best_val_loss = val_metrics.loss
                best_state = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "feature_mean": None if feature_mean is None else feature_mean.detach().cpu(),
                    "feature_std": None if feature_std is None else feature_std.detach().cpu(),
                    "config": vars(args),
                }
                non_improve_count = 0
            elif val_metrics is not None:
                non_improve_count += 1
                if non_improve_count > args.early_stopping:
                    print("Early stopping triggered.")
                    break

            if val_metrics is None:
                print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f}")
            else:
                mae_values = ", ".join(
                    f"{target}={val_metrics.mae_per_target[i].item():.6f}"
                    for i, target in enumerate(args.targets)
                )
                print(
                    f"Epoch {epoch:03d}: train_loss={train_loss:.6f}, "
                    f"val_loss={val_metrics.loss:.6f}, "
                    f"val_mae={val_metrics.overall_mae:.6f} ({mae_values})"
                )

        if best_state is None:
            best_state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": args.epochs,
                "feature_mean": None if feature_mean is None else feature_mean.detach().cpu(),
                "feature_std": None if feature_std is None else feature_std.detach().cpu(),
                "config": vars(args),
            }
    else:
        if args.inference_only:
            print("Inference-only mode: skipping training loop.")
        elif args.epochs <= 0:
            print("Epoch count set to 0; skipping training loop.")

    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
        stored_mean = best_state.get("feature_mean")
        stored_std = best_state.get("feature_std")
        if stored_mean is not None:
            feature_mean = stored_mean.to(device)
        if stored_std is not None:
            feature_std = stored_std.to(device)
        if args.save_model:
            ensure_output_dir(args.output_dir)
            checkpoint_path = os.path.join(args.output_dir, "baseline_mlp.pt")
            torch.save(best_state, checkpoint_path)
            print(f"Best model saved to {checkpoint_path}")

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        feature_mean=feature_mean,
        feature_std=feature_std,
        loss_fn=loss_fn,
        use_new_pipeline=use_new_pipeline,
    )

    if test_metrics is not None:
        mae_values = ", ".join(
            f"{target}={test_metrics.mae_per_target[i].item():.6f}"
            for i, target in enumerate(args.targets)
        )
        print(
            f"Test loss: {test_metrics.loss:.6f}, "
            f"test MAE: {test_metrics.overall_mae:.6f} ({mae_values})"
        )

    plot_loader = test_loader if test_loader is not None else val_loader
    # For interpolation analysis: only available in legacy pipeline
    if use_new_pipeline:
        plot_subset = None  # New pipeline doesn't expose datasets
    else:
        plot_subset = test_set if test_set is not None else val_set
    if not args.skip_plots:
        if plot_loader is None:
            print("Skipping plot generation because no validation/test loader is available.")
        else:
            plots_dir = args.plots_dir or os.path.join(args.output_dir, "plots")
            prediction_data = collect_predictions(
                model=model,
                loader=plot_loader,
                device=device,
                feature_mean=feature_mean,
                feature_std=feature_std,
                use_new_pipeline=use_new_pipeline,
            )
            if prediction_data is not None:
                preds_array, targets_array = prediction_data
                plot_predictions_vs_true(
                    predictions=preds_array,
                    targets=targets_array,
                    target_names=args.targets,
                    output_dir=plots_dir,
                    target_denormalizers=target_denormalizers,
                )
            run_interpolation_analysis(
                model=model,
                subset=plot_subset,
                device=device,
                feature_mean=feature_mean,
                feature_std=feature_std,
                target_names=args.targets,
                output_dir=plots_dir,
                num_pairs=args.num_interpolation_pairs,
                alphas=args.interpolation_alphas,
                seed=args.seed,
                pair_strategy=args.interpolation_pair_strategy,
                min_teff_diff=args.interpolation_min_teff_diff,
                target_denormalizers=target_denormalizers,
            )


if __name__ == "__main__":
    main()
