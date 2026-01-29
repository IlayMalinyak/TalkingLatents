"""Reusable helper for normalising spectral feature vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch


@dataclass
class _Stats:
    mean: np.ndarray
    std: np.ndarray
    epsilon: float


class FeatureNormalizer:
    """Compute and apply per-dimension mean/std scaling for feature arrays."""

    def __init__(self, enabled: bool = True, epsilon: float = 1e-6) -> None:
        self.enabled = bool(enabled)
        self.epsilon = float(epsilon)
        self._stats: Optional[_Stats] = None

    def initialize(
        self,
        features_array: Optional[np.ndarray],
        feature_indices: Iterable[int],
        provided_stats: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Derive stats from the provided subset or reuse pre-computed ones."""
        if not self.enabled or features_array is None:
            self._stats = None
            return

        if provided_stats is not None and "mean" in provided_stats and "std" in provided_stats:
            mean = np.asarray(provided_stats["mean"], dtype=np.float32).copy()
            std = np.asarray(provided_stats["std"], dtype=np.float32).copy()
            eps = float(provided_stats.get("epsilon", self.epsilon))
            std = np.where(std < eps, 1.0, std)
            self._stats = _Stats(mean=mean, std=std, epsilon=eps)
            return

        indices = list(feature_indices or [])
        if not indices:
            # Nothing to fit on; keep normalisation disabled.
            self.enabled = False
            self._stats = None
            return

        subset = features_array[indices].astype(np.float32)
        mean = subset.mean(axis=0)
        std = subset.std(axis=0)
        std = np.where(std < self.epsilon, 1.0, std)
        self._stats = _Stats(mean=mean, std=std, epsilon=self.epsilon)

    def is_active(self) -> bool:
        return self.enabled and self._stats is not None

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Return a normalised numpy array copy."""
        arr = np.asarray(features, dtype=np.float32).copy()
        if not self.is_active():
            return arr
        return (arr - self._stats.mean) / self._stats.std

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalise a tensor (used for logging/inference)."""
        if not self.is_active():
            return tensor
        mean = torch.from_numpy(self._stats.mean).to(tensor.device, tensor.dtype)
        std = torch.from_numpy(self._stats.std).to(tensor.device, tensor.dtype)
        return tensor * std + mean

    def get_stats(self, copy: bool = True) -> Optional[Dict[str, np.ndarray]]:
        if not self.is_active():
            return None
        if not copy:
            return {
                "mean": self._stats.mean,
                "std": self._stats.std,
                "epsilon": self._stats.epsilon,
            }
        return {
            "mean": self._stats.mean.copy(),
            "std": self._stats.std.copy(),
            "epsilon": self._stats.epsilon,
        }
