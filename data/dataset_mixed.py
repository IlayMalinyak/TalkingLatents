"""Mixed single-star and comparative dataset for stellar QA models."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .dataset_interpert import StellarQuestionsDataset
from .dataset_comparative import StellarComparativeDataset


@dataclass
class MixedSample:
    """Container describing the normalized payload returned by the dataset."""

    mode: str
    tokens: torch.Tensor
    target_ids: torch.Tensor
    span_indices: torch.Tensor
    x_raw: Optional[torch.Tensor] = None
    x_raw_a: Optional[torch.Tensor] = None
    x_raw_b: Optional[torch.Tensor] = None
    y_numeric: Optional[torch.Tensor] = None
    y_numeric_a: Optional[torch.Tensor] = None
    y_numeric_b: Optional[torch.Tensor] = None
    pair_label: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation (useful for collate functions)."""

        return {
            "mode": self.mode,
            "tokens": self.tokens,
            "target_ids": self.target_ids,
            "span_indices": self.span_indices,
            "x_raw": self.x_raw,
            "x_raw_a": self.x_raw_a,
            "x_raw_b": self.x_raw_b,
            "y_numeric": self.y_numeric,
            "y_numeric_a": self.y_numeric_a,
            "y_numeric_b": self.y_numeric_b,
            "pair_label": self.pair_label,
            "metadata": self.metadata,
        }


class MixedStellarQADataset(Dataset):
    """Blend single-star and comparative QA samples with a configurable mix."""

    def __init__(
        self,
        single_dataset: Dataset,
        comparative_dataset: Optional[Dataset] = None,
        single_sample_prob: float = 0.5,
        seed: int = 1234,
        length_strategy: str = "max",
        numeric_keys: Optional[Iterable[str]] = None,
    ) -> None:
        if single_dataset is None:
            raise ValueError("single_dataset must be provided")
        if not 0.0 <= single_sample_prob <= 1.0:
            raise ValueError("single_sample_prob must be in [0, 1]")
        if comparative_dataset is None and single_sample_prob < 1.0:
            single_sample_prob = 1.0

        self.single_dataset = single_dataset
        self.comparative_dataset = comparative_dataset
        self.single_sample_prob = float(single_sample_prob)
        self.seed = int(seed)
        self.length_strategy = length_strategy
        self.numeric_keys = tuple(numeric_keys) if numeric_keys is not None else None

        single_len = len(self.single_dataset)
        comp_len = len(self.comparative_dataset) if self.comparative_dataset is not None else 0
        if length_strategy == "max":
            self._length = max(single_len, comp_len) if comp_len > 0 else single_len
        elif length_strategy == "sum":
            self._length = single_len + comp_len
        elif length_strategy == "min":
            self._length = min(single_len, comp_len) if comp_len > 0 else single_len
        else:
            raise ValueError(
                "length_strategy must be one of {'max', 'sum', 'min'}; got " f"{length_strategy}"
            )
        if self._length == 0:
            raise ValueError("Resulting dataset is empty; check input datasets")

    def __len__(self) -> int:
        return self._length

    def update_single_probability(self, new_prob: float) -> None:
        """Update the mixing probability (useful for curriculum schedules)."""

        if not 0.0 <= new_prob <= 1.0:
            raise ValueError("new_prob must be in [0, 1]")
        if self.comparative_dataset is None and new_prob < 1.0:
            raise ValueError("Comparative dataset missing; cannot lower single probability")
        self.single_sample_prob = float(new_prob)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        mode = self._select_mode(idx)
        if mode == "single":
            base_idx = idx % len(self.single_dataset)
            raw = self.single_dataset[base_idx]
            sample = self._normalize_single(raw)
        else:
            assert self.comparative_dataset is not None
            base_idx = idx % len(self.comparative_dataset)
            raw = self.comparative_dataset[base_idx]
            sample = self._normalize_comparative(raw)
        return sample.to_dict()

    def _select_mode(self, idx: int) -> str:
        if self.comparative_dataset is None:
            return "single"
        rng = random.Random(self.seed + idx)
        return "single" if rng.random() < self.single_sample_prob else "comparative"

    def _normalize_single(self, raw: Dict[str, Any]) -> MixedSample:
        tokens = raw["input_ids"].clone()
        target_ids = raw.get("target_ids", tokens.new_full(tokens.shape, -100))
        span_start = int(raw.get("answer_start_idx", raw.get("target_length", 0)))
        span_length = int(raw.get("target_length", 0))
        span_end = max(span_start + span_length, span_start)
        span_indices = torch.tensor([span_start, span_end], dtype=torch.long)

        features = self._coerce_tensor(raw.get("features") or raw.get("masked_spectra"))
        numeric = self._extract_numeric(raw.get("stellar_data"))

        metadata = {
            "input_text": raw.get("input_text"),
            "target_text": raw.get("target_text"),
            "question_start_idx": raw.get("question_start_idx"),
            "feature_start_idx": raw.get("feature_start_idx"),
            "feature_length": raw.get("feature_length"),
            "masked_spectra": raw.get("masked_spectra"),
            "spectra": raw.get("spectra"),
            "answer_start_idx": raw.get("answer_start_idx"),
            "target_length": raw.get("target_length"),
            "raw": raw,
        }
        return MixedSample(
            mode="single",
            tokens=tokens,
            target_ids=target_ids,
            span_indices=span_indices,
            x_raw=features,
            y_numeric=numeric,
            metadata=metadata,
        )

    def _normalize_comparative(self, raw: Dict[str, Any]) -> MixedSample:
        tokens = raw["input_ids"].clone()
        target_ids = raw.get("target_ids", tokens.new_full(tokens.shape, -100))
        span_start = int(raw.get("answer_start_idx", 0))
        span_length = int(raw.get("target_length", 0))
        span_end = max(span_start + span_length, span_start)
        span_indices = torch.tensor([span_start, span_end], dtype=torch.long)

        features_a = self._coerce_tensor(
            raw.get("features_a")
            or raw.get("star_a_features")
            or raw.get("masked_spectra_a")
        )
        features_b = self._coerce_tensor(
            raw.get("features_b")
            or raw.get("star_b_features")
            or raw.get("masked_spectra_b")
        )
        numeric_a = self._extract_numeric(raw.get("star_a_params"))
        numeric_b = self._extract_numeric(raw.get("star_b_params"))
        pair_label = raw.get("pair_label") or raw.get("target_index")
        if isinstance(pair_label, int):
            pair_label = torch.tensor(pair_label, dtype=torch.long)
        elif isinstance(pair_label, torch.Tensor):
            pair_label = pair_label.clone()

        metadata = {
            "question_text": raw.get("question_text"),
            "input_text": raw.get("input_text"),
            "target_text": raw.get("target_text"),
            "options": raw.get("options"),
            "star_a_feature_indices": raw.get("star_a_feature_indices"),
            "star_b_feature_indices": raw.get("star_b_feature_indices"),
            "masked_spectra_a": raw.get("masked_spectra_a"),
            "masked_spectra_b": raw.get("masked_spectra_b"),
            "answer_start_idx": raw.get("answer_start_idx"),
            "target_length": raw.get("target_length"),
            "raw": raw,
        }
        return MixedSample(
            mode="comparative",
            tokens=tokens,
            target_ids=target_ids,
            span_indices=span_indices,
            x_raw_a=features_a,
            x_raw_b=features_b,
            y_numeric_a=numeric_a,
            y_numeric_b=numeric_b,
            pair_label=pair_label,
            metadata=metadata,
        )

    def _coerce_tensor(self, value: Any) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.clone()
        return torch.as_tensor(value)

    def _extract_numeric(self, source: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
        if not source:
            return None
        keys = self.numeric_keys or tuple(k for k, v in source.items() if isinstance(v, (int, float)))
        values = []
        for key in keys:
            val = source.get(key)
            if val is None:
                continue
            if isinstance(val, (int, float)):
                values.append(float(val))
        if not values:
            return None
        return torch.tensor(values, dtype=torch.float32)


def _pad_stack(values: Sequence[Optional[torch.Tensor]], *, dtype: Optional[torch.dtype] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    present: List[torch.Tensor] = [v for v in values if v is not None]
    if not present:
        return None, None
    ref = present[0]
    target_shape = ref.shape
    dtype = dtype or ref.dtype
    batch = torch.zeros((len(values),) + target_shape, dtype=dtype)
    mask = torch.zeros(len(values), dtype=torch.bool)
    for idx, value in enumerate(values):
        if value is None:
            continue
        batch[idx] = value.to(dtype)
        mask[idx] = True
    return batch, mask


def _pad_scalar(values: Sequence[Optional[torch.Tensor]], *, dtype: torch.dtype) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    present = [v for v in values if v is not None]
    if not present:
        return None, None
    batch = torch.zeros(len(values), dtype=dtype)
    mask = torch.zeros(len(values), dtype=torch.bool)
    for idx, value in enumerate(values):
        if value is None:
            continue
        if value.dim() != 0:
            raise ValueError("Expected scalar tensor for pair labels")
        batch[idx] = value.to(dtype)
        mask[idx] = True
    return batch, mask


def _pad_indices(index_lists: Sequence[List[int]], fill_value: int = -1) -> torch.Tensor:
    max_len = max((len(lst) for lst in index_lists), default=0)
    if max_len == 0:
        return torch.full((len(index_lists), 1), fill_value, dtype=torch.long)
    tensor = torch.full((len(index_lists), max_len), fill_value, dtype=torch.long)
    for row, indices in enumerate(index_lists):
        if not indices:
            continue
        tensor[row, :len(indices)] = torch.tensor(indices, dtype=torch.long)
    return tensor


def collate_mixed_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    tokens = torch.stack([item["tokens"] for item in batch])
    target_ids = torch.stack([item["target_ids"] for item in batch])
    span_indices = torch.stack([item["span_indices"] for item in batch])

    modes = [item["mode"] for item in batch]
    single_mask = torch.tensor([mode == "single" for mode in modes], dtype=torch.bool)
    comp_mask = ~single_mask

    x_raw_values = [item.get("x_raw") for item in batch]
    x_raw, x_raw_present = _pad_stack(x_raw_values)

    x_raw_a_values = [item.get("x_raw_a") for item in batch]
    x_raw_a, x_raw_a_present = _pad_stack(x_raw_a_values)
    x_raw_b_values = [item.get("x_raw_b") for item in batch]
    x_raw_b, x_raw_b_present = _pad_stack(x_raw_b_values)

    y_numeric_values = [item.get("y_numeric") for item in batch]
    y_numeric, y_numeric_present = _pad_stack(y_numeric_values, dtype=torch.float32)

    y_numeric_a_values = [item.get("y_numeric_a") for item in batch]
    y_numeric_a, y_numeric_a_present = _pad_stack(y_numeric_a_values, dtype=torch.float32)
    y_numeric_b_values = [item.get("y_numeric_b") for item in batch]
    y_numeric_b, y_numeric_b_present = _pad_stack(y_numeric_b_values, dtype=torch.float32)

    pair_label_values = [item.get("pair_label") for item in batch]
    pair_labels, pair_label_present = _pad_scalar(pair_label_values, dtype=torch.long)

    metadata = [item.get("metadata", {}) for item in batch]

    feature_start_indices: List[int] = []
    feature_lengths: List[int] = []
    answer_start_indices: List[int] = []
    target_lengths: List[int] = []
    star_a_lists: List[List[int]] = []
    star_b_lists: List[List[int]] = []

    masked_spectra_values: List[Optional[torch.Tensor]] = []
    masked_spectra_a_values: List[Optional[torch.Tensor]] = []
    masked_spectra_b_values: List[Optional[torch.Tensor]] = []

    for meta, span in zip(metadata, span_indices):
        raw_meta = meta.get("raw", {}) if meta else {}
        feature_start_indices.append(int(meta.get("feature_start_idx", -1)) if meta else -1)
        feature_lengths.append(int(meta.get("feature_length", 0) or 0) if meta else 0)
        answer_start_indices.append(int(span[0].item()))
        target_lengths.append(int((span[1] - span[0]).item()))

        def _to_list(value: Any) -> List[int]:
            if value is None:
                return []
            if isinstance(value, torch.Tensor):
                return value.view(-1).tolist()
            if isinstance(value, (list, tuple)):
                return [int(v) for v in value]
            return []

        star_a_lists.append(_to_list(meta.get("star_a_feature_indices") if meta else None))
        star_b_lists.append(_to_list(meta.get("star_b_feature_indices") if meta else None))

        masked_spectra_values.append(meta.get("masked_spectra") if meta else None)
        masked_spectra_a_values.append(meta.get("masked_spectra_a") if meta else None)
        masked_spectra_b_values.append(meta.get("masked_spectra_b") if meta else None)

    feature_start_indices_tensor = torch.tensor(feature_start_indices, dtype=torch.long)
    feature_lengths_tensor = torch.tensor(feature_lengths, dtype=torch.long)
    answer_start_indices_tensor = torch.tensor(answer_start_indices, dtype=torch.long)
    target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long)

    star_a_feature_indices = _pad_indices(star_a_lists)
    star_b_feature_indices = _pad_indices(star_b_lists)

    masked_spectra, masked_spectra_present = _pad_stack(masked_spectra_values, dtype=torch.float32)
    masked_spectra_a, masked_spectra_a_present = _pad_stack(masked_spectra_a_values, dtype=torch.float32)
    masked_spectra_b, masked_spectra_b_present = _pad_stack(masked_spectra_b_values, dtype=torch.float32)

    return {
        "mode": modes,
        "mode_mask_single": single_mask,
        "mode_mask_comparative": comp_mask,
        "input_ids": tokens,
        "target_ids": target_ids,
        "span_indices": span_indices,
        "feature_start_indices": feature_start_indices_tensor,
        "feature_lengths": feature_lengths_tensor,
        "answer_start_indices": answer_start_indices_tensor,
        "target_lengths": target_lengths_tensor,
        "star_a_feature_indices": star_a_feature_indices,
        "star_b_feature_indices": star_b_feature_indices,
        "x_raw": x_raw,
        "x_raw_present": x_raw_present,
        "x_raw_a": x_raw_a,
        "x_raw_a_present": x_raw_a_present,
        "x_raw_b": x_raw_b,
        "x_raw_b_present": x_raw_b_present,
        "y_numeric": y_numeric,
        "y_numeric_present": y_numeric_present,
        "y_numeric_a": y_numeric_a,
        "y_numeric_a_present": y_numeric_a_present,
        "y_numeric_b": y_numeric_b,
        "y_numeric_b_present": y_numeric_b_present,
        "pair_labels": pair_labels,
        "pair_label_present": pair_label_present,
        "masked_spectra": masked_spectra,
        "masked_spectra_present": masked_spectra_present,
        "masked_spectra_a": masked_spectra_a,
        "masked_spectra_a_present": masked_spectra_a_present,
        "masked_spectra_b": masked_spectra_b,
        "masked_spectra_b_present": masked_spectra_b_present,
        "metadata": metadata,
    }


def create_mixed_dataloaders(
    single_kwargs: Dict[str, Any],
    comparative_kwargs: Optional[Dict[str, Any]] = None,
    batch_size: int = 32,
    single_sample_prob: float = 0.5,
    seed: int = 1234,
    length_strategy: str = "max",
    num_workers: int = 0,
    persistent_workers: bool = False,
    drop_last: bool = False,
    pin_memory: bool = False,
    numeric_keys: Optional[Iterable[str]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    single_train = StellarQuestionsDataset(**{**single_kwargs, "split": "train"})
    single_val = StellarQuestionsDataset(**{**single_kwargs, "split": "val"})
    single_test = StellarQuestionsDataset(**{**single_kwargs, "split": "test"})

    comparative_train = comparative_val = comparative_test = None
    if comparative_kwargs is not None:
        comparative_train = StellarComparativeDataset(**{**comparative_kwargs, "split": "train"})
        comparative_val = StellarComparativeDataset(**{**comparative_kwargs, "split": "val"})
        comparative_test = StellarComparativeDataset(**{**comparative_kwargs, "split": "test"})

    train_dataset = MixedStellarQADataset(
        single_dataset=single_train,
        comparative_dataset=comparative_train,
        single_sample_prob=single_sample_prob,
        seed=seed,
        length_strategy=length_strategy,
        numeric_keys=numeric_keys,
    )

    val_dataset = MixedStellarQADataset(
        single_dataset=single_val,
        comparative_dataset=comparative_val,
        single_sample_prob=single_sample_prob,
        seed=seed + 1,
        length_strategy=length_strategy,
        numeric_keys=numeric_keys,
    )

    test_dataset = MixedStellarQADataset(
        single_dataset=single_test,
        comparative_dataset=comparative_test,
        single_sample_prob=single_sample_prob,
        seed=seed + 2,
        length_strategy=length_strategy,
        numeric_keys=numeric_keys,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=drop_last,
        collate_fn=collate_mixed_fn,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
