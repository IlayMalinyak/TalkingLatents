import textwrap

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Any, Dict, List, Optional, Tuple, Type

import os
os.system('pip install tiktoken fairscale fire blobfile')
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 


from data.dataset_interpert import StellarQuestionsDataset


class LateFusionDataset(StellarQuestionsDataset):
    """
    Dataset for Perceiver late-fusion training.

    Differences from StellarQuestionsDataset:
      * Text consists only of the descriptive portion (no Q/A split).
      * Always returns raw spectra (unmasked) plus a masked copy.
      * Provides attention masks instead of autoregressive targets.
    """

    def __init__(
        self,
        json_file: str,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
        spectral_transforms: Optional[Any] = None,
        cache_dir: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        max_length: int = 512,
        **kwargs,
    ):
        super().__init__(
            json_file=json_file,
            features_array=None,
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
            spectral_transforms=spectral_transforms,
            filter_valid_descriptions=True,
            cache_dir=cache_dir,
            tokenizer_path=tokenizer_path,
            max_length=max_length,
            num_spectral_features=0,
            normalize_features=False,
            **kwargs,
        )

    def _select_description_text(self, sample: Dict[str, Any]) -> str:
        parsed = self.parse_description_text(sample.get("description", ""))
        text = parsed.get("answer") or sample.get("description", "")
        return text.strip()

    @staticmethod
    def _build_attention_mask(tokens: torch.Tensor) -> torch.Tensor:
        return (tokens != 0).long()

    def _resolve_obsid(self, sample: Dict[str, Any]) -> Tuple[Optional[Any], str]:
        if "obsid" in sample and sample["obsid"] is not None:
            return sample["obsid"], "obsid"
        for key in ("APOGEE_ID", "apogee_id"):
            if key in sample and sample[key]:
                return sample[key], "APOGEE_ID"
        return None, "obsid"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_idx = self.split_indices[idx]
        sample = self.raw_data[sample_idx]

        text = self._select_description_text(sample)
        input_ids, _ = self._tokenize_text(text, bos=True)
        attention_mask = self._build_attention_mask(input_ids)

        obsid, id_type = self._resolve_obsid(sample)
        if obsid is None:
            raise ValueError(f"Sample {sample_idx} missing obsid/APOGEE_ID")
        spectra, masked_spectra, meta = self.get_raw_spectra(obsid, id_type=id_type)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "spectral_data": spectra.squeeze(),
            "masked_spectra": masked_spectra.squeeze(),
            "text": text,
            "obsid": obsid,
            "df_index": sample.get("index"),
            "stellar_data": sample.get("stellar_data", {}),
            "metadata": meta,
        }


def late_fusion_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    spectral_data = torch.stack([item["spectral_data"] for item in batch])
    masked_spectra = torch.stack([item["masked_spectra"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "spectral_data": spectral_data,
        "masked_spectra": masked_spectra,
        "texts": [item["text"] for item in batch],
        "obsids": [item["obsid"] for item in batch],
        "df_indices": [item["df_index"] for item in batch],
        "stellar_data": [item["stellar_data"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def create_late_fusion_dataloaders(
    json_file: str,
    batch_size: int = 8,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    num_workers: int = 0,
    cache_dir: Optional[str] = None,
    world_size: int = 1,
    dataset_cls: Type[LateFusionDataset] = LateFusionDataset,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset_kwargs = dict(
        json_file=json_file,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        cache_dir=cache_dir,
        **dataset_kwargs,
    )

    train_dataset = dataset_cls(split="train", **dataset_kwargs)
    val_dataset = dataset_cls(split="val", **dataset_kwargs)
    test_dataset = dataset_cls(split="test", **dataset_kwargs)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=late_fusion_collate_fn,
    )

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, shuffle=True, seed=random_state, drop_last=False
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, shuffle=False, seed=random_state, drop_last=False
        )
        test_sampler = DistributedSampler(
            test_dataset, num_replicas=world_size, shuffle=False, seed=random_state, drop_last=False
        )

        train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
        val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)
        test_loader = DataLoader(test_dataset, sampler=test_sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from pathlib import Path

    json_path = Path("/data/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json")
    tokenizer_path = Path("/data/.llama/Llama3.2-1B/tokenizer.model")

    dataset = LateFusionDataset(
        json_file=str(json_path),
        split="train",
        tokenizer_path=str(tokenizer_path) if tokenizer_path.exists() else None,
        max_length=256,
    )

    print(f"Dataset size: {len(dataset)} samples")
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("input_ids shape:", sample["input_ids"].shape)
    print("spectral_data shape:", sample["spectral_data"].shape)

    loader, _, _ = create_late_fusion_dataloaders(
        json_file=str(json_path),
        batch_size=2,
        tokenizer_path=str(tokenizer_path) if tokenizer_path.exists() else None,
        max_length=256,
    )
    first_batch = next(iter(loader))
    print("Batch input_ids shape:", first_batch["input_ids"].shape)
    print("Example text:", first_batch["texts"][0])
    print("Batch spectral_data shape:", first_batch["spectral_data"].shape)
    print("batch stellar data exampe: ", first_batch["stellar_data"][0])
    print("batch meta data exampe: ", first_batch["metadata"][0].keys())


    try:
        import matplotlib.pyplot as plt

        figs_dir = Path("figs")
        figs_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(min(3, len(dataset))):
            item = dataset[idx]
            spectrum = item["spectral_data"].squeeze().cpu().numpy()
            wv = item['metadata']['wavelength'].squeeze()
            spectrum = spectrum[:len(wv)]
            

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(wv, spectrum)
            ax.set_xlabel("Wavelength (A)")
            ax.set_ylabel("Flux")
            ax.set_title(f"Spectra Sample {idx}")

            caption = textwrap.fill(item["text"], width=100)
            fig.text(0.25, 0.2, caption, ha="left", va="bottom", fontsize=8)

            out_path = figs_dir / f"spectra_text_sample_{idx}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved plot: {out_path}")
    except ImportError:
        print("matplotlib not installed; skipping spectra plot generation.")
