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
      * Supports follow-up questions with three modes:
        - "stellar_type": Generate stellar classification questions (e.g., "What stellar type?")
        - "description": Use Q&A from a second JSON file (long descriptions)
        - "mixed": Randomly choose between stellar_type and description (50/50)
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
        # Cycle-fusion follow-up options
        use_followups: bool = False,
        followup_prob: float = 1.0,
        max_followups: int = 1,
        followup_seed: int = 42,
        append_answer_prompt: bool = True,
        # Optional second JSON for follow-up Q&A
        followup_json_file: Optional[str] = None,
        followup_mode: str = "mixed",  # "stellar_type", "description", or "mixed" (50/50)
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
        # Follow-up generation controls
        self.use_followups = bool(use_followups)
        self.followup_prob = float(followup_prob)
        self.max_followups = int(max_followups)
        self.followup_seed = int(followup_seed)
        self.append_answer_prompt = bool(append_answer_prompt)
        self.followup_json_file = followup_json_file
        self.followup_mode = followup_mode
        # Local RNG per-split for reproducibility
        import random as _random
        self._rng = _random.Random(self.followup_seed + (0 if split == "train" else (1 if split == "val" else 2)))

        # Load follow-up JSON if provided
        self.followup_data = None
        if self.followup_json_file and self.use_followups:
            self._load_followup_json()

        # Print follow-up mode info
        if self.use_followups:
            if self.followup_data is not None:
                print(f"Follow-up mode: {self.followup_mode} (with {len(self.followup_data)} description samples loaded)")
            else:
                print(f"Follow-up mode: stellar_type only (no followup_json_file provided)")

    def _load_followup_json(self):
        """Load the follow-up JSON file for Q&A pairs."""
        import json
        try:
            with open(self.followup_json_file, 'r') as f:
                self.followup_data = json.load(f)
            print(f"Loaded {len(self.followup_data)} follow-up samples from {self.followup_json_file}")
        except Exception as e:
            print(f"Warning: Could not load follow-up JSON {self.followup_json_file}: {e}")
            self.followup_data = None

    def _select_description_text(self, sample: Dict[str, Any]) -> str:
        parsed = self.parse_description_text(sample.get("description", ""))
        text = parsed.get("answer") or sample.get("description", "")
        return text.strip()

    def _get_followup_from_description(self, sample_idx: int) -> Optional[Tuple[str, str]]:
        """Extract question and answer from the follow-up JSON file."""
        if self.followup_data is None or sample_idx >= len(self.followup_data):
            return None

        followup_sample = self.followup_data[sample_idx]
        description = followup_sample.get("description", "")

        if not description:
            return None

        # Parse the description to get question and answer
        parsed = self.parse_description_text(description)
        question = parsed.get("question", "")
        answer = parsed.get("answer", "")

        if not question or not answer:
            return None

        return question, answer

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

        sample_out: Dict[str, Any] = {
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

        # Optional: synthesize a follow-up QA pair for cycle-fusion CE supervision
        if self.use_followups and (self._rng.random() < self.followup_prob):
            try:
                q = None
                a = None

                # Decide which mode to use based on followup_mode
                use_description = False
                if self.followup_mode == "description" and self.followup_data is not None:
                    use_description = True
                elif self.followup_mode == "mixed" and self.followup_data is not None:
                    # 50% probability for description, 50% for stellar type
                    use_description = self._rng.random() < 0.5

                # Try to use description from follow-up JSON file
                if use_description:
                    qa_pair = self._get_followup_from_description(sample_idx)
                    if qa_pair is not None:
                        q, a = qa_pair

                # Generate stellar type questions from templates (if not using description or as fallback)
                if q is None or a is None:
                    from src.follow_up_templates import create_follow_up_specs, PARAM_KEY_ALIASES
                    # Extract known params from stellar_data dict
                    raw_params = sample_out.get("stellar_data", {}) or {}
                    params = {}
                    for canonical, aliases in PARAM_KEY_ALIASES.items():
                        val = None
                        for key in aliases:
                            if key in raw_params and raw_params[key] is not None:
                                try:
                                    val = float(raw_params[key])
                                    break
                                except Exception:
                                    continue
                        params[canonical] = val
                    specs = create_follow_up_specs(params, self._rng, max_pairs=max(1, self.max_followups), include_answers=True)
                    if specs:
                        # Pick first spec for simplicity
                        q = specs[0].get("question", "")
                        a = specs[0].get("answer", "")

                # Tokenize if we have valid Q&A
                if q and a:
                    if self.append_answer_prompt:
                        q = f"{q}\nAnswer:"
                    q_ids, _ = self._tokenize_text(q, bos=True)
                    a_ids, _ = self._tokenize_text(a, bos=False)
                    sample_out["followup_question_ids"] = q_ids
                    sample_out["followup_answer_ids"] = a_ids
                    sample_out["has_followup"] = torch.tensor(1, dtype=torch.uint8)
                else:
                    sample_out["has_followup"] = torch.tensor(0, dtype=torch.uint8)
            except Exception:
                sample_out["has_followup"] = torch.tensor(0, dtype=torch.uint8)

        return sample_out


def late_fusion_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    spectral_data = torch.stack([item["spectral_data"] for item in batch])
    masked_spectra = torch.stack([item["masked_spectra"] for item in batch])

    out: Dict[str, Any] = {
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

    # Optional follow-up tensors: collate only if present for at least one item
    if any("followup_question_ids" in item for item in batch):
        fq_tensors = [item.get("followup_question_ids", torch.zeros_like(input_ids[0])) for item in batch]
        out["followup_question_ids"] = torch.stack(fq_tensors)
    if any("followup_answer_ids" in item for item in batch):
        fa_tensors = [item.get("followup_answer_ids", torch.zeros_like(input_ids[0])) for item in batch]
        out["followup_answer_ids"] = torch.stack(fa_tensors)
    if any("has_followup" in item for item in batch):
        out["has_followup"] = torch.stack([item.get("has_followup", torch.tensor(0, dtype=torch.uint8)) for item in batch])

    return out


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
