"""Tokenizer adapters to provide a unified interface for LLaMA SentencePiece and Hugging Face tokenizers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TokenizerAdapter:
    """Lightweight adapter that mimics the subset of methods used in the data pipeline."""

    pad_id: Optional[int] = None
    eos_id: Optional[int] = None
    bos_id: Optional[int] = None

    def encode(self, text: str, bos: bool = True, eos: bool = False) -> List[int]:
        raise NotImplementedError

    def decode(self, tokens: List[int]) -> str:
        raise NotImplementedError


class LlamaTokenizerAdapter(TokenizerAdapter):
    """Adapter around the Meta LLaMA tokenizer implementation."""

    def __init__(self, tokenizer_path: str):
        if not tokenizer_path or not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"LLaMA tokenizer model not found at {tokenizer_path}")
        from llama3.llama.tokenizer import Tokenizer  # Lazy import to avoid heavy dependency when unused

        self._tokenizer = Tokenizer(model_path=tokenizer_path)
        self.pad_id = getattr(self._tokenizer, "pad_id", 0)
        self.eos_id = getattr(self._tokenizer, "eos_id", None)
        self.bos_id = getattr(self._tokenizer, "bos_id", None)

    def encode(self, text: str, bos: bool = True, eos: bool = False) -> List[int]:
        return self._tokenizer.encode(text, bos=bos, eos=eos)

    def decode(self, tokens: List[int]) -> str:
        return self._tokenizer.decode(tokens)


class HFAutoTokenizerAdapter(TokenizerAdapter):
    """Adapter around a Hugging Face AutoTokenizer."""

    def __init__(
        self,
        model_name_or_path: str,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        padding_side: str = "right",
    ):
        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Hugging Face tokenizer requested but transformers is not installed. "
                "Install it via `pip install transformers`."
            ) from exc

        tokenizer_kwargs = {
            "trust_remote_code": trust_remote_code,
        }
        if revision:
            tokenizer_kwargs["revision"] = revision

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
        )
        if self._tokenizer.pad_token is None:
            if self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            elif self._tokenizer.unk_token is not None:
                self._tokenizer.pad_token = self._tokenizer.unk_token
            else:
                self._tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        self._tokenizer.padding_side = padding_side
        self._tokenizer.truncation_side = "right"

        self.pad_id = self._tokenizer.pad_token_id
        self.eos_id = self._tokenizer.eos_token_id or self.pad_id
        self.bos_id = self._tokenizer.bos_token_id or self._tokenizer.cls_token_id or self.pad_id

    def encode(self, text: str, bos: bool = True, eos: bool = False) -> List[int]:
        tokens = self._tokenizer.encode(text, add_special_tokens=False)
        if bos and self.bos_id is not None:
            tokens = [self.bos_id] + tokens
        if eos and self.eos_id is not None:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens: List[int]) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=True)

    @property
    def tokenizer(self):
        """Expose the underlying AutoTokenizer for advanced use-cases."""
        return self._tokenizer


def load_tokenizer_adapter(
    backend: str,
    tokenizer_path: Optional[str] = None,
    hf_model_name: Optional[str] = None,
    trust_remote_code: bool = False,
    hf_revision: Optional[str] = None,
    padding_side: str = "right",
) -> TokenizerAdapter:
    """Factory helper to build a tokenizer adapter for the requested backend."""
    backend = (backend or "llama").lower()
    if backend == "llama":
        if tokenizer_path is None:
            raise ValueError("tokenizer_path must be provided for the LLaMA backend.")
        return LlamaTokenizerAdapter(tokenizer_path)
    if backend in {"hf", "qwen"}:
        if hf_model_name is None:
            raise ValueError("hf_model_name must be provided when llm_backend is 'hf'/'qwen'.")
        return HFAutoTokenizerAdapter(
            model_name_or_path=hf_model_name,
            trust_remote_code=trust_remote_code,
            revision=hf_revision,
            padding_side=padding_side,
        )
    raise ValueError(f"Unsupported llm_backend '{backend}'. Expected 'llama' or 'hf/qwen'.")
