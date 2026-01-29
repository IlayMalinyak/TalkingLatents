from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Union


@dataclass
class LLMBackendConfig:
    """Metadata describing which LLM/tokenizer backend is active."""

    backend: str
    tokenizer_backend: str
    model_name_or_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    quantization: str = "none"
    precision: str = "fp16"
    device_map: Optional[Union[str, Dict[str, Any]]] = None
    cache_dir: Optional[str] = None
    trust_remote_code: bool = False
    revision: Optional[str] = None
    max_memory_gb: Optional[float] = None
    auth_token: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation for logging/checkpointing."""
        return asdict(self)
