import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class FiLMSpectralDecoder(nn.Module):
    """Condition a bank of spectral queries with FiLM-style modulation."""

    def __init__(self, d_model: int, spectral_length: int, hidden_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.spectral_length = spectral_length
        self.query_bank = nn.Parameter(torch.randn(1, spectral_length, d_model))
        hidden_dim = d_model * hidden_mult
        self.conditioner = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * d_model),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: [B, M, d_model]
        Returns:
            Spectral reconstruction [B, spectral_length]
        """
        bsz, _, dim = latents.shape
        pooled = latents.mean(dim=1)  # [B, d_model]
        gamma_beta = self.conditioner(pooled)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        queries = self.query_bank.expand(bsz, -1, -1)
        queries = queries * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        preds = self.output(queries).squeeze(-1)
        return preds


class QueryCrossAttentionDecoder(nn.Module):
    """Cross-attend learned spectral queries to latents."""

    def __init__(
        self,
        d_model: int,
        spectral_length: int,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, spectral_length, d_model))
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: [B, M, d_model]
        Returns:
            Spectral reconstruction [B, spectral_length]
        """
        bsz = latents.size(0)
        queries = self.queries.expand(bsz, -1, -1)
        attn_out, _ = self.attn(queries, latents, latents)
        preds = self.output(attn_out).squeeze(-1)
        return preds


class SpectralDecoder(nn.Module):
    """Wrapper that selects the decoder implementation based on config."""

    def __init__(
        self,
        mode: Literal["film_1d", "query_crossattn"],
        d_model: int,
        spectral_length: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if spectral_length <= 0:
            raise ValueError("spectral_length must be a positive integer.")

        if mode == "film_1d":
            self.decoder = FiLMSpectralDecoder(d_model=d_model, spectral_length=spectral_length, dropout=dropout)
        elif mode == "query_crossattn":
            self.decoder = QueryCrossAttentionDecoder(
                d_model=d_model,
                spectral_length=spectral_length,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported spectral decoder mode: {mode}")

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)
