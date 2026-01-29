import math
from typing import Optional

import torch
import torch.nn as nn

from nn.positional_encodings import (
    rotary_positional_frequencies,
    apply_rotary_embedding,
    build_position_indices,
)


class FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = dim * ffn_mult
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class RotaryAttention(nn.Module):
    """Multi-head attention with rotary positional encoding support."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.1,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope_theta = rope_theta

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return (
            x.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_positions: torch.Tensor,
        kv_positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, S_q, dim]
            key/value: [B, S_k, dim]
            q_positions: [S_q]
            kv_positions: [S_k]
            key_padding_mask: optional [B, S_k] with 1 for tokens to keep.
        """
        q = self._shape(self.q_proj(query))
        k = self._shape(self.k_proj(key))
        v = self._shape(self.v_proj(value))

        freq_q = rotary_positional_frequencies(q_positions.to(query.device), self.head_dim, self.rope_theta)
        freq_k = rotary_positional_frequencies(kv_positions.to(query.device), self.head_dim, self.rope_theta)
        q, k = apply_rotary_embedding(q, k, freq_q, freq_k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.bool()
            mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)  # [B,1,1,S_k]
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        context = torch.matmul(attn_weights, v)  # [B, H, S_q, D]
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(query.size(0), query.size(1), self.dim)
        )
        output = self.out_proj(context)
        output = self.proj_dropout(output)
        return output


class CrossAttentionBlock(nn.Module):
    """Latent queries attending to serialized multimodal inputs."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.latent_norm = nn.LayerNorm(dim)
        self.input_norm = nn.LayerNorm(dim)
        self.attn = RotaryAttention(
            dim=dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            rope_theta=rope_theta,
        )
        self.ff = FeedForward(dim, ffn_mult=ffn_mult, dropout=dropout)

    def forward(
        self,
        latents: torch.Tensor,
        inputs: torch.Tensor,
        input_mask: Optional[torch.Tensor],
        latent_positions: torch.Tensor,
        input_positions: torch.Tensor,
    ) -> torch.Tensor:
        latents_norm = self.latent_norm(latents)
        inputs_norm = self.input_norm(inputs)
        attn_out = self.attn(
            latents_norm,
            inputs_norm,
            inputs_norm,
            q_positions=latent_positions,
            kv_positions=input_positions,
            key_padding_mask=input_mask,
        )
        latents = latents + attn_out
        latents = self.ff(latents)
        return latents


class LatentSelfAttentionBlock(nn.Module):
    """Transformer block operating purely on latent tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = RotaryAttention(
            dim=dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            rope_theta=rope_theta,
        )
        self.ff = FeedForward(dim, ffn_mult=ffn_mult, dropout=dropout)

    def forward(self, latents: torch.Tensor, latent_positions: torch.Tensor) -> torch.Tensor:
        latents_norm = self.norm(latents)
        attn_out = self.attn(
            latents_norm,
            latents_norm,
            latents_norm,
            q_positions=latent_positions,
            kv_positions=latent_positions,
        )
        latents = latents + attn_out
        latents = self.ff(latents)
        return latents


class PerceiverEncoder(nn.Module):
    """Perceiver IO style encoder with a latent bottleneck."""

    def __init__(
        self,
        d_model: int,
        num_latents: int,
        num_latent_blocks: int,
        num_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, d_model))
        self.latent_positions = nn.Parameter(
            build_position_indices(num_latents), requires_grad=False
        )
        self.cross_attn = CrossAttentionBlock(
            dim=d_model,
            num_heads=num_heads,
            ffn_mult=ffn_mult,
            dropout=dropout,
            attn_dropout=attn_dropout,
            rope_theta=rope_theta,
        )
        self.latent_blocks = nn.ModuleList(
            [
                LatentSelfAttentionBlock(
                    dim=d_model,
                    num_heads=num_heads,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    rope_theta=rope_theta,
                )
                for _ in range(num_latent_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        inputs: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        input_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            inputs: [B, N, d_model]
            input_mask: optional [B, N] bool/float mask (1=keep)
            input_positions: optional [N] tensor
        Returns:
            Latent array [B, num_latents, d_model]
        """
        if inputs.dim() != 3:
            raise ValueError("inputs must be of shape [B, N, d_model].")

        bsz, seq_len, _ = inputs.shape
        if input_positions is None:
            input_positions = build_position_indices(seq_len, device=inputs.device)
        if input_mask is None:
            input_mask = torch.ones(bsz, seq_len, device=inputs.device, dtype=torch.bool)

        latents = self.latents.unsqueeze(0).expand(bsz, -1, -1)
        latent_positions = self.latent_positions.to(inputs.device)
        latents = self.cross_attn(
            latents,
            inputs,
            input_mask,
            latent_positions=latent_positions,
            input_positions=input_positions,
        )
        for block in self.latent_blocks:
            latents = block(latents, latent_positions=latent_positions)
        latents = self.final_norm(latents)
        return latents
