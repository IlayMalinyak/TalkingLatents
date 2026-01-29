# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


import torch
import torch.nn.functional as F

try:
    import fairscale.nn.model_parallel.initialize as fs_init
    from fairscale.nn.model_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
        VocabParallelEmbedding,
    )
except ImportError:
    class MockFsInit:
        @staticmethod
        def get_model_parallel_world_size():
            return 1
        @staticmethod
        def get_model_parallel_rank():
            return 0
    fs_init = MockFsInit()

    class ColumnParallelLinear(torch.nn.Linear):
        def __init__(self, in_features, out_features, *args, **kwargs):
            kwargs.pop('gather_output', None)
            kwargs.pop('init_method', None)
            super().__init__(in_features, out_features, *args, **kwargs)

    class RowParallelLinear(torch.nn.Linear):
        def __init__(self, in_features, out_features, *args, **kwargs):
            kwargs.pop('input_is_parallel', None)
            kwargs.pop('init_method', None)
            super().__init__(in_features, out_features, *args, **kwargs)

    class VocabParallelEmbedding(torch.nn.Embedding):
         def __init__(self, num_embeddings, embedding_dim, *args, **kwargs):
            kwargs.pop('init_method', None)
            super().__init__(num_embeddings, embedding_dim, *args, **kwargs)

from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = True

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.args = args # Store args for lazy allocation

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        # Lazy allocation to save memory when cache is not used (e.g. training)
        self.cache_k = None
        self.cache_v = None

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        cache_rows: Optional[Sequence[int]] = None,
        use_cache: bool = True,
    ):
        bsz, seqlen, _ = x.shape
        if cache_rows is not None:
            if len(cache_rows) != bsz:
                raise ValueError(f"cache_rows length {len(cache_rows)} != batch size {bsz}")
            row_indices = [int(r) for r in cache_rows]
        else:
            row_indices = list(range(bsz))
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # DEBUG: Check for NaNs in Q/K/V
        if torch.isnan(xq).any() or torch.isinf(xq).any():
             print(f"DEBUG: Attention NaN/Inf detected in XQ! Shape: {xq.shape}, Max: {xq.max()}, Min: {xq.min()}")
        if torch.isnan(xk).any() or torch.isinf(xk).any():
             print(f"DEBUG: Attention NaN/Inf detected in XK! Shape: {xk.shape}")
        if torch.isnan(xv).any() or torch.isinf(xv).any():
             print(f"DEBUG: Attention NaN/Inf detected in XV! Shape: {xv.shape}")

        if use_cache:
            if self.cache_k is None:
                # Lazy allocation on first use
                self.cache_k = torch.zeros(
                    (
                        self.args.max_batch_size,
                        self.args.max_seq_len,
                        self.n_local_kv_heads,
                        self.head_dim,
                    ),
                    device=xq.device,
                    dtype=xq.dtype # Match dtype
                )
                self.cache_v = torch.zeros(
                    (
                        self.args.max_batch_size,
                        self.args.max_seq_len,
                        self.n_local_kv_heads,
                        self.head_dim,
                    ),
                    device=xq.device,
                    dtype=xq.dtype
                )
            
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)
    
            for i, row in enumerate(row_indices):
                self.cache_k[row, start_pos : start_pos + seqlen] = xk[i]
                self.cache_v[row, start_pos : start_pos + seqlen] = xv[i]
    
            keys = torch.stack(
                [self.cache_k[row, : start_pos + seqlen] for row in row_indices], dim=0
            )
            values = torch.stack(
                [self.cache_v[row, : start_pos + seqlen] for row in row_indices], dim=0
            )
        else:
            # Bypass cache storage, use current inputs directly
            # This avoids OOM/Graph leaks during training
            keys = xk
            values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        
        # Clamp scores to prevent overflow/NaN in Softmax (and backprop)
        scores = torch.clamp(scores, min=-10000.0, max=10000.0)
        
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # --- HARD DEBUG PATCH (CONDITIONAL) ---
        # print(f"DEBUG: Attention forward {id(self)} run. store_attention={getattr(self, 'store_attention', 'MISSING')}")
        if getattr(self, 'store_attention', False):
            try:
                self.last_attn_scores = scores.detach().cpu()
            except Exception as e:
                print(f"Error capturing attention: {e}")
        # ------------------------
        
        # Force float32 for weighted sum to avoid FP16 overflow/underflow
        # Also disable autocast to ensure gradients don't downcast
        with torch.autocast(device_type='cuda', enabled=False):
             scores_f = scores.float()
             values_f = values.float()
             output = torch.matmul(scores_f, values_f)  # (bs, n_local_heads, seqlen, head_dim)
        
        output = output.type_as(xq)
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        cache_rows: Optional[Sequence[int]] = None,
        use_cache: bool = True,
    ):
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask, cache_rows=cache_rows, use_cache=use_cache
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def forward(
        self, tokens: torch.Tensor, start_pos: int, cache_rows: Optional[Sequence[int]] = None
    ):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)
        
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask, cache_rows=cache_rows)
        h = self.norm(h)
        output = self.output(h).float()
        return output, h
