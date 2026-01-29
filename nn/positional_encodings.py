import torch
from typing import Optional


def _build_inv_freq(head_dim: int, theta: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Compute inverse frequencies for RoPE."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)
    )
    return inv_freq


def rotary_positional_frequencies(
    positions: torch.Tensor,
    head_dim: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """
    Build complex RoPE frequencies for the provided positions.

    Args:
        positions: Long tensor containing token indices [seq_len].
        head_dim: Per-head dimensionality (must be even).
        theta: RoPE base.
    Returns:
        Tensor of shape [seq_len, head_dim] stored as complex pairs (via torch.polar).
    """
    if head_dim % 2 != 0:
        raise ValueError("RoPE head dimension must be even.")

    if positions.dim() != 1:
        raise ValueError("positions must be a 1D tensor.")

    device = positions.device
    dtype = torch.float32
    inv_freq = _build_inv_freq(head_dim, theta, device, dtype)
    # positions is int -> convert to float for multiplication
    pos = positions.to(dtype=dtype)
    freqs = torch.einsum("i,j->ij", pos, inv_freq).to(dtype)
    ones = torch.ones_like(freqs, dtype=dtype)
    freqs_cis = torch.polar(ones, freqs)
    return freqs_cis


def apply_rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    freq_q: torch.Tensor,
    freq_k: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE rotations to query/key tensors.

    Args:
        q: Query tensor [B, H, S_q, D]
        k: Key tensor [B, H, S_k, D]
        freq_q: frequencies for queries [S_q, D]
        freq_k: frequencies for keys [S_k, D]
    Returns:
        Tuple of rotated (q, k) with same shapes.
    """
    def reshape_freq(freqs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Broadcasting helper so freqs aligns with (B,H,S,D)
        shape = [1, 1, freqs.size(0), freqs.size(1)]
        return freqs.view(*shape)

    if q.size(-1) % 2 != 0 or k.size(-1) % 2 != 0:
        raise ValueError("Head dimension must be even for RoPE transform.")

    expected_q_dim = q.size(-1) // 2
    expected_k_dim = k.size(-1) // 2
    if freq_q.size(-1) != expected_q_dim or freq_k.size(-1) != expected_k_dim:
        raise ValueError("Head dimension mismatch between q/k and RoPE frequencies.")

    # Convert to complex representation for rotation
    q_complex = torch.view_as_complex(
        q.float().reshape(*q.shape[:-1], -1, 2)
    )
    k_complex = torch.view_as_complex(
        k.float().reshape(*k.shape[:-1], -1, 2)
    )
    freq_q = reshape_freq(freq_q, q_complex)
    freq_k = reshape_freq(freq_k, k_complex)

    q_rot = torch.view_as_real(q_complex * freq_q).flatten(-2)
    k_rot = torch.view_as_real(k_complex * freq_k).flatten(-2)

    return q_rot.type_as(q), k_rot.type_as(k)


def build_position_indices(length: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Utility to create a 1D positions tensor [length]."""
    return torch.arange(length, device=device, dtype=torch.long)
