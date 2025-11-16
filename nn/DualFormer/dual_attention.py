"""
DualFormer: A transformer-based architecture for cross-modal attention between two modalities.

This implementation provides a symmetric attention mechanism where each modality can attend to the 
other, with the outputs combined to form a joint representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, Dict, List


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos_q, sin_q, cos_k, sin_k):
    # print("in apply_rotary_pos_emb: ", q.shape, k.shape, cos.shape, sin.shape)
    cos_q, sin_q = cos_q[...,:q.shape[2],:], sin_q[...,:q.shape[2],:]
    cos_k, sin_k = cos_k[...,:k.shape[2],:], sin_k[...,:k.shape[2],:]
    return (q * cos_q) + (rotate_half(q) * sin_q), (k * cos_k) + (rotate_half(k) * sin_k)

class DualRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        # Create inverse frequencies for the rotary embeddings
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for efficiency
        self.seq_len1_cached = None
        self.seq_len2_cached = None
        self.cos1_cached = None
        self.sin1_cached = None
        self.cos2_cached = None
        self.sin2_cached = None

    def forward(self, x1, x2):
        """
        Calculate rotary embeddings for cross-modal attention between two modalities.
        
        Args:
            x1: First modality tensor [batch_size, seq_len_1, embed_dim]
            x2: Second modality tensor [batch_size, seq_len_2, embed_dim]
            
        Returns:
            Tuple of (cos1, sin1, cos2, sin2) for applying to attention heads
        """
        # Get sequence lengths and device
        seq_len1 = x1.shape[1]
        seq_len2 = x2.shape[1]
        device = x1.device
        
        # Generate embeddings for first modality (for q1) if needed
        if seq_len1 != self.seq_len1_cached:
            self.seq_len1_cached = seq_len1
            t1 = torch.arange(seq_len1, device=device)
            freqs1 = torch.einsum('i,j->ij', t1, self.inv_freq)
            # Create embeddings - expand to cover full rotary dimensions
            emb1 = torch.cat((freqs1, freqs1), dim=-1)
            self.cos1_cached = emb1.cos().unsqueeze(0)  # [1, seq_len1, rotary_dim]
            self.sin1_cached = emb1.sin().unsqueeze(0)  # [1, seq_len1, rotary_dim]
        
        # Generate embeddings for second modality (for k2) if needed
        if seq_len2 != self.seq_len2_cached:
            self.seq_len2_cached = seq_len2
            t2 = torch.arange(seq_len2, device=device)
            freqs2 = torch.einsum('i,j->ij', t2, self.inv_freq)
            # Create embeddings - expand to cover full rotary dimensions
            emb2 = torch.cat((freqs2, freqs2), dim=-1)
            self.cos2_cached = emb2.cos().unsqueeze(0)  # [1, seq_len2, rotary_dim]
            self.sin2_cached = emb2.sin().unsqueeze(0)  # [1, seq_len2, rotary_dim]
            
        # Return embeddings for both modalities
        return self.cos1_cached, self.sin1_cached, self.cos2_cached, self.sin2_cached

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        return torch.stack([self.cos_cached, self.sin_cached])

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    Adds positional information to the input embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class DualAttention(nn.Module):
    """
    Dual Attention module for cross-modal interaction.
    Takes embeddings from two modalities and computes attention between them.
    
    Each modality acts as both query and key/value provider for the other modality.
    """
    def __init__(
        self, 
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bidirectional: bool = True,
        attention_type: str = 'cross'
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rotary_ndims = int(self.head_dim * 0.5)
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.bidirectional = bidirectional
        self.scaling = self.head_dim ** -0.5
        self.attention_type = attention_type
        
        # Projections for modality 1 -> modality 2 attention
        self.q1_proj = nn.Linear(embed_dim, embed_dim)
        self.k2_proj = nn.Linear(embed_dim, embed_dim)
        self.v2_proj = nn.Linear(embed_dim, embed_dim)
        
        # Projections for modality 2 -> modality 1 attention (if bidirectional)
        if bidirectional:
            self.q2_proj = nn.Linear(embed_dim, embed_dim)
            self.k1_proj = nn.Linear(embed_dim, embed_dim)
            self.v1_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projections
        self.out1_proj = nn.Linear(embed_dim, embed_dim)
        self.out2_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.attention_dropout = nn.Dropout(dropout)
        
    def _reshape_for_multihead(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape to [batch_size, num_heads, seq_len, head_dim]"""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]
        
    def _attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute scaled dot-product attention"""
        # q, k, v shapes: [batch, heads, seq_len, head_dim]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float("-inf"))
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)  # [batch, heads, seq_len, head_dim]
        output = output.transpose(1, 2).contiguous().view(
            output.shape[0], output.shape[2], self.embed_dim
        )  # [batch, seq_len, embed_dim]
        
        return output
        
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x1_padding_mask: Optional[torch.Tensor] = None,
        x2_padding_mask: Optional[torch.Tensor] = None,
        rope: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-modal attention between two modalities.
        
        Args:
            x1: First modality embeddings [batch_size, seq_len_1, embed_dim]
            x2: Second modality embeddings [batch_size, seq_len_2, embed_dim]
            x1_padding_mask: Padding mask for x1 [batch_size, seq_len_1]
            x2_padding_mask: Padding mask for x2 [batch_size, seq_len_2]
            
        Returns:
            Tuple containing updated embeddings for both modalities
            [batch_size, seq_len_1, embed_dim], [batch_size, seq_len_2, embed_dim]
        """
        batch_size, seq_len_1, _ = x1.shape
        _, seq_len_2, _ = x2.shape
        
        # Convert padding masks to attention masks if provided
        x1_attn_mask = None
        x2_attn_mask = None
        if x1_padding_mask is not None:
            # Convert from [batch, seq_len] to [batch, 1, 1, seq_len]
            x1_attn_mask = x1_padding_mask.unsqueeze(1).unsqueeze(2)
        if x2_padding_mask is not None:
            x2_attn_mask = x2_padding_mask.unsqueeze(1).unsqueeze(2)
        
        # Modality 1 attends to modality 2
        q1 = self._reshape_for_multihead(self.q1_proj(x1))  # [batch, heads, seq_len_1, head_dim]
        k2 = self._reshape_for_multihead(self.k2_proj(x2))  # [batch, heads, seq_len_2, head_dim]
        v2 = self._reshape_for_multihead(self.v2_proj(x2))  # [batch, heads, seq_len_2, head_dim]

        q2 = self._reshape_for_multihead(self.q2_proj(x2))  # [batch, heads, seq_len_2, head_dim]
        k1 = self._reshape_for_multihead(self.k1_proj(x1))  # [batch, heads, seq_len_1, head_dim]
        v1 = self._reshape_for_multihead(self.v1_proj(x1))  # [batch, heads, seq_len_1, head_dim]

        if rope is not None:
            cos1, sin1, cos2, sin2 = rope
            q1, query_pass_1 = q1[..., :self.rotary_ndims], q1[..., self.rotary_ndims:]
            k2, key_pass_2 = k2[..., :self.rotary_ndims], k2[..., self.rotary_ndims:]
            q1, k2 = apply_rotary_pos_emb(q1, k2, cos1, sin1, cos2, sin2) 
            q1 = torch.cat((q1, query_pass_1), dim=-1)
            k2 = torch.cat((k2, key_pass_2), dim=-1)

            q2, query_pass_2 = q2[..., :self.rotary_ndims], q2[..., self.rotary_ndims:]
            k1, key_pass_1 = k1[..., :self.rotary_ndims], k1[..., self.rotary_ndims:]
            q2, k1 = apply_rotary_pos_emb(q2, k1, cos2, sin2, cos1, sin1)
            q2 = torch.cat((q2, query_pass_2), dim=-1)
            k1 = torch.cat((k1, key_pass_1), dim=-1) 

        if self.attention_type == 'cross':
            attended_1 = self._attention(q1, k2, v2, x2_attn_mask)  # [batch, seq_len_1, embed_dim]
        elif self.attention_type == 'self':
            attended_1 = self._attention(q1, k1, v1, x1_attn_mask)
        elif self.attention_type == 'cross_self':
            attended_1 = self._attention(q1, k2, v2, x2_attn_mask) \
                         + self._attention(q1, k1, v1, x1_attn_mask)
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        attended_1 = self.out1_proj(attended_1)
        
        # Modality 2 attends to modality 1 (if bidirectional)
        if self.bidirectional:
            if self.attention_type == 'cross':
                attended_2 = self._attention(q2, k1, v1, x1_attn_mask) # [batch, seq_len_2, embed_dim]
            elif self.attention_type == 'self':
                attended_2 = self._attention(q2, k2, v2, x2_attn_mask)
            elif self.attention_type == 'cross_self':
                attended_2 = self._attention(q2, k1, v1, x1_attn_mask) \
                             + self._attention(q2, k2, v2, x2_attn_mask)
            else:
                raise ValueError(f"Unsupported attention type: {self.attention_type}")
            attended_2 = self.out2_proj(attended_2)
        else:
            attended_2 = x2  # No update if not bidirectional
        
        return attended_1, attended_2


class DualFeedForward(nn.Module):
    """
    Dual feed-forward network for processing both modalities.
    Each modality is processed by a separate FFN with the same architecture.
    """
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        # Select activation function
        self.activation_fn = {
            "relu": F.relu,
            "gelu": F.gelu,
            "silu": F.silu
        }.get(activation, F.gelu)
        
        # FFN for modality 1
        self.ffn1 = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(ffn_dim),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # FFN for modality 2
        self.ffn2 = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(ffn_dim),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process both modalities through their respective FFNs.
        
        Args:
            x1: First modality embeddings [batch_size, seq_len_1, embed_dim]
            x2: Second modality embeddings [batch_size, seq_len_2, embed_dim]
            
        Returns:
            Tuple of processed embeddings:
            [batch_size, seq_len_1, embed_dim], [batch_size, seq_len_2, embed_dim]
        """
        # Apply activation function at the appropriate point in the network
        x1_mid = self.ffn1[2](self.activation_fn(self.ffn1[0](x1)))
        x1_out = self.ffn1[4](self.ffn1[3](x1_mid))
        
        x2_mid = self.ffn2[2](self.activation_fn(self.ffn2[0](x2)))
        x2_out = self.ffn2[4](self.ffn2[3](x2_mid))
        
        return x1_out, x2_out


class DualFormerBlock(nn.Module):
    """
    A single block in the DualFormer, consisting of:
    1. Layer normalization
    2. Dual attention
    3. Residual connection
    4. Layer normalization
    5. FFN
    6. Residual connection
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        bidirectional: bool = True,
        norm_first: bool = True,  # Pre-norm (True) or post-norm (False)
        attention_type: str = 'cross'
    ):
        super().__init__()
        
        self.norm_first = norm_first
        
        # Layer normalization
        self.norm1_1 = nn.LayerNorm(embed_dim)
        self.norm1_2 = nn.LayerNorm(embed_dim)
        self.norm2_1 = nn.LayerNorm(embed_dim)
        self.norm2_2 = nn.LayerNorm(embed_dim)
        
        # Dual attention
        self.attention = DualAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            bidirectional=bidirectional,
            attention_type=attention_type
        )
        
        # Dual feedforward
        self.feed_forward = DualFeedForward(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
            activation=activation
        )
        
        # Dropouts
        self.dropout = nn.Dropout(dropout)
        
    def _attention_block(
        self, 
        x1: torch.Tensor, 
        x2: torch.Tensor,
        x1_padding_mask: Optional[torch.Tensor] = None,
        x2_padding_mask: Optional[torch.Tensor] = None,
        rope: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention with residual connection"""
        if self.norm_first:
            attended_1, attended_2 = self.attention(
                self.norm1_1(x1), 
                self.norm1_2(x2),
                x1_padding_mask,
                x2_padding_mask,
                rope
            )
            x1 = x1 + self.dropout(attended_1)
            x2 = x2 + self.dropout(attended_2)
        else:
            attended_1, attended_2 = self.attention(x1, x2, x1_padding_mask, x2_padding_mask, rope)
            x1 = self.norm1_1(x1 + self.dropout(attended_1))
            x2 = self.norm1_2(x2 + self.dropout(attended_2))
        return x1, x2
    
    def _feedforward_block(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply feedforward with residual connection"""
        if self.norm_first:
            ff_out_1, ff_out_2 = self.feed_forward(self.norm2_1(x1), self.norm2_2(x2))
            x1 = x1 + ff_out_1
            x2 = x2 + ff_out_2
        else:
            ff_out_1, ff_out_2 = self.feed_forward(x1, x2)
            x1 = self.norm2_1(x1 + ff_out_1)
            x2 = self.norm2_2(x2 + ff_out_2)
        return x1, x2
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x1_padding_mask: Optional[torch.Tensor] = None,
        x2_padding_mask: Optional[torch.Tensor] = None,
        rope: Optional[torch.Tensor] = None,
        
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process both modalities through attention and FFN.
        
        Args:
            x1: First modality embeddings [batch_size, seq_len_1, embed_dim]
            x2: Second modality embeddings [batch_size, seq_len_2, embed_dim]
            x1_padding_mask: Padding mask for x1 [batch_size, seq_len_1]
            x2_padding_mask: Padding mask for x2 [batch_size, seq_len_2]
            
        Returns:
            Tuple of processed embeddings:
            [batch_size, seq_len_1, embed_dim], [batch_size, seq_len_2, embed_dim]
        """
        x1, x2 = self._attention_block(x1, x2, x1_padding_mask, x2_padding_mask, rope)
        x1, x2 = self._feedforward_block(x1, x2)
        
        return x1, x2


class DualFormer(nn.Module):
    """
    DualFormer: A transformer model that processes two modalities using cross-attention.
    
    This model takes embeddings from two different modalities and processes them using:
    1. Optional positional encoding for each modality
    2. Multiple DualFormer blocks with cross-attention
    3. Optional pooling to get sequence-level representations
    
    The output can be a joint representation or separate representations for each modality.
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        output_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        bidirectional: bool = True,
        norm_first: bool = True,
        use_positional_encoding: bool = True,
        max_seq_len: int = 5000,
        pooling: str = "mean",  # Options: mean, max, cls, none
        use_cls_token: bool = False,
        use_prediction_head: bool = False,
        latent_dim: int = 0,
        attention_type: str = 'cross',
        projection_type: str = 'transpose'
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_positional_encoding = use_positional_encoding
        self.pooling = pooling
        self.use_cls_token = use_cls_token
        self.use_prediction_head = use_prediction_head
        self.projection_type = projection_type

        self.embedding = nn.Linear(input_dim, embed_dim)

        self.prediction_head1 = nn.Sequential(
                                nn.Linear(embed_dim, embed_dim//2),
                                nn.LayerNorm(embed_dim//2),
                                nn.Dropout(0.1),
                                nn.GELU(),
                                nn.Linear(embed_dim//2, output_dim)
        )
        self.prediction_head2 = nn.Sequential(
                                nn.Linear(embed_dim, embed_dim//2),
                                nn.LayerNorm(embed_dim//2),
                                nn.Dropout(0.1),
                                nn.GELU(),
                                nn.Linear(embed_dim//2, output_dim)
        )

        self.projection_head = nn.Linear(embed_dim + latent_dim, embed_dim + latent_dim, bias=False)

        
        # Create CLS tokens if needed
        if use_cls_token:
            self.cls_token1 = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.cls_token2 = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional encoding
        if use_positional_encoding:
            self.rotary_ndims = int((embed_dim // num_heads) * 0.5)
            self.pe = DualRotaryEmbedding(self.rotary_ndims)
            # self.pos_encoder1 = PositionalEncoding(embed_dim, max_seq_len, dropout)
            # self.pos_encoder2 = PositionalEncoding(embed_dim, max_seq_len, dropout)
        
        # DualFormer blocks
        self.layers = nn.ModuleList([
            DualFormerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation=activation,
                bidirectional=bidirectional,
                norm_first=norm_first,
                attention_type=attention_type
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def _add_cls_token(self, x: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
        """Add CLS token to the beginning of the sequence"""
        batch_size = x.shape[0]
        cls_tokens = cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, embed_dim]
        return torch.cat((cls_tokens, x), dim=1)
    
    def _pool(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply pooling to get sequence-level representation"""
        if self.pooling == "mean":
            if padding_mask is not None:
                # Apply mask for proper averaging
                expanded_mask = padding_mask.unsqueeze(-1).expand_as(x)
                x = (x * expanded_mask).sum(dim=1) / expanded_mask.sum(dim=1)
            else:
                x = x.mean(dim=1)
        elif self.pooling == "max":
            if padding_mask is not None:
                # Apply mask before max pooling
                expanded_mask = padding_mask.unsqueeze(-1).expand_as(x)
                x = (x * expanded_mask).max(dim=1)[0]
            else:
                x = x.max(dim=1)[0]
        elif self.pooling == "cls" and self.use_cls_token:
            # Use the [CLS] token representation
            x = x[:, 0]
        # If pooling is "none", return the full sequence
        return x
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x1_padding_mask: Optional[torch.Tensor] = None,
        x2_padding_mask: Optional[torch.Tensor] = None,
        latent_variables: Optional[torch.Tensor] = None,
        output_attentions: bool = True,
        output_hidden_states: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, Union[torch.Tensor, List[torch.Tensor]]]
    ]:
        """
        Process embeddings from two modalities through the DualFormer.
        
        Args:
            x1: First modality embeddings [batch_size, seq_len_1, embed_dim]
            x2: Second modality embeddings [batch_size, seq_len_2, embed_dim]
            x1_padding_mask: Padding mask for x1 [batch_size, seq_len_1] (1 for valid, 0 for pad)
            x2_padding_mask: Padding mask for x2 [batch_size, seq_len_2] (1 for valid, 0 for pad)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states from all layers
            
        Returns:
            If output_attentions or output_hidden_states is True:
                Dictionary with keys:
                - mod1_output: Final output for modality 1
                - mod2_output: Final output for modality 2
                - attentions: List of attention weights (if output_attentions=True)
                - hidden_states: List of hidden states (if output_hidden_states=True)
            Otherwise:
                Tuple of (mod1_output, mod2_output)
        """
        if len(x1.shape) == 2:
            x1 = x1.unsqueeze(-1)
        if len(x2.shape) == 2:
            x2 = x2.unsqueeze(-1)
        
        x1 = self.embedding(x1)  # [batch_size, seq_len_1, embed_dim]
        x2 = self.embedding(x2)  # [batch_size, seq_len_2, embed_dim]
        
        attentions = [] if output_attentions else None
        hidden_states = [] if output_hidden_states else None
        
        # Add CLS tokens if needed
        if self.use_cls_token:
            x1 = self._add_cls_token(x1, self.cls_token1)
            x2 = self._add_cls_token(x2, self.cls_token2)
            
            # Update padding masks to account for CLS tokens
            if x1_padding_mask is not None:
                x1_padding_mask = torch.cat([
                    torch.ones(x1_padding_mask.shape[0], 1, device=x1_padding_mask.device),
                    x1_padding_mask
                ], dim=1)
            if x2_padding_mask is not None:
                x2_padding_mask = torch.cat([
                    torch.ones(x2_padding_mask.shape[0], 1, device=x2_padding_mask.device),
                    x2_padding_mask
                ], dim=1)
        
        # Apply positional encoding
        if self.use_positional_encoding:
            RoPE = self.pe(x1, x2) # RoPE: [2, B, L, encoder_dim], 2: sin, cos
        else:
            RoPE = None
            
        # Record initial hidden states
        if output_hidden_states:
            hidden_states.append((x1.clone(), x2.clone()))
            
        # Process through DualFormer blocks
        for layer in self.layers:
            x1, x2 = layer(x1, x2, x1_padding_mask, x2_padding_mask, RoPE)
            
            if output_hidden_states:
                hidden_states.append((x1.clone(), x2.clone()))
                
            # TODO: If implementing output_attentions, collect attention weights from each layer
            
        # Final layer normalization
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        
        # Pooling if requested
        if self.pooling != "none":
            pooled_x1 = self._pool(x1, x1_padding_mask)
            pooled_x2 = self._pool(x2, x2_padding_mask)
            mod1_output = pooled_x1
            mod2_output = pooled_x2
        else:
            mod1_output = x1
            mod2_output = x2
            
        if latent_variables is not None:
            mod1_output = torch.cat([mod1_output, latent_variables], dim=-1)
            mod2_output = torch.cat([mod2_output, latent_variables], dim=-1)
        
        output_dict = {
            "emb1": mod1_output,
            "emb2": mod2_output,
        }


        proj1 = self.projection_head(mod1_output)
        if self.projection_type == 'transpose':
            proj2 = F.linear(mod2_output, self.projection_head.weight.t(), bias=None)
        elif self.projection_type == 'normal':
            proj2 = self.projection_head(mod2_output)
        else:
            raise ValueError(f"Unsupported projection type: {self.projection_type}")
       

        output_dict["proj1"] = proj1
        output_dict["proj2"] = proj2

        # symmetry_loss = torch.norm(self.projection_head.weight - self.projection_head.weight.t())
        # output_dict["symmetry_loss"] = symmetry_loss

        if self.use_prediction_head:
            pred1 = self.prediction_head1(mod1_output)
            pred2 = self.prediction_head2(mod2_output)
            output_dict["pred1"] = pred1
            output_dict["pred2"] = pred2
            
        if output_hidden_states:
            output_dict["hidden_states"] = hidden_states
            
        if output_attentions:
            output_dict["attentions"] = attentions
                
        return output_dict
        


class DualFormerForJointEmbedding(nn.Module):
    """
    A DualFormer model for creating joint embeddings from two modalities.
    This can be used for tasks like cross-modal retrieval or contrastive learning.
    """
    def __init__(
        self,
        embed_dim: int,
        projection_dim: int = 256,
        **kwargs
    ):
        super().__init__()
        
        # Create the base DualFormer
        self.dual_former = DualFormer(embed_dim=embed_dim, **kwargs)
        
        # Create projection heads to map to a common embedding space
        self.proj1 = nn.Linear(embed_dim, projection_dim)
        self.proj2 = nn.Linear(embed_dim, projection_dim)
        
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x1_padding_mask: Optional[torch.Tensor] = None,
        x2_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create joint embeddings for cross-modal tasks.
        
        Args:
            x1: First modality embeddings [batch_size, seq_len_1, embed_dim]
            x2: Second modality embeddings [batch_size, seq_len_2, embed_dim]
            x1_padding_mask: Padding mask for x1 [batch_size, seq_len_1]
            x2_padding_mask: Padding mask for x2 [batch_size, seq_len_2]
            
        Returns:
            Tuple of projected embeddings:
            [batch_size, projection_dim], [batch_size, projection_dim]
        """
        # Process through DualFormer
        mod1_output, mod2_output = self.dual_former(x1, x2, x1_padding_mask, x2_padding_mask)
        
        # Project to common space
        proj1 = F.normalize(self.proj1(mod1_output), p=2, dim=-1)
        proj2 = F.normalize(self.proj2(mod2_output), p=2, dim=-1)
        
        return proj1, proj2


class DualFormerForRegression(nn.Module):
    """
    A DualFormer model for regression tasks using information from two modalities.
    This can be used for prediction tasks where information from both modalities is relevant.
    """
    def __init__(
        self,
        embed_dim: int,
        num_outputs: int = 1,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        fusion: str = "concat",  # Options: concat, sum, product
        **kwargs
    ):
        super().__init__()
        
        self.fusion = fusion
        
        # Create the base DualFormer
        self.dual_former = DualFormer(embed_dim=embed_dim, **kwargs)
        
        # Determine the input size for the regressor based on fusion method
        if fusion == "concat":
            regressor_input_dim = embed_dim * 2
        else:  # sum or product
            regressor_input_dim = embed_dim
        
        # Create the regressor
        self.regressor = nn.Sequential(
            nn.Linear(regressor_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs)
        )
        
    def _fuse_embeddings(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """Fuse embeddings from two modalities"""
        if self.fusion == "concat":
            return torch.cat([emb1, emb2], dim=-1)
        elif self.fusion == "sum":
            return emb1 + emb2
        elif self.fusion == "product":
            return emb1 * emb2
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion}")
        
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x1_padding_mask: Optional[torch.Tensor] = None,
        x2_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict regression targets using both modalities.
        
        Args:
            x1: First modality embeddings [batch_size, seq_len_1, embed_dim]
            x2: Second modality embeddings [batch_size, seq_len_2, embed_dim]
            x1_padding_mask: Padding mask for x1 [batch_size, seq_len_1]
            x2_padding_mask: Padding mask for x2 [batch_size, seq_len_2]
            
        Returns:
            Regression outputs [batch_size, num_outputs]
        """
        # Process through DualFormer
        mod1_output, mod2_output = self.dual_former(x1, x2, x1_padding_mask, x2_padding_mask)
        
        # Fuse embeddings
        fused_embedding = self._fuse_embeddings(mod1_output, mod2_output)
        
        # Predict outputs
        return self.regressor(fused_embedding)