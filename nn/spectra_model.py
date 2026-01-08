import torch
import torch.nn as nn
import torch.nn.functional as F
from nn.Modules.conformer import ConformerEncoder, ConformerDecoder
from nn.Modules.mhsa_pro import RotaryEmbedding, ContinuousRotaryEmbedding
from nn.Modules.flash_mhsa import MHA as Flash_Mha
from nn.Modules.mlp import Mlp as MLP
import numbers
import torch.nn.init as init
from typing import Union, List, Optional, Tuple
from torch import Size, Tensor


def get_activation(args):
    
    if args.activation == 'silu':
        return nn.SiLU()
    elif args.activation == 'sine':
        return Sine(w0=args.sine_w0)
    elif args.activation == 'relu':
        return nn.ReLU()
    elif args.activation == 'gelu':
        return nn.GELU()
    else:
        return nn.ReLU()

class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)

class ConvBlock(nn.Module):
  def __init__(self, args, num_layer) -> None:
    super().__init__()
    self.activation = get_activation(args)
    in_channels = args.encoder_dims[num_layer-1] if num_layer < len(args.encoder_dims) else args.encoder_dims[-1]
    out_channels = args.encoder_dims[num_layer] if num_layer < len(args.encoder_dims) else args.encoder_dims[-1]
    self.layers = nn.Sequential(
        nn.Conv1d(in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=args.kernel_size,
                stride=1, padding='same', bias=False),
        nn.BatchNorm1d(num_features=out_channels),
        self.activation,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:  
    return self.layers(x)

class CNNEncoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.activation = get_activation(args)
        self.embedding = nn.Sequential(nn.Conv1d(in_channels = args.in_channels,
                kernel_size=3, out_channels = args.encoder_dims[0], stride=1, padding = 'same', bias = False),
                        nn.BatchNorm1d(args.encoder_dims[0]),
                        self.activation,
        )
        self.in_channels = args.in_channels 
        self.layers = nn.ModuleList([ConvBlock(args, i+1)
        for i in range(args.num_layers)])
        self.pool = nn.MaxPool1d(2)
        self.output_dim = args.encoder_dims[-1]
        self.min_seq_len = 2 
        self.avg_output = args.avg_output
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape)==2:
            x = x.unsqueeze(1)
        if len(x.shape)==3 and x.shape[-1]==1:
            x = x.permute(0,2,1)
        x = self.embedding(x.float())
        for m in self.layers:
            x = m(x)
            if x.shape[-1] > self.min_seq_len:
                x = self.pool(x)
        if self.avg_output:
            x = x.mean(dim=-1)
        else:
            x = x.permute(0,2,1)
        return x

class CNNDecoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        
        # Reverse the encoder dimensions for upsampling
        decoder_dims = args.encoder_dims[::-1]
        
        if args.activation == 'silu':
            self.activation = nn.SiLU()
        elif args.activation == 'sine':
            self.activation = Sine(w0=args.sine_w0)
        else:
            self.activation = nn.ReLU()
        
        # Initial embedding layer to expand the compressed representation
        self.initial_expand = nn.Linear(decoder_dims[0], decoder_dims[0] * 4)
        
        # Transposed Convolutional layers for upsampling
        self.layers = nn.ModuleList()
        for i in range(args.num_layers):
            if i  < len(decoder_dims) - 1:
                in_channels = decoder_dims[i] 
                out_channels = decoder_dims[i+1]
            else:
                in_channels = decoder_dims[-1]
                out_channels = decoder_dims[-1]
            
            # Transposed Convolution layer
            layer = nn.Sequential(
                nn.ConvTranspose1d(in_channels=in_channels, 
                                   out_channels=out_channels, 
                                   kernel_size=4, 
                                   stride=2, 
                                   padding=1, 
                                   bias=False),
                nn.BatchNorm1d(out_channels),
                self.activation
            )
            self.layers.append(layer)
        
        # Final layer to match original input channels
        self.final_conv = nn.ConvTranspose1d(in_channels=decoder_dims[-1], 
                                             out_channels=1, 
                                             kernel_size=3, 
                                             stride=1, 
                                             padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand the compressed representation
        # x = self.initial_expand(x)
        # x = x.unsqueeze(-1)  # Add sequence dimension
        
        # Apply transposed convolution layers
        x = x.float()
        for layer in self.layers:
            x = layer(x)
        
        # Final convolution to get back to original input channels
        x = self.final_conv(x)
        
        return x.squeeze()

class MultiTaskRegressor(nn.Module):
    def __init__(self, args, conformer_args):
        
        super().__init__()
        self.encoder = MultiEncoder(args, conformer_args)
        self.decoder = CNNDecoder(args)
        # self.projector = projection_MLP(conformer_args.encoder_dim)
        
        if args.activation == 'silu':
            self.activation = nn.SiLU()
        elif args.activation == 'sine':
            self.activation = Sine(w0=args.sine_w0)
        else:
            self.activation = nn.ReLU()

        self.avg_output = args.avg_output
        encoder_dim = conformer_args.encoder_dim
        self.output_dim = encoder_dim
        self.regressor = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim//2),
            nn.BatchNorm1d(encoder_dim//2),
            self.activation,
            nn.Dropout(conformer_args.dropout_p),
            nn.Linear(encoder_dim//2, args.output_dim*args.num_quantiles)
        )
    
    def forward(self, x, y=None, return_all=False):
        x_enc, x = self.encoder(x)
        if len(x.shape) == 3:
            x = x.permute(0,2,1)
        else:
            x = x.unsqueeze(-1)
        # print(x_enc.shape)
        output_reg = self.regressor(x_enc.sum(dim=1))
        output_dec = self.decoder(x)
        return output_reg, output_dec, x_enc


class MultiEncoder(nn.Module):
    def __init__(self, args, conformer_args):
        super().__init__()
       
        self.backbone = CNNEncoder(args)
        self.head_size = conformer_args.encoder_dim // conformer_args.num_heads
        self.rotary_ndims = int(self.head_size * 0.5)
        self.pe = RotaryEmbedding(self.rotary_ndims)
        self.encoder = ConformerEncoder(conformer_args)
        self.output_dim = conformer_args.encoder_dim
        self.avg_output = args.avg_output
        
    def forward(self, x):
        backbone_out = self.backbone(x)
        if len(backbone_out.shape) == 2:
            x_enc = backbone_out.unsqueeze(1)
        else:
            x_enc = backbone_out
        RoPE = self.pe(x_enc, x_enc.shape[1])
        x_enc = self.encoder(x_enc, RoPE)
        # print('avg_output', self.avg_output)
        if (len(x_enc.shape) == 3) and self.avg_output:
            x_enc = x_enc.sum(dim=1)
        return x_enc, backbone_out
class RoPETransformerBlock(nn.Module):
    """Transformer block with RoPE positional encoding"""
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Multi-head attention with RoPE
        self.attention = MHA_rotary(args)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(args.encoder_dim, args.encoder_dim * 4),
            nn.GELU(),
            nn.Linear(args.encoder_dim * 4, args.encoder_dim),
            nn.Dropout(args.dropout_p)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(args.encoder_dim)
        self.ln2 = nn.LayerNorm(args.encoder_dim)
        
        # Dropout
        self.dropout = nn.Dropout(args.dropout_p)
        
    def forward(self, x, RoPE, key_padding_mask=None):
        # Pre-norm architecture
        # Self-attention with residual connection
        attn_out = self.attention(self.ln1(x), RoPE, key_padding_mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)
        
        return x


class RoPETransformer(nn.Module):
    """Transformer encoder with RoPE positional encoding"""
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Model dimensions
        self.encoder_dim = args.encoder_dim
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.dropout_p = args.dropout_p
        
        # RoPE setup
        self.head_size = args.encoder_dim // args.num_heads
        self.rotary_ndims = int(self.head_size * 0.5)
        self.pe = RotaryEmbedding(self.rotary_ndims)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            RoPETransformerBlock(args) for _ in range(self.num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(args.encoder_dim)
        
    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: Input tensor of shape [B, T, C]
            key_padding_mask: Optional mask for padding tokens [B, T]
        
        Returns:
            Output tensor of shape [B, T, C]
        """
        B, T, C = x.size()
        
        # Generate RoPE embeddings
        RoPE = self.pe(x, seq_len=T).nan_to_num(0)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, RoPE, key_padding_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        return x


class RoPETransformerConfig:
    """Configuration class for RoPE Transformer"""
    
    def __init__(self, 
                 encoder_dim=512,
                 num_layers=6,
                 num_heads=8,
                 dropout_p=0.1,
                 timeshift=False):
        self.encoder_dim = encoder_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.timeshift = timeshift
        
        # Validate that encoder_dim is divisible by num_heads
        assert encoder_dim % num_heads == 0, f"encoder_dim ({encoder_dim}) must be divisible by num_heads ({num_heads})"


class SpectralViT(nn.Module):
    """ViT-style architecture treating spectra as 1D images with patches"""
    
    def __init__(self, args, transformer_args):
        super().__init__()
        
        # Extract key parameters
        self.seq_len = args.seq_len if hasattr(args, 'seq_len') else 1024
        self.patch_size = args.patch_size if hasattr(args, 'patch_size') else 16
        self.in_channels = args.in_channels
        self.dim = transformer_args.encoder_dim
        self.depth = args.num_layers
        self.heads = transformer_args.num_heads
        self.num_patches = self.seq_len // self.patch_size
        self.output_dim = args.output_dim
        self.num_quantiles = args.num_quantiles
        
        # Activation
        if args.activation == 'silu':
            self.activation = nn.SiLU()
        elif args.activation == 'sine':
            from nn.models import Sine
            self.activation = Sine(w0=args.sine_w0)
        else:
            self.activation = nn.ReLU()
        
        # Patchify and embed
        self.patch_embed = nn.Conv1d(
            self.in_channels, 
            self.dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # Positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=self.heads,
            dim_feedforward=self.dim * 4,
            dropout=transformer_args.dropout_p,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.depth)
        
        # Decoder for reconstruction
        self.decoder = nn.ModuleList([
            nn.Linear(self.dim, self.dim * 2),
            nn.LayerNorm(self.dim * 2),
            self.activation,
            nn.Linear(self.dim * 2, self.patch_size * self.in_channels)
        ])
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim // 2),
            nn.BatchNorm1d(self.dim // 2),
            self.activation,
            nn.Dropout(transformer_args.dropout_p),
            nn.Linear(self.dim // 2, self.output_dim * self.num_quantiles)
        )
        
    def forward(self, x, y=None, meta=None,
                return_tokens: bool = False,
                return_all: bool = False):
        B = x.shape[0]
        
        # Handle input shape
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[-1] == 1:
            x = x.transpose(-1, -2)
                
        # Patchify: [B, C, L] -> [B, num_patches, dim]
        x = self.patch_embed(x.float())  # [B, dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, dim]
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, dim]
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Transformer encoding
        x_enc = self.transformer(x)
        
        # Split cls token and patches
        cls_output = x_enc[:, 0]  # [B, dim]
        patch_output = x_enc[:, 1:]  # [B, num_patches, dim]
        
        # Regression from cls token
        output_reg = self.regressor(cls_output)
        
        # Reconstruction from patches
        x_dec = patch_output
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                x_dec = layer(x_dec)
            else:
                x_dec = layer(x_dec)
        
        # Reshape to original size
        x_dec = x_dec.reshape(B, self.num_patches, self.patch_size, self.in_channels)
        x_dec = x_dec.permute(0, 3, 1, 2)  # [B, C, num_patches, patch_size]
        output_dec = x_dec.reshape(B, self.in_channels, self.seq_len)
        
        if self.in_channels == 1:
            output_dec = output_dec.squeeze(1)
        
        if return_all:
            # convenient dict API for experiments
            return {
                "regression": output_reg,
                "reconstruction": output_dec,
                "cls": cls_output,
                "tokens": x_enc,          # [B, 1+num_patches, dim]
                "patch_tokens": patch_output,
            }

        if return_tokens:
            # for pure encoder usage in probes
            return x_enc, cls_output, patch_output
        
        return output_reg, output_dec, cls_output


