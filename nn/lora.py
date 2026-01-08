import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Dict, Union
from nn.llm import apply_rotary_emb, repeat_kv, LatentFeatureEncoder 

class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.1,
        device=None,
        dtype=None,
        name="lora_layer"
    ):
        super().__init__()
        self.name = name
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Force LoRA weights to use bfloat16 even if base layer is quantized (uint8/int8)
        # This prevents "normal_kernel_cuda not implemented for 'Byte'" error
        # Force LoRA weights to use bfloat16 for stability
        # avoiding float16 overflow issues
        lora_dtype = torch.bfloat16
        
        # LoRA matrices - initialize on correct device with float dtype
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features, device=device, dtype=lora_dtype) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, device=device, dtype=lora_dtype)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # Convert input to bfloat16 for LoRA computation
        # This provides better stability than float16
        target_dtype = self.lora_A.dtype
        x_lora = x.to(dtype=target_dtype) if x.dtype != target_dtype else x
        
        if torch.isnan(x_lora).any() or torch.isinf(x_lora).any():
            print(f"DEBUG: NaN/Inf detected in LoRA input ({self.name} rank {self.rank})")
        
        # LoRA forward: x @ A^T @ B^T * scaling
        lora_out = (x_lora @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        if torch.isnan(lora_out).any() or torch.isinf(lora_out).any():
            print(f"DEBUG: NaN/Inf detected in LoRA output ({self.name} rank {self.rank})")

        return self.dropout(lora_out)


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation - supports both regular Linear and FairScale parallel layers"""

    def __init__(self, original_layer, rank: int = 16, alpha: float = 16.0, dropout: float = 0.1, name: str = "lora_linear"):
        super().__init__()
        self.original_layer = original_layer
        self.name = name

        # Get device and dtype from original layer
        p = next(original_layer.parameters())
        device = p.device
        dtype = p.dtype

        # Handle different layer types
        if hasattr(original_layer, 'in_features') and hasattr(original_layer, 'out_features'):
            # Regular nn.Linear
            in_features = original_layer.in_features
            out_features = original_layer.out_features
        elif hasattr(original_layer, 'input_size') and hasattr(original_layer, 'output_size'):
            # FairScale parallel layers
            in_features = original_layer.input_size
            out_features = original_layer.output_size
        else:
            # Fallback - inspect weight shape
            weight = original_layer.weight
            out_features, in_features = weight.shape

        self.lora = LoRALayer(
            in_features,
            out_features,
            rank,
            alpha,
            dropout,
            device=device,
            dtype=dtype,
            name=name
        )

        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Get base layer output (may be in various dtypes from quantization)
        # Cast x to match original layer weights if needed
        if hasattr(self.original_layer, 'weight') and x.dtype != self.original_layer.weight.dtype:
            x_base = x.to(dtype=self.original_layer.weight.dtype)
        else:
            x_base = x
        original_out = self.original_layer(x_base)
        
        # Compute LoRA delta (always in bfloat16)
        lora_out = self.lora(x)
        
        # Ensure lora_out matches original_out dtype
        if lora_out.dtype != original_out.dtype:
            lora_out = lora_out.to(dtype=original_out.dtype)
        
        return original_out + lora_out


def apply_lora_to_model(model: nn.Module, target_modules: List[str], rank: int = 16, alpha: float = 16.0,
                        dropout: float = 0.1):
    """Apply LoRA to specified modules in the model"""
    
    # Get supported linear layer types
    try:
        from fairscale.nn.model_parallel.layers import RowParallelLinear, ColumnParallelLinear
        linear_types = (nn.Linear, RowParallelLinear, ColumnParallelLinear)
        print("Using FairScale parallel layers support")
    except ImportError:
        linear_types = (nn.Linear,)
        print("FairScale not available, using only torch.nn.Linear")
    
    lora_modules = {}

    def replace_with_lora(module, name=""):
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            # Check if this is a target module - EXACT MATCH
            if full_name in target_modules and isinstance(child_module, linear_types):
                print(f"Applying LoRA to: {full_name} ({type(child_module).__name__})")
                lora_layer = LoRALinear(child_module, rank=rank, alpha=alpha, dropout=dropout, name=full_name)
                setattr(module, child_name, lora_layer)
                lora_modules[full_name] = lora_layer
            else:
                # Recursively apply to children
                replace_with_lora(child_module, full_name)

    replace_with_lora(model)
    return lora_modules