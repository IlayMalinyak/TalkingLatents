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
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Force LoRA weights to use bfloat16 even if base layer is quantized (uint8/int8)
        # This prevents "normal_kernel_cuda not implemented for 'Byte'" error
        lora_dtype = torch.bfloat16 if dtype in [torch.uint8, torch.int8, None] else dtype
        
        # LoRA matrices - initialize on correct device with float dtype
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features, device=device, dtype=lora_dtype) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, device=device, dtype=lora_dtype)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # Convert input to bfloat16 for LoRA computation if needed
        # This ensures we work in a stable floating point format even if input is from quantized layer
        x_lora = x.to(dtype=torch.bfloat16) if x.dtype != torch.bfloat16 else x
        
        # LoRA parameters stay in bfloat16 for gradient stability
        # x_lora shape: (..., in_features)
        # LoRA forward: x @ A^T @ B^T * scaling
        lora_out = (x_lora @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return self.dropout(lora_out)


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation - supports both regular Linear and FairScale parallel layers"""

    def __init__(self, original_layer, rank: int = 16, alpha: float = 16.0, dropout: float = 0.1):
        super().__init__()
        self.original_layer = original_layer

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
        )

        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Get base layer output (may be in various dtypes from quantization)
        original_out = self.original_layer(x)
        
        # Compute LoRA delta (always in bfloat16)
        lora_out = self.lora(x)
        
        # Ensure both outputs are in the same dtype before adding
        # Convert to bfloat16 for numerical stability
        if original_out.dtype != torch.bfloat16:
            original_out = original_out.to(dtype=torch.bfloat16)
        
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
                lora_layer = LoRALinear(child_module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(module, child_name, lora_layer)
                lora_modules[full_name] = lora_layer
            else:
                # Recursively apply to children
                replace_with_lora(child_module, full_name)

    replace_with_lora(model)
    return lora_modules