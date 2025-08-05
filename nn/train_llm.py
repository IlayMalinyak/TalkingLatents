
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from typing import Dict, Any, List
import re

# Import our custom modules
from llama3.llama.model_single_gpu import Transformer, ModelArgs  # Your non-parallel LLaMA
from llama3.llama.tokenizer import Tokenizer
from data.star_dataset import StarDataset, setup_dataloaders, create_sample_data
from nn.llm import MultimodalLlamaModel, create_multimodal_llama, train_step, MODEL_PATH, TOKENIZER_PATH
from util.utils import merge_dataframes_and_filter_array


def extract_predicted_parameters_differentiable(logits: torch.Tensor,
                                                param_loss_mask: torch.Tensor,
                                                vocab_to_value_map: torch.Tensor) -> torch.Tensor:
    """
    Extract predicted numerical values using differentiable operations

    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        param_loss_mask: Mask indicating parameter token positions [batch_size, seq_len]
        vocab_to_value_map: Mapping from vocab indices to numerical values [vocab_size]

    Returns:
        Predicted parameters [batch_size, 3] for [Teff, logg, Lstar]
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Use softmax to get probability distribution (differentiable)
    probs = F.softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]

    # Compute expected value using the probability distribution
    # This is differentiable unlike argmax
    expected_values = torch.matmul(probs, vocab_to_value_map.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len]

    # Apply parameter mask to extract only parameter positions
    masked_values = expected_values * param_loss_mask  # Zero out non-parameter positions

    # Sum over sequence length and divide by number of parameter tokens
    # This assumes we have 3 parameters per sample
    param_sum = masked_values.sum(dim=1)  # [batch_size]
    param_count = param_loss_mask.sum(dim=1).clamp(min=1)  # Avoid division by zero

    avg_param_value = param_sum / param_count  # [batch_size]

    # For now, return the same value for all 3 parameters
    # You can modify this to extract specific parameters if you have position info
    predicted_params = avg_param_value.unsqueeze(1).repeat(1, 3)  # [batch_size, 3]

    return predicted_params


def create_vocab_to_value_mapping(tokenizer, device='cuda'):
    """
    Create a mapping from vocabulary indices to numerical values
    This is a simplified version - you'll need to customize this based on your tokenizer
    """
    vocab_size = tokenizer.n_words
    vocab_to_value = torch.zeros(vocab_size, device=device)

    # Try to map number tokens to their values
    for i in range(vocab_size):
        try:
            token_str = tokenizer.decode([i])
            # Try to extract number from token
            if token_str.isdigit():
                vocab_to_value[i] = float(token_str)
            elif '.' in token_str and token_str.replace('.', '').isdigit():
                vocab_to_value[i] = float(token_str)
            else:
                # For non-numeric tokens, assign a default value or use position-based heuristic
                vocab_to_value[i] = 0.0
        except:
            vocab_to_value[i] = 0.0

    return vocab_to_value


def extract_predicted_parameters_hybrid(tokenizer, logits: torch.Tensor,
                                        param_loss_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Hybrid approach: Use differentiable operations for gradient flow,
    but also provide interpretable parameter extraction

    Args:
        tokenizer: The tokenizer
        logits: Model output logits [batch_size, seq_len, vocab_size]
        param_loss_mask: Mask indicating parameter token positions [batch_size, seq_len]

    Returns:
        Predicted parameters [batch_size, 3] for [Teff, logg, Lstar]
    """
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device

    if param_loss_mask is None:
        # If no mask provided, use all positions (fallback)
        param_loss_mask = torch.ones(batch_size, seq_len, device=device)

    # Method 1: Differentiable approach using temperature-based sampling
    # Use low temperature to approximate argmax but keep differentiability
    temperature = 0.1
    soft_predictions = F.softmax(logits / temperature, dim=-1)  # [batch_size, seq_len, vocab_size]

    # Create a simple mapping from token indices to approximate numerical values
    # This is a heuristic - you might want to improve this based on your specific tokenizer
    token_values = torch.arange(vocab_size, device=device, dtype=torch.float32)

    # Compute expected token index (differentiable)
    expected_token_indices = torch.sum(soft_predictions * token_values, dim=-1)  # [batch_size, seq_len]

    # Apply parameter mask
    if param_loss_mask is not None:
        masked_predictions = expected_token_indices * param_loss_mask
        # Average over parameter positions
        param_counts = param_loss_mask.sum(dim=1).clamp(min=1)
        param_averages = masked_predictions.sum(dim=1) / param_counts  # [batch_size]
    else:
        # Fallback: average over all positions
        param_averages = expected_token_indices.mean(dim=1)  # [batch_size]

    # Scale to reasonable parameter ranges (this is a heuristic)
    # You'll want to adjust these based on your actual parameter ranges
    teff_pred = param_averages * 10 + 3000  # Scale to temperature range
    logg_pred = (param_averages / vocab_size) * 5  # Scale to log(g) range
    lstar_pred = torch.abs(param_averages / vocab_size) * 100  # Scale to luminosity range

    predicted_params = torch.stack([teff_pred, logg_pred, lstar_pred], dim=1)  # [batch_size, 3]

    return predicted_params


def extract_numerical_values_from_text(text: str) -> Dict[str, float]:
    """
    Extract numerical values for stellar parameters from generated text

    Args:
        text: Generated text containing stellar parameters

    Returns:
        Dictionary with extracted parameter values
    """
    extracted = {}

    # Patterns for different parameters
    patterns = {
        'Teff': [
            r'(?:Teff[=\s]*|temperature[=\s]*|T[=\s]*)(\d+(?:\.\d+)?)\s*K',
            r'(\d+(?:\.\d+)?)\s*K',  # Just number followed by K
        ],
        'logg': [
            r'(?:logg[=\s]*|log\(g\)[=\s]*|gravity[=\s]*)(\d+(?:\.\d+)?)',
            r'log\(g\)\s*[=\s]*(\d+(?:\.\d+)?)',
        ],
        'Lstar': [
            r'(?:Lstar[=\s]*|L[=\s]*|luminosity[=\s]*)(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*(?:solar|L_sun|L☉)',
        ]
    }

    for param_name, param_patterns in patterns.items():
        for pattern in param_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    extracted[param_name] = value
                    break  # Found value for this parameter
                except ValueError:
                    continue

    return extracted

class EarlyStopping:
    """Early stopping utility class"""

    def __init__(self, patience: int = 5, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Initialize early stopping

        Args:
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore the best model weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module = None) -> bool:
        """
        Check if early stopping criteria is met

        Args:
            val_loss: Current validation loss
            model: Model to save weights from (if restore_best_weights=True)

        Returns:
            True if training should stop, False otherwise
        """
        # Check if this is an improvement
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0

            # Save best weights if requested
            if self.restore_best_weights and model is not None:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        else:
            self.counter += 1

        # Check if we should stop
        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

    def restore_best_model(self, model: nn.Module):
        """Restore the best model weights"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            print(f"Restored best model weights (validation loss: {self.best_loss:.4f})")
        else:
            print("Warning: No best weights to restore")


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer"""

    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 16.0, dropout: float = 0.1,
                 device=None):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices - initialize on correct device
        self.lora_A = nn.Parameter(torch.randn(rank, in_features, device=device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # x shape: (..., in_features)
        # LoRA forward: x @ A^T @ B^T * scaling
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return self.dropout(lora_out)


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""

    def __init__(self, original_layer: nn.Linear, rank: int = 16, alpha: float = 16.0, dropout: float = 0.1):
        super().__init__()
        self.original_layer = original_layer

        # Get device from original layer
        device = next(original_layer.parameters()).device

        self.lora = LoRALayer(
            original_layer.in_features,
            original_layer.out_features,
            rank, alpha, dropout, device=device
        )

        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        original_out = self.original_layer(x)
        lora_out = self.lora(x)
        return original_out + lora_out


def apply_lora_to_model(model: nn.Module, target_modules: List[str], rank: int = 16, alpha: float = 16.0,
                        dropout: float = 0.1):
    """Apply LoRA to specified modules in the model"""
    lora_modules = {}

    def replace_with_lora(module, name=""):
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            # Check if this is a target module
            if any(target in full_name for target in target_modules) and isinstance(child_module, nn.Linear):
                print(f"Applying LoRA to: {full_name}")
                lora_layer = LoRALinear(child_module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(module, child_name, lora_layer)
                lora_modules[full_name] = lora_layer
            else:
                # Recursively apply to children
                replace_with_lora(child_module, full_name)

    replace_with_lora(model)
    return lora_modules


class StarLlamaTrainer:
    """Complete training pipeline for Star LLaMA with multimodal capabilities, LoRA, and early stopping"""

    def __init__(self,
                 model_checkpoint_path: str,
                 multimodal_checkpoints: str,
                 tokenizer_path: str,
                 latent_features: np.ndarray,
                 metadata_df: pd.DataFrame,
                 config: Dict[str, Any] = None):
        """
        Initialize the trainer

        Args:
            model_checkpoint_path: Path to LLaMA checkpoint directory
            tokenizer_path: Path to tokenizer.model file
            latent_features: Array of shape (n_samples, latent_dim)
            metadata_df: DataFrame with star metadata
            config: Training configuration dictionary
        """
        self.model_checkpoint_path = model_checkpoint_path
        self.multimodal_checkpoints = multimodal_checkpoints
        self.tokenizer_path = tokenizer_path
        self.latent_features = latent_features
        self.metadata_df = metadata_df
        self.num_samples = len(latent_features)

        # Default configuration
        self.config = {
            'batch_size': 4,
            'learning_rate': 5e-4,  # Higher LR for LoRA
            'num_epochs': 10,
            'max_sequence_length': 512,
            'noise_std': 0.01,
            'train_split': 0.8,
            'save_every_n_epochs': 2,
            'eval_every_n_steps': 50,
            'special_token': 'STAR',
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,

            # Early stopping configuration
            'early_stopping_patience': 5,  # Stop after 5 epochs without improvement
            'early_stopping_min_delta': 0.001,  # Minimum improvement threshold
            'restore_best_weights': True,  # Restore best model when stopping

            # Freezing configuration
            'freeze_strategy': 'encoder_only',  # 'encoder_only', 'lora', 'none'
            'encoder_only_epochs': 3,  # Train only encoder for first N epochs
            'encoder_lr_multiplier': 1.0,  # LR multiplier for encoder vs base model
            'lora_lr_multiplier': 1.0,  # LR multiplier for LoRA layers

            # LoRA configuration
            'lora_rank': 16,
            'lora_alpha': 16.0,
            'lora_dropout': 0.1,
            'lora_target_modules': [
                'base_model.layers.*.attention.wq',
                'base_model.layers.*.attention.wk',
                'base_model.layers.*.attention.wv',
                'base_model.layers.*.attention.wo',
                'base_model.layers.*.feed_forward.w1',
                'base_model.layers.*.feed_forward.w2',
                'base_model.layers.*.feed_forward.w3'
            ],
        }

        if config:
            self.config.update(config)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Initialize components
        self.tokenizer = None
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.scheduler = None
        self.current_freeze_state = None
        self.lora_modules = {}

        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['early_stopping_patience'],
            min_delta=self.config['early_stopping_min_delta'],
            restore_best_weights=self.config['restore_best_weights']
        )

    def setup(self):
        """Setup all components for training"""
        print("Setting up training components...")

        # Load tokenizer with custom special token
        print("Loading tokenizer...")
        self.tokenizer = Tokenizer(model_path=self.tokenizer_path)

        # print(f"Added custom special token: {self.config['special_token']} "
        #       f"(ID: {self.tokenizer.get_custom_token_id(self.config['special_token'])})")

        # Load base LLaMA model
        print("Loading base model...")
        self.base_model = self._load_base_model()

        # Create multimodal model
        print("Creating multimodal model...")
        latent_dim = self.latent_features.shape[1]
        self.model = create_multimodal_llama(
            self.base_model,
            latent_dim,
            3,
            self.tokenizer,
            self.config['special_token']
        )

        if self.multimodal_checkpoints is not None:
            print(f"Loading multimodal checkpoints from {self.multimodal_checkpoints}...")
            checkpoint = torch.load(self.multimodal_checkpoints, map_location='cpu')

            # Check if this checkpoint contains LoRA layers
            state_dict = checkpoint['model_state_dict']
            has_lora = any('lora' in key for key in state_dict.keys())

            if has_lora:
                print("Checkpoint contains LoRA layers, applying LoRA first...")
                # Apply LoRA before loading the state dict
                self._apply_lora()

            self.model.load_state_dict(state_dict)

        self.model.to(self.device)

        # Setup data loaders
        print("Setting up data loaders...")
        self.train_loader, self.val_loader, self.test_loader = setup_dataloaders(
            self.latent_features,
            self.metadata_df,
            self.tokenizer,
            self.config['special_token'],
            batch_size=self.config['batch_size'],
            noise_std=self.config['noise_std'],
            test=True
        )

        # Setup optimizer and scheduler
        print("Setting up optimizer...")
        self._setup_optimizer()

        print("Setup complete!")
        print(f"Early stopping enabled: patience={self.config['early_stopping_patience']}, "
              f"min_delta={self.config['early_stopping_min_delta']}")

    def _load_base_model(self) -> Transformer:
        """Load the base LLaMA model"""
        # Load model parameters
        with open(Path(self.model_checkpoint_path) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args = ModelArgs(
            max_batch_size=self.config['batch_size'] * 2,
            max_seq_len=self.config['max_sequence_length'],
            **params,
        )

        # Create model
        model = Transformer(model_args)

        # Load checkpoint
        checkpoints = sorted(Path(self.model_checkpoint_path).glob("*.pth"))
        if checkpoints:
            print(f"Loading checkpoint: {checkpoints[0]}")
            checkpoint = torch.load(checkpoints[0], map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
        else:
            print("Warning: No checkpoint found, using random initialization")

        return model

    def _apply_lora(self):
        """Apply LoRA to the model"""
        print("Applying LoRA layers...")

        # First, let's see what modules actually exist in the model
        print("Available modules in model:")
        all_modules = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                all_modules.append(name)

        print(f"Found {len(all_modules)} Linear modules:")
        for i, name in enumerate(all_modules[:10]):  # Show first 10
            print(f"  {name}")
        if len(all_modules) > 10:
            print(f"  ... and {len(all_modules) - 10} more")

        # Convert wildcard patterns to actual module names
        target_modules = []
        import re

        for pattern in self.config['lora_target_modules']:
            if '*' in pattern:
                # Convert pattern to regex, handling multiple wildcards
                pattern_regex = pattern.replace('.', r'\.').replace('*', r'[^.]+')
                pattern_regex = f"^{pattern_regex}$"

                # Find matching modules
                matched_modules = []
                for name in all_modules:
                    if re.match(pattern_regex, name):
                        matched_modules.append(name)
                        target_modules.append(name)

                print(f"Pattern '{pattern}' matched {len(matched_modules)} modules")
                if matched_modules:
                    print(f"  Examples: {matched_modules[:3]}")
            else:
                if pattern in all_modules:
                    target_modules.append(pattern)
                    print(f"Direct match: {pattern}")
                else:
                    print(f"Warning: Pattern '{pattern}' not found in model")

        print(f"\nTotal target modules for LoRA: {len(target_modules)}")
        if target_modules:
            print("Target modules:")
            for i, name in enumerate(target_modules[:5]):
                print(f"  {name}")
            if len(target_modules) > 5:
                print(f"  ... and {len(target_modules) - 5} more")

        if not target_modules:
            print("ERROR: No target modules found! Check your lora_target_modules patterns.")
            print("Available Linear module examples:")
            for name in all_modules[:10]:
                print(f"  {name}")
            return

        self.lora_modules = apply_lora_to_model(
            self.model,
            target_modules,
            rank=self.config['lora_rank'],
            alpha=self.config['lora_alpha'],
            dropout=self.config['lora_dropout']
        )

        print(f"Successfully applied LoRA to {len(self.lora_modules)} modules")

    def _apply_freeze_strategy(self, strategy: str):
        """Apply different freezing strategies to the model"""
        print(f"Applying freeze strategy: {strategy}")

        if strategy == 'encoder_only':
            # Freeze everything except latent encoder AND latent regressor
            for name, param in self.model.named_parameters():
                if 'latent_encoder' in name or 'latent_regressor' in name:
                    param.requires_grad = True
                    print(f"  ✓ Unfrozen: {name}")
                else:
                    param.requires_grad = False

        elif strategy == 'lora':
            # Apply LoRA if not already applied
            if not self.lora_modules:
                self._apply_lora()

            # Freeze base model, enable encoder, regressor, AND LoRA parameters
            for name, param in self.model.named_parameters():
                if 'latent_encoder' in name or 'latent_regressor' in name:
                    param.requires_grad = True
                    print(f"  ✓ Unfrozen (multimodal): {name}")
                elif 'lora' in name:
                    param.requires_grad = True
                    print(f"  ✓ Unfrozen (LoRA): {name}")
                else:
                    param.requires_grad = False

        elif strategy == 'none':
            # Unfreeze everything
            for param in self.model.parameters():
                param.requires_grad = True

        self.current_freeze_state = strategy

        # Verify critical components are trainable
        critical_components = ['latent_encoder', 'latent_regressor']
        for component in critical_components:
            component_params = [p for name, p in self.model.named_parameters()
                                if component in name and p.requires_grad]
            if not component_params:
                print(f"ERROR: No trainable parameters in {component}!")
            else:
                total_params = sum(p.numel() for p in component_params)
                print(f"✓ {component}: {len(component_params)} layers, {total_params:,} trainable params")

        # Print what's trainable
        trainable_modules = set()
        trainable_count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_count += param.numel()
                if 'latent_encoder' in name:
                    trainable_modules.add('latent_encoder')
                elif 'latent_regressor' in name:
                    trainable_modules.add('latent_regressor')
                elif 'lora' in name:
                    trainable_modules.add('lora')
                else:
                    module_type = name.split('.')[0]
                    trainable_modules.add(f'base_{module_type}')

        print(f"Trainable module types: {list(trainable_modules)}")
        print(f"Total trainable parameters: {trainable_count:,}")

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Apply freeze strategy
        self._apply_freeze_strategy(self.config['freeze_strategy'])

        # Get parameter groups with different learning rates
        encoder_params = []
        lora_params = []
        base_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if 'latent_encoder' in name:
                encoder_params.append(param)
            elif 'lora' in name:
                lora_params.append(param)
            else:
                base_params.append(param)

        # Check if we have any trainable parameters
        if not encoder_params and not lora_params and not base_params:
            raise ValueError("No trainable parameters found! Check your freeze strategy.")

        param_groups = []
        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': self.config['learning_rate'] * self.config['encoder_lr_multiplier'],
                'name': 'encoder'
            })

        if lora_params:
            param_groups.append({
                'params': lora_params,
                'lr': self.config['learning_rate'] * self.config['lora_lr_multiplier'],
                'name': 'lora'
            })

        if base_params:
            param_groups.append({
                'params': base_params,
                'lr': self.config['learning_rate'] * 0.1,  # Lower LR for base model
                'name': 'base_model'
            })

        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config['weight_decay']
        )

        # Learning rate scheduler
        total_steps = len(self.train_loader) * self.config['num_epochs']
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config['learning_rate'] * 0.01
        )

        # Print parameter info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")

        # Debug: Print specific trainable parameters by type
        param_counts = {'encoder': 0, 'lora': 0, 'base': 0}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'latent_encoder' in name:
                    param_counts['encoder'] += param.numel()
                elif 'lora' in name:
                    param_counts['lora'] += param.numel()
                else:
                    param_counts['base'] += param.numel()

        print("Trainable parameter breakdown:")
        for param_type, count in param_counts.items():
            if count > 0:
                print(f"  {param_type}: {count:,} parameters")

    def debug_model_setup(self):
        """Debug method to check model setup"""
        print("\n=== DEBUG: Model Setup ===")

        # Check if model has latent_encoder
        has_encoder = hasattr(self.model, 'latent_encoder')
        print(f"Model has latent_encoder: {has_encoder}")

        if has_encoder:
            encoder_params = sum(p.numel() for p in self.model.latent_encoder.parameters())
            print(f"Latent encoder parameters: {encoder_params:,}")

        # Check LoRA modules
        if self.lora_modules:
            lora_params = sum(sum(p.numel() for p in module.lora.parameters())
                              for module in self.lora_modules.values())
            print(f"LoRA parameters: {lora_params:,}")

        # Check parameter requires_grad status by type
        param_types = {'encoder': 0, 'lora': 0, 'base': 0, 'frozen': 0}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                param_types['frozen'] += param.numel()
            elif 'latent_encoder' in name:
                param_types['encoder'] += param.numel()
            elif 'lora' in name:
                param_types['lora'] += param.numel()
            else:
                param_types['base'] += param.numel()

        print("Parameter status by type:")
        for param_type, count in param_types.items():
            print(f"  {param_type}: {count:,} parameters")

        # Test forward pass
        print("Testing forward pass...")
        try:
            # Create dummy batch
            dummy_input = torch.randint(0, 1000, (1, 10)).to(self.device)
            dummy_latent = torch.randn(1, self.latent_features.shape[1]).to(self.device)
            dummy_pos = torch.tensor([5]).to(self.device)

            with torch.no_grad():
                output = self.model(dummy_input, dummy_latent, dummy_pos, start_pos=0)
                print(f"Forward pass successful! Output shape: {output.shape}")
        except Exception as e:
            print(f"Forward pass failed: {e}")

        print("=== End Debug ===\n")

    def update_freeze_strategy(self, new_strategy: str):
        """Update freezing strategy and reinitialize optimizer"""
        print(f"Updating freeze strategy from '{self.current_freeze_state}' to '{new_strategy}'")

        # Update config and reapply
        self.config['freeze_strategy'] = new_strategy
        self._setup_optimizer()

    def get_loss(self, out_dict, target_ids, numerical_targets, loss_mask, param_loss_mask=None):
        """
        Compute loss with differentiable parameter extraction
        """
        logits = out_dict['logits']
        reg_out = out_dict['reg_out']

        # Cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=-100,
            reduction='none'
        )
        ce_loss = ce_loss.view(target_ids.shape)

        numerical_loss = F.mse_loss(reg_out.float(), numerical_targets.float(), reduction='mean')

        # Apply loss masking based on focus mode
        if self.config['loss_focus'] == 'parameters_only':
            # param_weight = 100.0
            # other_weight = 0.01
            # masked_ce_loss = (ce_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)
            # loss = numerical_loss * param_weight + masked_ce_loss * other_weight
            loss = numerical_loss

        elif self.config['loss_focus'] == 'weighted':
            # Weight parameter tokens more heavily than others
            param_weight = self.config.get('param_weight', 10.0)
            other_weight = self.config.get('other_weight', 0.1)
            masked_ce_loss = (ce_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)
            loss = numerical_loss * param_weight + masked_ce_loss * other_weight

        else:  # 'standard'
            # Use original loss mask (all answer tokens)
            loss = (ce_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        return loss

    def train(self):
        """Main training loop with early stopping"""
        print("Starting training...")
        print(f"Early stopping: patience={self.early_stopping.patience}, "
              f"min_delta={self.early_stopping.min_delta}")

        best_val_loss = float('inf')
        step = 0

        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")

            # Handle automatic freeze strategy changes
            if (self.config['freeze_strategy'] == 'encoder_only' and
                    epoch >= self.config['encoder_only_epochs'] and
                    self.current_freeze_state == 'encoder_only'):
                print(f"Switching from encoder-only to LoRA training at epoch {epoch + 1}")
                self.update_freeze_strategy('lora')

            # Training phase
            self.model.train()
            epoch_train_loss = 0
            num_batches = 0

            for batch_idx, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                loss_mask = batch['loss_mask'].to(self.device)
                latent_features = batch['latent_features'].to(self.device)
                special_token_positions = batch['special_token_positions'].to(self.device)
                numerical_targets = batch['numerical_targets'].to(self.device)

                # Training step
                out_dict = train_step(self.model, batch, self.optimizer, self.device)
                loss = self.get_loss(out_dict, target_ids, numerical_targets, loss_mask)

                # Backward pass
                loss.backward()

                self.optimizer.step()

                epoch_train_loss += loss
                num_batches += 1
                step += 1

                # Update learning rate
                self.scheduler.step()

                # Log progress
                if batch_idx % 10 == 0:
                    current_lrs = [group['lr'] for group in self.optimizer.param_groups]
                    lr_str = ', '.join([f"{lr:.2e}" for lr in current_lrs])
                    print(f"  Batch {batch_idx}/{len(self.train_loader)}, "
                          f"Loss: {loss:.4f}, LRs: [{lr_str}], "
                          f"Freeze: {self.current_freeze_state}, "
                          f"ES counter: {self.early_stopping.counter}/{self.early_stopping.patience}")

                # Evaluation
                if step % self.config['eval_every_n_steps'] == 0:
                    val_loss = self.evaluate_model()
                    print(f"  Step {step} - Validation Loss: {val_loss:.4f}")

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self._save_checkpoint(epoch, step, val_loss, is_best=True)

                    self.model.train()  # Back to training mode

            # Epoch summary and early stopping check
            avg_train_loss = epoch_train_loss / num_batches
            val_loss = self.evaluate_model()

            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Average Train Loss: {avg_train_loss:.4f}")
            print(f"  Validation Loss: {val_loss:.4f}")
            print(f"  Current Freeze Strategy: {self.current_freeze_state}")
            print(f"  Early Stopping Counter: {self.early_stopping.counter}/{self.early_stopping.patience}")

            # Check early stopping
            if self.early_stopping(val_loss, self.model):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                print(f"Best validation loss: {self.early_stopping.best_loss:.4f}")

                # Restore best weights if requested
                if self.config['restore_best_weights']:
                    self.early_stopping.restore_best_model(self.model)

                break

            # Save checkpoint
            if (epoch + 1) % self.config['save_every_n_epochs'] == 0:
                self._save_checkpoint(epoch, step, val_loss)

            # Generate sample response
            self._generate_sample_response(epoch)

        # Final summary
        if self.early_stopping.early_stop:
            print(f"\nTraining stopped early due to no improvement in validation loss")
            print(f"Total epochs trained: {epoch + 1}")
        else:
            print(f"\nTraining completed all {self.config['num_epochs']} epochs")

        print(f"Best validation loss achieved: {self.early_stopping.best_loss:.4f}")

    def evaluate_model(self) -> float:
        """Modified evaluation with parameter-focused loss"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                loss_mask = batch['loss_mask'].to(self.device)
                latent_features = batch['latent_features'].to(self.device)
                special_token_positions = batch['special_token_positions'].to(self.device)
                numerical_targets = batch['numerical_targets'].to(self.device)
                with torch.no_grad():
                    logits = self.model(input_ids, latent_features, special_token_positions, start_pos=0)

                batch_loss = self.get_loss(logits, target_ids, numerical_targets, loss_mask)
                total_loss += batch_loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else float('inf')

    def _save_checkpoint(self, epoch: int, step: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'lora_modules': list(self.lora_modules.keys()) if self.lora_modules else [],
            'early_stopping_state': {
                'best_loss': self.early_stopping.best_loss,
                'counter': self.early_stopping.counter,
                'early_stop': self.early_stopping.early_stop
            }
        }

        if is_best:
            filename = f'checkpoints/best_model_lora_{self.num_samples}_samples.pt'
        else:
            filename = f'checkpoints/checkpoint_lora_epoch_{epoch + 1}_{self.num_samples}_samples.pt'

        torch.save(checkpoint, filename)
        print(f"Saved checkpoint: {filename}")

    def _generate_sample_response(self, epoch: int):
        """Generate a sample response to see how the model is doing"""
        try:
            # Get a random sample from validation set
            sample_idx = np.random.randint(0, len(self.val_loader.dataset))
            sample = self.test_loader.dataset[sample_idx]

            # Create a question
            question = f"Describe the physical parameters of this star {self.config['special_token']}"
            question_tokens = self.tokenizer.encode(question, bos=True, eos=False)

            # Generate response
            response = self.model.generate_response(
                question_tokens,
                sample['latent_features'],
                self.tokenizer,
                max_new_tokens=50,
                temperature=0.7,
                device=self.device
            )

            print(f"\n--- Sample Generation (Epoch {epoch + 1}) ---")
            print(f"Question: {question}")
            print(f"Generated Response: {response}")
            print(f"Expected Answer: {sample['answer_text']}")
            print("--- End Sample ---\n")

        except Exception as e:
            print(f"Error generating sample: {e}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load a saved checkpoint"""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Check if this checkpoint contains LoRA layers
        state_dict = checkpoint['model_state_dict']
        has_lora = any('lora' in key for key in state_dict.keys())

        if has_lora and not self.lora_modules:
            print("Checkpoint contains LoRA layers, applying LoRA first...")
            # Apply LoRA before loading the state dict
            self._apply_lora()

        self.model.load_state_dict(state_dict)

        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore early stopping state if available
        if 'early_stopping_state' in checkpoint:
            es_state = checkpoint['early_stopping_state']
            self.early_stopping.best_loss = es_state['best_loss']
            self.early_stopping.counter = es_state['counter']
            self.early_stopping.early_stop = es_state['early_stop']
            print(f"Restored early stopping state: best_loss={es_state['best_loss']:.4f}, "
                  f"counter={es_state['counter']}")

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint

    def evaluate_latent_interpretability(self, num_samples: int = 5):
        """
        Conservative approach to interpreting latent features using the LLM
        Focuses on confidence, comparisons, and validation rather than specific dimension attribution
        """
        print(f"\n--- Latent Feature Interpretability Analysis ({num_samples} samples) ---")

        self.model.eval()
        samples_data = []

        # Collect samples and their predictions
        with torch.no_grad():
            for i in range(min(num_samples, len(self.test_loader.dataset))):
                sample = self.val_loader.dataset[i]
                latent_features = sample['latent_features']

                # Get basic prediction
                question = f"Analyze this stellar object {self.config['special_token']} and provide temperature, gravity, and luminosity estimates."
                question_tokens = self.tokenizer.encode(question, bos=True, eos=False)

                prediction = self.model.generate_response(
                    question_tokens,
                    latent_features,
                    self.tokenizer,
                    max_new_tokens=60,
                    temperature=0.5,
                    device=self.device
                )

                samples_data.append({
                    'index': i,
                    'latent': latent_features,
                    'prediction': prediction,
                    'expected': sample['answer_text'],
                    'latent_stats': {
                        'mean': float(latent_features.mean()),
                        'std': float(latent_features.std()),
                        'min': float(latent_features.min()),
                        'max': float(latent_features.max())
                    }
                })

        # 1. CONFIDENCE ANALYSIS
        print("\n" + "=" * 50)
        print("1. CONFIDENCE & UNCERTAINTY ANALYSIS")
        print("=" * 50)

        for sample in samples_data:
            print(f"\n--- Sample {sample['index'] + 1} ---")
            print(f"Prediction: {sample['prediction']}")
            print(f"Expected: {sample['expected']}")

            # Ask about confidence levels
            confidence_questions = [
                f"How confident are you in your temperature estimate for {self.config['special_token']}? What makes you certain or uncertain?",
                f"Which parameter (temperature, gravity, or luminosity) are you most confident about for {self.config['special_token']} and why?",
                f"Are there any aspects of {self.config['special_token']} that seem ambiguous or contradictory for classification?"
            ]

            for q in confidence_questions[:2]:  # Ask 2 confidence questions per sample
                q_tokens = self.tokenizer.encode(q, bos=True, eos=False)
                confidence_response = self.model.generate_response(
                    q_tokens,
                    sample['latent'],
                    self.tokenizer,
                    max_new_tokens=50,
                    temperature=0.4,
                    device=self.device
                )
                print(f"Q: {q}")
                print(f"A: {confidence_response}")

        # 2. COMPARATIVE ANALYSIS
        print("\n" + "=" * 50)
        print("2. COMPARATIVE ANALYSIS")
        print("=" * 50)

        # Compare pairs of samples
        for i in range(min(3, len(samples_data) - 1)):  # Compare up to 3 pairs
            sample_a = samples_data[i]
            sample_b = samples_data[i + 1]

            print(f"\n--- Comparing Sample {sample_a['index'] + 1} vs Sample {sample_b['index'] + 1} ---")
            print(f"Sample A: {sample_a['prediction']}")
            print(f"Sample B: {sample_b['prediction']}")

            # Create comparison prompt with both latent representations
            comparison_q = f"Compare these two stellar objects. Object A: {self.config['special_token']} Object B: {self.config['special_token']} What are the key differences that lead to different classifications?"

            # For comparison, we'll use sample A's latent (limitation: can't easily pass two latents)
            # This is a known limitation we'll note
            comp_tokens = self.tokenizer.encode(comparison_q, bos=True, eos=False)
            comparison_response = self.model.generate_response(
                comp_tokens,
                sample_a['latent'],  # Using first sample's latent
                self.tokenizer,
                max_new_tokens=70,
                temperature=0.4,
                device=self.device
            )

            print(f"Q: {comparison_q}")
            print(f"A: {comparison_response}")
            print(f"Note: Comparison limited to single latent input")

        # 3. PERTURBATION-STYLE QUESTIONS
        print("\n" + "=" * 50)
        print("3. PERTURBATION & SENSITIVITY ANALYSIS")
        print("=" * 50)

        for sample in samples_data[:3]:  # Analyze first 3 samples
            print(f"\n--- Sample {sample['index'] + 1} Sensitivity ---")
            print(f"Original: {sample['prediction']}")

            # Ask about what would change the assessment
            perturbation_questions = [
                f"If the encoded stellar data in {self.config['special_token']} represented a much hotter star, what would be different in your analysis?",
                f"What changes in {self.config['special_token']} would make you classify this as a giant star instead?",
                f"If {self.config['special_token']} had stronger metallicity signatures, how would your assessment change?"
            ]

            for q in perturbation_questions[:2]:  # Ask 2 perturbation questions
                q_tokens = self.tokenizer.encode(q, bos=True, eos=False)
                pert_response = self.model.generate_response(
                    q_tokens,
                    sample['latent'],
                    self.tokenizer,
                    max_new_tokens=60,
                    temperature=0.4,
                    device=self.device
                )
                print(f"Q: {q}")
                print(f"A: {pert_response}")

        # 4. STATISTICAL ANALYSIS OF LATENT SPACE
        print("\n" + "=" * 50)
        print("4. LATENT SPACE STATISTICAL ANALYSIS")
        print("=" * 50)

        # Analyze the relationship between latent statistics and predictions
        for sample in samples_data:
            stats = sample['latent_stats']
            print(f"\nSample {sample['index'] + 1}:")
            print(
                f"  Latent Stats: mean={stats['mean']:.3f}, std={stats['std']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"  Prediction: {sample['prediction']}")

            # Ask about the relationship between statistics and predictions
            stats_q = f"The encoded data {self.config['special_token']} has an average value of {stats['mean']:.2f} and varies by {stats['std']:.2f}. Does this pattern suggest anything about the star type?"
            stats_tokens = self.tokenizer.encode(stats_q, bos=True, eos=False)
            stats_response = self.model.generate_response(
                stats_tokens,
                sample['latent'],
                self.tokenizer,
                max_new_tokens=50,
                temperature=0.4,
                device=self.device
            )
            print(f"  Q: {stats_q}")
            print(f"  A: {stats_response}")

        # 5. VALIDATION QUESTIONS
        print("\n" + "=" * 50)
        print("5. VALIDATION & CONSISTENCY CHECK")
        print("=" * 50)

        for sample in samples_data[:2]:  # Validate first 2 samples
            print(f"\n--- Sample {sample['index'] + 1} Validation ---")

            # Ask the same question multiple times to check consistency
            base_q = f"What type of star is represented by {self.config['special_token']}?"
            responses = []

            for temp in [0.3, 0.7]:  # Different temperatures
                q_tokens = self.tokenizer.encode(base_q, bos=True, eos=False)
                response = self.model.generate_response(
                    q_tokens,
                    sample['latent'],
                    self.tokenizer,
                    max_new_tokens=40,
                    temperature=temp,
                    device=self.device
                )
                responses.append(f"T={temp}: {response}")

            print(f"Original: {sample['prediction']}")
            for response in responses:
                print(f"Consistency Check - {response}")

        print("\n" + "=" * 50)
        print("INTERPRETABILITY ANALYSIS COMPLETE")
        print("=" * 50)
        print("\nKey Insights to Look For:")
        print("1. Does the model show consistent confidence patterns?")
        print("2. Are comparisons between samples physically reasonable?")
        print("3. Do perturbation responses make sense?")
        print("4. Is there correlation between latent statistics and predictions?")
        print("5. Are responses consistent across different temperatures?")

    def evaluate_on_samples(self, num_samples: int = 10, include_followup: bool = True):
        """Evaluate model on a few samples and print results with optional follow-up questions"""
        print(f"\n--- Standard Evaluation on {num_samples} samples ---")

        self.model.eval()
        questions = [
            f"Describe the physical parameters of this star {self.config['special_token']}",
            f"What are the characteristics of this stellar object {self.config['special_token']}?",
            f"What type of star is this {self.config['special_token']}?",
        ]

        # Conservative follow-up questions focused on reasoning rather than specific features
        followup_questions = [
            f"What aspects of the stellar data make you most confident in your temperature estimate?",
            f"Which of your estimates (temperature, gravity, luminosity) do you trust most and why?",
            f"If you had to identify potential uncertainties in your analysis, what would they be?",
            f"How does this star compare to a typical solar-type star in your assessment?",
        ]

        with torch.no_grad():
            for i in range(min(num_samples, len(self.test_loader.dataset))):
                sample = self.test_loader.dataset[i]
                question = str(np.random.choice(questions))
                question_tokens = self.tokenizer.encode(question, bos=True, eos=False)

                # Generate initial response
                response = self.model.generate_response(
                    question_tokens,
                    sample['latent_features'],
                    self.tokenizer,
                    sample['special_token_positions'],
                    max_new_tokens=50,
                    temperature=0.7,
                    device=self.device
                )

                print(f"\nSample {i + 1}:")
                print(f"Initial Question: {question}")
                print(f"Generated Response: {response}")
                print(f"Expected Answer: {sample['answer_text']}")

                # Generate follow-up explanation if requested
                if include_followup:
                    followup_question = str(np.random.choice(followup_questions))

                    # Create a more explicit conversation context
                    conversation_context = f"Q: {question}\nA: {response}\nQ: {followup_question}\nA:"
                    followup_tokens = self.tokenizer.encode(conversation_context, bos=True, eos=False)

                    # Generate explanation response
                    explanation = self.model.generate_response(
                        followup_tokens,
                        sample['latent_features'],
                        self.tokenizer,
                        max_new_tokens=60,
                        temperature=0.5,
                        device=self.device
                    )

                    print(f"Follow-up Question: {followup_question}")
                    print(f"Explanation: {explanation}")


    def print_model_status(self):
        """Print current model parameter status"""
        print("\n=== Model Parameter Status ===")

        total_params = 0
        param_types = {'encoder': 0, 'lora': 0, 'base_trainable': 0, 'base_frozen': 0}

        for name, param in self.model.named_parameters():
            param_count = param.numel()
            total_params += param_count

            if 'latent_encoder' in name:
                if param.requires_grad:
                    param_types['encoder'] += param_count
            elif 'lora' in name:
                if param.requires_grad:
                    param_types['lora'] += param_count
            else:
                if param.requires_grad:
                    param_types['base_trainable'] += param_count
                else:
                    param_types['base_frozen'] += param_count

        trainable_params = param_types['encoder'] + param_types['lora'] + param_types['base_trainable']

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")
        print(f"Current freeze strategy: {self.current_freeze_state}")

        print("\nParameter breakdown:")
        for param_type, count in param_types.items():
            if count > 0:
                pct = 100 * count / total_params
                status = "✓" if 'trainable' in param_type or param_type in ['encoder', 'lora'] else "✗"
                print(f"  {status} {param_type}: {count:,} ({pct:.1f}%)")

        if self.lora_modules:
            print(f"\nLoRA modules applied: {len(self.lora_modules)}")
            print(f"LoRA rank: {self.config['lora_rank']}, alpha: {self.config['lora_alpha']}")

        # Early stopping status
        print(f"\nEarly stopping status:")
        print(f"  Patience: {self.early_stopping.patience}")
        print(f"  Current counter: {self.early_stopping.counter}")
        print(f"  Best validation loss: {self.early_stopping.best_loss:.4f}")
        print(f"  Early stop triggered: {self.early_stopping.early_stop}")

        print("=" * 40)


def main(num_samples=5000):
    """Main function to run training"""

    # Configuration for progressive training with LoRA and early stopping
    config = {
        'special_token': ' STAR',
        'batch_size': 4,
        'learning_rate': 5e-4,  # Higher LR for LoRA
        'num_epochs': 10,  # Set high, early stopping will handle it
        'max_sequence_length': 256,
        'noise_std': 0.002,
        'train_split': 0.8,
        'eval_every_n_steps': 10,
        'save_every_n_epochs': 2,
        'loss_focus': 'standard',

        # Early stopping configuration
        'early_stopping_patience': 7,  # Stop after 7 epochs without improvement
        'early_stopping_min_delta': 0.001,  # Minimum improvement of 0.001
        'restore_best_weights': True,  # Restore best model when stopping

        # Progressive training: start with encoder only, then LoRA
        'freeze_strategy': 'lora',
        'encoder_only_epochs': 3,  # Switch to LoRA after 3 epochs
        'encoder_lr_multiplier': 1.0,
        'lora_lr_multiplier': 1.0,

        # LoRA configuration
        'lora_rank': 32,
        'lora_alpha': 16.0,
        'lora_dropout': 0.1,
        'lora_target_modules': [
            'base_model.layers.*.attention.wq',
            'base_model.layers.*.attention.wk',
            'base_model.layers.*.attention.wv',
            'base_model.layers.*.attention.wo',
            'base_model.layers.*.feed_forward.w1',
            'base_model.layers.*.feed_forward.w2',
            'base_model.layers.*.feed_forward.w3',
            'base_model.output',
            'base_model.tok_embeddings'
        ],
    }

    # Paths (update these to your actual paths)
    model_checkpoint_path = MODEL_PATH
    tokenizer_path = TOKENIZER_PATH
    multimmodal_checkpoints = 'checkpoints/best_model_lora_5000_samples.pt'

    # Create sample data for testing
    print("Creating sample data...")
    latent_features = np.load('logs/2025-07-29/features.npy')
    metadata_df = pd.read_csv('logs/2025-07-29/info.csv')
    metadata_df = metadata_df.loc[:, ~metadata_df.columns.str.contains('^Unnamed')]

    berger_lamost = pd.read_csv('tables/lamost_dr8_with_berger_labels.csv')
    berger_lamost = berger_lamost.loc[:, ~berger_lamost.columns.str.contains('^Unnamed')]

    metadata_df, latent_features = merge_dataframes_and_filter_array(
        berger_lamost, metadata_df, 'obsid', 'obsid', latent_features
    )

    latent_features = latent_features[:num_samples, :]
    metadata_df = metadata_df.iloc[:num_samples]

    # Create trainer
    trainer = StarLlamaTrainer(
        model_checkpoint_path=model_checkpoint_path,
        multimodal_checkpoints=None,
        tokenizer_path=tokenizer_path,
        latent_features=latent_features,
        metadata_df=metadata_df,
        config=config
    )

    try:
        trainer.setup()

        # Debug model setup
        trainer.debug_model_setup()

        # Debug special token detection
        print("Debugging special token detection...")
        trainer.train_loader.dataset.debug_special_token(0)

        # Test first batch to catch gradient issues early
        print("Testing first batch...")
        first_batch = next(iter(trainer.train_loader))

        # Check special token positions
        special_positions = first_batch['special_token_positions']
        print(f"Special token positions: {special_positions}")

        # Quick gradient test
        trainer.model.train()
        input_ids = first_batch['input_ids'].to(trainer.device)
        latent_features = first_batch['latent_features'].to(trainer.device)
        special_token_positions = first_batch['special_token_positions'].to(trainer.device)

        out_dict = trainer.model(input_ids, latent_features, special_token_positions, start_pos=0)
        logits = out_dict['logits']
        print(f"Test forward pass successful! Logits requires_grad: {logits.requires_grad}")

        # Print initial model status
        trainer.print_model_status()

        # Train with automatic freeze strategy transitions and early stopping
        trainer.train()

        # Standard evaluation
        trainer.evaluate_on_samples(num_samples=5, include_followup=True)

        # Comprehensive latent interpretability analysis
        trainer.evaluate_latent_interpretability(num_samples=4)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Still try to restore best weights if available
        if trainer.early_stopping.best_weights and config['restore_best_weights']:
            print("Restoring best model weights before exit...")
            trainer.early_stopping.restore_best_model(trainer.model)
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()

    print("Training completed!")


if __name__ == "__main__":
    main()