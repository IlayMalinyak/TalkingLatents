import math
from typing import Optional, Tuple, Dict, Any, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Reuse RoPE utils from the existing implementation
from .llm import apply_rotary_emb, repeat_kv
from llama3.llama.model import precompute_freqs_cis
from nn.cfm import SpectralFlowBridge


class SpectralTokensProjector(nn.Module):
    """
    Project a spectral/latent feature vector to K token embeddings of size d_model.
    Produces a tensor of shape (B, K, d_model).
    """

    def __init__(self, latent_dim: int, d_model: int, hidden_dim: int, num_tokens: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens * d_model),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, latent_dim)
        b = features.size(0)
        x = self.mlp(features)  # (B, K*d_model)
        x = x.view(b, self.num_tokens, self.d_model)  # (B, K, d_model)
        x = self.ln(x)
        return x


class StellarParameterPredictor(nn.Module):
    """Predicts stellar parameters from hidden representations"""

    def __init__(self, hidden_dim: int, stellar_params: List[str] = ['Teff', 'logg', 'FeH']):
        super().__init__()
        self.stellar_params = stellar_params
        self.num_params = len(stellar_params)

        # Parameter-specific prediction heads
        self.param_heads = nn.ModuleDict()
        for param in stellar_params:
            layers = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)
            )

            # Initialize weights to prevent initial NaN issues
            with torch.no_grad():
                for layer in layers:
                    if isinstance(layer, nn.Linear):
                        # Xavier/Glorot initialization
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)

            self.param_heads[param] = layers

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
        Returns:
            Dict of predicted stellar parameters (normalized to 0-1 range)
        """

        if hidden_states.dim() == 3:
            # Pool across sequence dimension (mean pooling)
            # Add small epsilon to avoid division by zero in case of all-zero sequences
            seq_mask = (hidden_states.abs().sum(dim=-1) > 1e-8).float().unsqueeze(-1)  # [batch_size, seq_len, 1]
            if seq_mask.sum() > 0:
                hidden_states = (hidden_states * seq_mask).sum(dim=1) / (seq_mask.sum(dim=1) + 1e-8)  # [batch_size, hidden_dim]
            else:
                hidden_states = hidden_states.mean(dim=1)  # Fallback to simple mean

        # Check for NaN/inf in pooled hidden states
        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            print("Warning: NaN/inf detected in pooled hidden states, using zeros")
            hidden_states = torch.zeros_like(hidden_states)

        # Ensure hidden states match the dtype of the predictor parameters
        predictor_dtype = next(self.param_heads[self.stellar_params[0]].parameters()).dtype
        hidden_states = hidden_states.to(dtype=predictor_dtype)

        predictions = {}
        for param in self.stellar_params:
            raw_pred = self.param_heads[param](hidden_states).squeeze(-1)  # [batch_size]

            # Check for NaN in raw predictions
            if torch.isnan(raw_pred).any() or torch.isinf(raw_pred).any():
                print(f"Warning: NaN/inf detected in raw {param} prediction, using zeros")
                raw_pred = torch.zeros_like(raw_pred)

            # Apply sigmoid to get 0-1 range, then apply small bounds to avoid exact 0/1
            normalized_pred = torch.sigmoid(raw_pred)
            # Clamp to [0.001, 0.999] to avoid extreme values in loss computation
            predictions[param] = torch.clamp(normalized_pred, min=0.001, max=0.999)

        return predictions


class FeaturePredictor(nn.Module):
    """Predicts latent features from hidden representations using MLP"""

    def __init__(self, hidden_dim: int, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

        # MLP for feature prediction
        self.feature_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, feature_dim)
        )

        # Initialize weights
        with torch.no_grad():
            for layer in self.feature_head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
        Returns:
            Predicted features: [batch_size, feature_dim]
        """
        if hidden_states.dim() == 3:
            # Pool across sequence dimension (mean pooling)
            seq_mask = (hidden_states.abs().sum(dim=-1) > 1e-8).float().unsqueeze(-1)  # [batch_size, seq_len, 1]
            if seq_mask.sum() > 0:
                hidden_states = (hidden_states * seq_mask).sum(dim=1) / (seq_mask.sum(dim=1) + 1e-8)  # [batch_size, hidden_dim]
            else:
                hidden_states = hidden_states.mean(dim=1)  # Fallback to simple mean

        # Check for NaN/inf
        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            print("Warning: NaN/inf detected in hidden states for feature prediction, using zeros")
            hidden_states = torch.zeros_like(hidden_states)

        # Predict features
        features = self.feature_head(hidden_states)  # [batch_size, feature_dim]

        # Check for NaN in predictions
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("Warning: NaN/inf detected in feature predictions, using zeros")
            features = torch.zeros_like(features)

        return features


class MultimodalBackboneBase(nn.Module):
    """Shared multimodal utilities (projectors, auxiliary heads, losses)."""

    def __init__(
        self,
        base_model,
        fm_model,
        latent_dim,
        hidden_dim,
        num_spectral_features: int = 8,
        use_checkpoint: bool = True,
        mode: str = "single_star",
        use_cfm: bool = True,
        cfm_weight: float = 0.1,
        predict_stellar_params: bool = True,
        stellar_params: Optional[List[str]] = None,
        quantiles: Optional[List[float]] = None,
        enable_classification: bool = True,
        predict_features: bool = False,
        feature_dim: int = 2048,
        feature_loss_weight: float = 1.0,
        embedding_dim: Optional[int] = None,
        vocab_size: Optional[int] = None,
        feature_stats: Optional[Dict[str, Any]] = None,
        pooling_type: str = 'mean',
    ):
        super().__init__()
        self.pooling_type = pooling_type
        stellar_params = stellar_params or ['Teff', 'logg', 'FeH']
        quantiles = quantiles or [0.159, 0.5, 0.841]

        self.base_model = base_model
        self.fm_model = fm_model
        self.mode = mode
        self.use_checkpoint = use_checkpoint
        self.use_cfm = use_cfm
        self.cfm_weight = cfm_weight
        self.predict_stellar_params = predict_stellar_params
        self.stellar_params = stellar_params
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.enable_classification = enable_classification
        self.predict_features = predict_features
        self.feature_dim = feature_dim
        self.feature_loss_weight = feature_loss_weight
        self.num_spectral_features = int(num_spectral_features)

        self._init_feature_normalization(feature_stats)

        self.embedding_dim = embedding_dim
        if self.embedding_dim is None:
            base_params = getattr(base_model, "params", None)
            self.embedding_dim = getattr(base_params, "dim", None)
        if self.embedding_dim is None and hasattr(base_model, "config"):
            self.embedding_dim = getattr(base_model.config, "hidden_size", None)
        if self.embedding_dim is None:
            raise ValueError("embedding_dim could not be inferred; provide it explicitly.")

        if vocab_size is None:
            base_params = getattr(base_model, "params", None)
            vocab_size = getattr(base_params, "vocab_size", None)
        if vocab_size is None and hasattr(base_model, "config"):
            vocab_size = getattr(base_model.config, "vocab_size", None)
        self.vocab_size = vocab_size

        self._init_projectors(latent_dim, hidden_dim)

        if self.use_cfm:
            if self.vocab_size is None:
                raise ValueError("vocab_size is required when use_cfm is True.")
            self.flow_bridge = SpectralFlowBridge(
                vocab_size=self.vocab_size,
                feature_dim=latent_dim,
                hidden_dim=hidden_dim,
            )
        else:
            self.flow_bridge = None

        self._init_stellar_predictor()
        self._init_classification_head()
        self._init_feature_predictor()

    # ------------------------------------------------------------------ #
    # Initialization helpers
    # ------------------------------------------------------------------ #
    def _init_projectors(self, latent_dim: int, hidden_dim: int) -> None:
        # self.projector_a = SpectralTokensProjector(
        #     latent_dim=latent_dim,
        #     d_model=self.embedding_dim,
        #     hidden_dim=hidden_dim,
        #     num_tokens=self.num_spectral_features,
        # )
        # self.projector_b = SpectralTokensProjector(
        #     latent_dim=latent_dim,
        #     d_model=self.embedding_dim,
        #     hidden_dim=hidden_dim,
        #     num_tokens=self.num_spectral_features,
        # )
        self.projector = SpectralTokensProjector(
            latent_dim=latent_dim,
            d_model=self.embedding_dim,
            hidden_dim=hidden_dim,
            num_tokens=self.num_spectral_features,
        )

    def _init_stellar_predictor(self) -> None:
        if not self.predict_stellar_params:
            self.stellar_transformer = None
            self.stellar_predictor = None
            return
        self.stellar_transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=8,
                dim_feedforward=self.embedding_dim * 2,
                dropout=0.1,
                batch_first=True,
            )
            for _ in range(2)
        ])
        self.stellar_predictor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.LayerNorm(self.embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim // 2, len(self.stellar_params) * self.num_quantiles),
        )

    def _init_classification_head(self) -> None:
        print("creating classification head... ")
        if not self.enable_classification:
            self.classification_head = None
            return
        self.classification_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.LayerNorm(self.embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim // 2, 2),
        )

    def _init_feature_predictor(self) -> None:
        if not self.predict_features:
            self.feature_predictor = None
            return
        print("creating features predictor...")
        self.feature_predictor = FeaturePredictor(
            hidden_dim=self.embedding_dim,
            feature_dim=self.feature_dim,
        ).float()

    def _init_feature_normalization(self, feature_stats: Optional[Dict[str, Any]]) -> None:
        """Initialize normalization buffers if provided"""
        self.register_buffer('feature_mean', None)
        self.register_buffer('feature_std', None)
        
        if feature_stats is not None:
            if 'mean' in feature_stats:
                mean_val = feature_stats['mean']
                if not isinstance(mean_val, torch.Tensor):
                    mean_val = torch.tensor(mean_val)
                self.feature_mean = mean_val
                
            if 'std' in feature_stats:
                std_val = feature_stats['std']
                if not isinstance(std_val, torch.Tensor):
                    std_val = torch.tensor(std_val)
                self.feature_std = std_val
                
            print(f"Initialized feature normalization with mean shape {self.feature_mean.shape} and std shape {self.feature_std.shape}")

    # ------------------------------------------------------------------ #
    # Shared utilities
    # ------------------------------------------------------------------ #
    def _normalize_hidden_for_cfm(self, h: torch.Tensor) -> torch.Tensor:
        mean = h.mean(dim=-1, keepdim=True)
        std = h.std(dim=-1, keepdim=True) + 1e-8
        return (h - mean) / std

    def _compute_cfm_loss(self, h: torch.Tensor, cfm_targets: List[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.use_cfm or self.flow_bridge is None or not cfm_targets:
            return None
        cfm_param_dtype = next(self.flow_bridge.parameters()).dtype
        hidden_states = h.to(dtype=cfm_param_dtype)
        normalized_hidden = self._normalize_hidden_for_cfm(hidden_states)
        cfm_losses = []
        for i, target in enumerate(cfm_targets):
            cfm_target = target.to(dtype=cfm_param_dtype)
            cfm_loss = self.flow_bridge.training_step(normalized_hidden[i:i + 1], cfm_target.unsqueeze(0))
            cfm_losses.append(cfm_loss)
        if cfm_losses:
            return torch.stack(cfm_losses).mean()
        return None

    def _run_stellar_predictor(self, h: torch.Tensor) -> Optional[torch.Tensor]:
        if self.stellar_predictor is None:
            return None
        stellar_h = h.float()
        for layer in self.stellar_transformer:
            stellar_h = layer(stellar_h)
        pooled_h = stellar_h.mean(dim=1)
        return self.stellar_predictor(pooled_h.float())

    def _run_classification_head(self, h: torch.Tensor, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        if self.classification_head is None:
            return None
        comp_mask = batch.get('mode_mask_comparative', None)
        if comp_mask is None or not torch.any(comp_mask):
            return None
        class_head_dtype = next(self.classification_head.parameters()).dtype
        class_h = h.to(dtype=class_head_dtype)
        if self.stellar_transformer is not None:
            class_h_fp32 = class_h.float()
            for layer in self.stellar_transformer:
                class_h_fp32 = layer(class_h_fp32)
            class_h = class_h_fp32.to(dtype=class_head_dtype)
        pooled = class_h.mean(dim=1)
        return self.classification_head(pooled)

    def _run_feature_predictor(self, h: torch.Tensor) -> Optional[torch.Tensor]:
        if self.feature_predictor is None:
            return None
        return self.feature_predictor(h.float())

    def _add_common_outputs(
        self,
        h: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
        cfm_targets: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if cfm_targets is not None:
            cfm_loss = self._compute_cfm_loss(h, cfm_targets)
            if cfm_loss is not None:
                outputs['cfm_loss'] = cfm_loss
        stellar_preds = self._run_stellar_predictor(h)
        if stellar_preds is not None:
            outputs['stellar_predictions'] = stellar_preds
        class_logits = self._run_classification_head(h, batch)
        if class_logits is not None:
            outputs['classification_logits'] = class_logits
        feature_preds = self._run_feature_predictor(h)
        if feature_preds is not None:
            outputs['predicted_features'] = feature_preds
        return outputs

    def _encode_latent_features(self, spectra: torch.Tensor) -> torch.Tensor:
        if self.fm_model is not None:
            self.fm_model.eval()
            with torch.no_grad():
                out = self.fm_model(spectra, return_all=True)
            if isinstance(out, dict):
                latent_features = out['tokens']
            else:
                latent_features = out[-1]
            if torch.isnan(latent_features).any():
                print("DEBUG: NaN detected in fm_model output (latent_features)")
            
            # If still 3D (e.g. tokens), pool it
            if latent_features.dim() == 3:
                if self.pooling_type == 'sum':
                    latent_features = latent_features.sum(dim=1)
                else:
                    latent_features = latent_features.mean(dim=1)
                
            # Apply feature normalization if available
            if self.feature_mean is not None and self.feature_std is not None:
                # Ensure correct device/dtype
                if self.feature_mean.device != latent_features.device:
                    self.feature_mean = self.feature_mean.to(latent_features.device)
                if self.feature_std.device != latent_features.device:
                    self.feature_std = self.feature_std.to(latent_features.device)
                    
                # Normalize: (x - mean) / (std + epsilon)
                # Ensure types match
                mean = self.feature_mean.to(dtype=latent_features.dtype)
                std = self.feature_std.to(dtype=latent_features.dtype)
                
                latent_features = (latent_features - mean) / (std + 1e-6)
        else:
            latent_features = spectra.float()
        return latent_features

    def _project_spectra(self, spectra: torch.Tensor, projector: nn.Module) -> torch.Tensor:
        latent = self._encode_latent_features(spectra)
        if torch.isnan(latent).any():
            print("DEBUG: NaN detected in encoded latent features")
            
        proj_param = next(projector.parameters())
        latent = latent.to(device=proj_param.device, dtype=proj_param.dtype)
        output = projector(latent)
        if torch.isnan(output).any():
            print(f"DEBUG: NaN detected in projector output (input dtype: {latent.dtype}, param dtype: {proj_param.dtype})")
        return output

    def _extract_text_fields(self, batch_data: Dict[str, Any], batch_idx: int) -> Tuple[str, str]:
        input_text = ''
        target_text = ''
        if 'input_texts' in batch_data:
            input_list = batch_data.get('input_texts', [''])
            target_list = batch_data.get('target_texts', [''])
            if len(input_list) > batch_idx:
                input_text = input_list[batch_idx]
            if len(target_list) > batch_idx:
                target_text = target_list[batch_idx]
        elif 'input_text' in batch_data:
            raw_input = batch_data.get('input_text', '')
            raw_target = batch_data.get('target_text', '')
            if isinstance(raw_input, list):
                if len(raw_input) > batch_idx:
                    input_text = raw_input[batch_idx]
            else:
                input_text = raw_input
            if isinstance(raw_target, list):
                if len(raw_target) > batch_idx:
                    target_text = raw_target[batch_idx]
            else:
                target_text = raw_target
        elif 'metadata' in batch_data and batch_data['metadata']:
            meta = batch_data['metadata'][batch_idx]
            if meta and isinstance(meta, dict):
                if 'raw' in meta and isinstance(meta['raw'], dict):
                    input_text = meta['raw'].get('input_text', '') or ''
                    target_text = meta['raw'].get('target_text', '') or ''
                else:
                    input_text = meta.get('input_text', '') or ''
                    target_text = meta.get('target_text', '') or ''
        return input_text or '', target_text or ''


class MultimodalLlamaModelMultiTokens(MultimodalBackboneBase):
    """
    Multimodal wrapper that injects K spectral tokens into the LLaMA token sequence.

    Supports two modes:
    
    Single-star mode (mode="single_star"):
      - Expects the dataloader to provide:
        - 'feature_start_indices': start index for the K tokens
        - 'masked_spectra': spectral data to be processed by fm_model
      - Replaces K consecutive tokens starting from feature_start_indices
    
    Two-star mode (mode="two_star"):
      - Expects the dataloader to provide:
        - 'star_a_feature_indices': exact indices for Star A features 
        - 'star_b_feature_indices': exact indices for Star B features
        - 'star_a_spectra': preprocessed features for Star A
        - 'star_b_spectra': preprocessed features for Star B
      - Replaces tokens at exact positions specified by the indices
    """

    def __init__(self, base_model, fm_model, latent_dim, hidden_dim, num_spectral_features: int = 8,
                 use_checkpoint: bool = True, mode: str = "single_star", use_cfm=True, cfm_weight=0.1,
                 predict_stellar_params: bool = True, stellar_params: List[str] = ['Teff', 'logg', 'FeH'],
                 quantiles: List[float] = [0.159, 0.5, 0.841], enable_classification: bool = True,
                 predict_features: bool = False, feature_dim: int = 2048, feature_loss_weight: float = 1.0, 
                 feature_stats: Optional[Dict[str, Any]] = None, pooling_type: str = 'mean'):
        super().__init__(
            base_model=base_model,
            fm_model=fm_model,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_spectral_features=num_spectral_features,
            use_checkpoint=use_checkpoint,
            mode=mode,
            use_cfm=use_cfm,
            cfm_weight=cfm_weight,
            predict_stellar_params=predict_stellar_params,
            stellar_params=stellar_params,
            quantiles=quantiles,
            enable_classification=enable_classification,
            predict_features=predict_features,
            feature_dim=feature_dim,
            embedding_dim=None,
            vocab_size=None,
            feature_stats=feature_stats,
            pooling_type=pooling_type,
        )
        print("self.embedding dim: ", self.embedding_dim)

    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass that handles both single-star and two-star samples in one pass.
        """
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings from base model
        token_embeddings = self.base_model.tok_embeddings(input_ids)
        
        # Process spectral features for all samples
        cfm_targets = []
        
        # Handle single-star samples
        if 'masked_spectra' in batch and batch['masked_spectra'] is not None:
            single_mask = batch.get('mode_mask_single', torch.ones(batch_size, dtype=torch.bool, device=device))
            single_indices = torch.nonzero(single_mask, as_tuple=False).squeeze(-1)
            
            if single_indices.numel() > 0:
                single_spectra = batch['masked_spectra'].index_select(0, single_indices)
                single_positions = batch['feature_start_indices'].index_select(0, single_indices)
                spec_tokens = self._project_spectra(single_spectra, self.projector)
                if torch.isnan(spec_tokens).any():
                    print("DEBUG: NaN detected in spec_tokens (single_star)")
                
                # Insert tokens at specified positions
                for i, global_idx in enumerate(single_indices):
                    start_pos = single_positions[i].item()
                    end_pos = start_pos + self.num_spectral_features
                    if 0 <= start_pos and end_pos <= seq_len:
                        token_embeddings[global_idx, start_pos:end_pos, :] = spec_tokens[i]
                    else:
                        # Fallback to prefix insertion
                        token_embeddings[global_idx, :self.num_spectral_features, :] = spec_tokens[i]
                
                # Store CFM targets
                cfm_targets.extend([single_spectra[i] for i in range(len(single_indices))])
        
        # Handle two-star samples
        # if 'masked_spectra_a' in batch and batch['masked_spectra_a'] is not None:
        #     comp_mask = batch.get('mode_mask_comparative', torch.ones(batch_size, dtype=torch.bool, device=device))
        #     comp_indices = torch.nonzero(comp_mask, as_tuple=False).squeeze(-1)
            
        #     if comp_indices.numel() > 0:
        #         comp_spectra_a = batch['masked_spectra_a'].index_select(0, comp_indices)
        #         comp_spectra_b = batch['masked_spectra_b'].index_select(0, comp_indices)
        #         comp_indices_a = batch['star_a_feature_indices'].index_select(0, comp_indices)
        #         comp_indices_b = batch['star_b_feature_indices'].index_select(0, comp_indices)
                
        #         spec_tokens_a = self._project_spectra(comp_spectra_a, self.projector)
        #         spec_tokens_b = self._project_spectra(comp_spectra_b, self.projector)
                
        #         # Insert tokens at exact positions
        #         for i, global_idx in enumerate(comp_indices):
        #             indices_a = comp_indices_a[i]
        #             indices_b = comp_indices_b[i]
                    
        #             # Insert star A tokens
        #             valid_indices_a = indices_a[indices_a < seq_len]
        #             if len(valid_indices_a) > 0:
        #                 num_tokens_a = min(len(valid_indices_a), spec_tokens_a.shape[1])
        #                 token_embeddings[global_idx, valid_indices_a[:num_tokens_a], :] = spec_tokens_a[i, :num_tokens_a, :].to(token_embeddings.dtype)
                    
        #             # Insert star B tokens
        #             valid_indices_b = indices_b[indices_b < seq_len]
        #             if len(valid_indices_b) > 0:
        #                 num_tokens_b = min(len(valid_indices_b), spec_tokens_b.shape[1])
        #                 token_embeddings[global_idx, valid_indices_b[:num_tokens_b], :] = spec_tokens_b[i, :num_tokens_b, :].to(token_embeddings.dtype)
                
        #         # Store CFM targets (concatenated for two-star)
        #         cfm_targets.extend([torch.cat([comp_spectra_a[i], comp_spectra_b[i]], dim=-1) for i in range(len(comp_indices))])
        
        # Single transformer forward pass for all samples
        h = self._transformer_forward(token_embeddings, attention_mask=attention_mask)
        logits = self.base_model.output(h).float()
        
        # Check for NaN in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Warning: NaN/inf detected in model logits")
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits), 
                               torch.zeros_like(logits), logits)
        
        outputs = {"logits": logits, "h": h}
        return self._add_common_outputs(h, outputs, batch, cfm_targets)

    def _ensure_generation_capacity(self,
                                    batch_size: int,
                                    required_len: int,
                                    device: torch.device) -> None:
        base_model = self.base_model
        # Extend RoPE cache if needed
        if required_len > base_model.freqs_cis.shape[0]:
            head_dim = base_model.params.dim // base_model.params.n_heads
            new_len = max(required_len, base_model.freqs_cis.shape[0] * 2)
            base_model.freqs_cis = precompute_freqs_cis(
                head_dim,
                new_len,
                base_model.params.rope_theta,
            ).to(base_model.freqs_cis.device)

        for layer in base_model.layers:
            attn = layer.attention
            cache_k = attn.cache_k
            cache_v = attn.cache_v

            # If cache is not initialized, create it with initial size
            if cache_k is None or cache_v is None:
                # Initial size: (batch_size, required_len, n_local_heads, head_dim)
                # n_local_heads and head_dim should be attributes of attn
                n_local_heads = attn.n_local_kv_heads
                head_dim = attn.head_dim
                
                # Infer dtype from model parameters
                cache_dtype = torch.float16
                if hasattr(attn, 'wq'):
                    cache_dtype = attn.wq.weight.dtype
                elif hasattr(base_model, 'tok_embeddings'):
                    cache_dtype = base_model.tok_embeddings.weight.dtype
                
                # Force cache to bfloat16 for stability if needed (optional debug step)
                # cache_dtype = torch.bfloat16
                
                new_shape = (batch_size, required_len, n_local_heads, head_dim)
                attn.cache_k = torch.zeros(new_shape, device=device, dtype=cache_dtype)
                attn.cache_v = torch.zeros(new_shape, device=device, dtype=cache_dtype)
                continue
            
            need_batch = max(batch_size, cache_k.shape[0])
            need_len = max(required_len, cache_k.shape[1])
            if need_batch == cache_k.shape[0] and need_len == cache_k.shape[1]:
                continue
            new_shape = (need_batch, need_len, cache_k.shape[2], cache_k.shape[3])
            new_k = cache_k.new_zeros(new_shape)
            new_v = cache_v.new_zeros(new_shape)
            new_k[:cache_k.shape[0], :cache_k.shape[1]] = cache_k
            new_v[:cache_v.shape[0], :cache_v.shape[1]] = cache_v
            attn.cache_k = new_k
            attn.cache_v = new_v

    def _forward_no_cache(self, input_ids: torch.Tensor, latent_features: torch.Tensor,
                          feature_start_indices) -> Dict[str, torch.Tensor]:
        return self._forward_single_mode(
            input_ids=input_ids,
            latent_features=latent_features,
            feature_start_indices=feature_start_indices,
            start_pos=0,
            use_cache=False,
        )

    def _transformer_forward(self,
                             token_embeddings: torch.Tensor,
                             start_pos: int = 0,
                             use_cache: bool = False,
                             cache_rows: Optional[Sequence[int]] = None,
                             attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Unified transformer forward pass for token embeddings.
        """
        h = token_embeddings
        device = token_embeddings.device
        seqlen = token_embeddings.size(1)
        normalized_rows: Optional[List[int]] = None
        if cache_rows is not None:
            if len(cache_rows) != token_embeddings.size(0):
                raise ValueError("cache_rows must match batch size for cached forward passes")
            normalized_rows = [int(r) for r in cache_rows]

        if use_cache:
            base_model = self.base_model
            required_len = start_pos + seqlen
            effective_batch = token_embeddings.size(0)
            if normalized_rows:
                effective_batch = max(effective_batch, max(normalized_rows) + 1)
            self._ensure_generation_capacity(effective_batch, required_len, device)
            base_model.freqs_cis = base_model.freqs_cis.to(device)
            freqs_cis = base_model.freqs_cis[start_pos:required_len]
            for layer in base_model.layers:
                h = layer(h, start_pos, freqs_cis, mask=None, cache_rows=normalized_rows)
            h = base_model.norm(h)
            return h
        
        # Build RoPE frequencies
        head_dim = self.base_model.params.dim // self.base_model.params.n_heads
        freqs = 1.0 / (self.base_model.params.rope_theta ** (
            torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        t = torch.arange(seqlen, device=device, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        # Causal mask
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=device, dtype=h.dtype)
            mask = torch.triu(mask, diagonal=1)

        if attention_mask is not None:
            # Broadcast attention_mask to match heads: (B, S, S) -> (B, 1, S, S)
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            
            # Broadcast attention_mask to potentially match heads/batch
            if mask is not None:
                mask = mask + attention_mask.to(device=device, dtype=mask.dtype)
            else:
                mask = attention_mask.to(device=device, dtype=h.dtype)

        # --- DEBUG: Visualize Mask (One time) ---
        if not hasattr(self, '_debug_mask_saved'):
            self._debug_mask_saved = True
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                print("DEBUG: Saving attention mask visualization to 'debug_attention_mask.png'...")
                m_cpu = mask
                if m_cpu is not None:
                    m_cpu = m_cpu.detach().float().cpu().numpy()
                    # Handle shapes: (S,S), (B,S,S), (B,1,S,S)
                    if m_cpu.ndim == 3: m_cpu = m_cpu[0]
                    elif m_cpu.ndim == 4: m_cpu = m_cpu[0, 0]
                    
                    plt.figure(figsize=(12, 10))
                    # Use a copy to handle -inf for plotting
                    m_plot = m_cpu.copy()
                    m_plot[m_plot == float('-inf')] = -1000 # Map -inf to a finite low value
                    
                    plt.imshow(m_plot, cmap='viridis', interpolation='nearest')
                    plt.colorbar()
                    plt.title(f"Combined Attention Mask (Shape: {m_cpu.shape})")
                    plt.savefig("/home/ilay.kamai/work/TalkingLatents/figs/debug_attention_mask.png")
                    plt.close()
                    print("DEBUG: Saved.")
                    
                    # Also print a small subsection to stdout
                    print("DEBUG: Mask sample [Followup region?]:")
                    # Heuristic: check middle-ish rows
                    mid = m_cpu.shape[0] // 2
                    print(m_cpu[mid:mid+5, mid-5:mid])
            except Exception as e:
                print(f"DEBUG: Failed to save mask: {e}")
        # ----------------------------------------

        def layer_block(h_in, layer):
            # Attention
            h_norm = layer.attention_norm(h_in)
            attn_out = self._attn_no_cache(h_norm, freqs_cis, mask, layer.attention)
            h_mid = h_in + attn_out
            # FFN
            ff_norm = layer.ffn_norm(h_mid)
            return h_mid + layer.feed_forward(ff_norm)

        for i, layer in enumerate(self.base_model.layers):
            if self.training and self.use_checkpoint:
                h = checkpoint(layer_block, h, layer, use_reentrant=False)
            else:
                h = layer_block(h, layer)
            
            if torch.isnan(h).any():
                print(f"DEBUG: NaN detected after layer {i}")
                # Optional: Break early if NaN
                # break

        h = self.base_model.norm(h)
        if torch.isnan(h).any():
            print("DEBUG: NaN detected after base_model.norm")
        return h

    def _forward_single_mode(self,
                             input_ids: torch.Tensor,
                             latent_features: torch.Tensor,
                             feature_start_indices,
                             start_pos: int = 0,
                             use_cache: bool = False,
                             cache_rows: Optional[Sequence[int]] = None) -> Dict[str, torch.Tensor]:
        bsz, seqlen = input_ids.shape
        device = input_ids.device

        # Token embeddings from base model
        token_embeddings = self.base_model.tok_embeddings(input_ids)

        # Project spectral features to K token embeddings
        spec_tokens = self.projector(latent_features)  # (B, K, d_model)
        # spec_tokens = spec_tokens.to(dtype=token_embeddings.dtype)
        # Normalize feature_start_indices to a 1D tensor of length bsz
        if feature_start_indices is None:
            fsi = torch.zeros(bsz, dtype=torch.long, device=input_ids.device)
        elif isinstance(feature_start_indices, torch.Tensor):
            if feature_start_indices.dim() == 0:
                fsi = feature_start_indices.view(1).repeat(bsz)
            elif feature_start_indices.dim() == 1:
                if feature_start_indices.numel() == bsz:
                    fsi = feature_start_indices.to(device=input_ids.device, dtype=torch.long)
                else:
                    fsi = torch.zeros(bsz, dtype=torch.long, device=input_ids.device)
            else:
                fsi = feature_start_indices.view(-1)[:bsz].to(device=input_ids.device, dtype=torch.long)
        else:
            # python int
            fsi = torch.full((bsz,), int(feature_start_indices), dtype=torch.long, device=input_ids.device)

        # Insert the K tokens per sample at reserved positions
        K = spec_tokens.shape[1]
        for b in range(bsz):
            start_idx = int(fsi[b].item())
            relative_start = start_idx - start_pos
            relative_end = relative_start + K
            if relative_end <= 0 or relative_start >= seqlen:
                continue
            insert_start = max(0, relative_start)
            insert_end = min(seqlen, relative_end)
            spec_start = insert_start - relative_start
            spec_end = spec_start + (insert_end - insert_start)
            token_embeddings[b, insert_start:insert_end, :] = spec_tokens[b, spec_start:spec_end, :]

        h = self._transformer_forward(
            token_embeddings,
            start_pos=start_pos,
            use_cache=use_cache,
            cache_rows=cache_rows,
        )
        logits = self.base_model.output(h).float()
        
        # Check for NaN in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Warning: NaN/inf detected in model logits: ", torch.isnan(logits).sum(), logits.shape)
            # Replace NaN/inf with zeros
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits), 
                               torch.zeros_like(logits), logits)
        
        outputs = {"logits": logits, "h": h}
        
        # Add stellar parameter predictions if enabled
        if self.predict_stellar_params and hasattr(self, 'stellar_predictor'):
            # Ensure hidden states match predictor dtype
            # h_for_predictor = h.to(dtype=next(self.stellar_predictor.parameters()).dtype)
            
            # Check for NaN in hidden states before prediction
            # if torch.isnan(h_for_predictor).any() or torch.isinf(h_for_predictor).any():
            #     print("Warning: NaN/inf detected in hidden states for stellar prediction")
            #     # Create dummy predictions to avoid breaking the forward pass (use 0.5 for normalized range)
            #     predictor_dtype = next(self.stellar_predictor.parameters()).dtype
            #     stellar_preds = {param: torch.full((h.size(0),), 0.5, device=h.device, dtype=predictor_dtype) 
            #                    for param in self.stellar_predictor.stellar_params}
            # else:
            cls_token = h[:, 0, :].float()  # Convert to float32 for stellar predictor
            stellar_preds = self.stellar_predictor(cls_token)  # Use hidden states
                
                # # Check for NaN in stellar predictions
                # for param, pred in stellar_preds.items():
                #     if torch.isnan(pred).any() or torch.isinf(pred).any():
                #         print(f"Warning: NaN/inf detected in stellar prediction for {param}")
                #         # Use small positive values instead of zeros to match normalized range
                #         stellar_preds[param] = torch.full_like(pred, 0.5, dtype=pred.dtype)
                #         print("nans in h_for_predictor: ", torch.isnan(h_for_predictor))
                #         print("nans in h: ", torch.isnan(h))
                #         exit()
            
            outputs['stellar_predictions'] = stellar_preds
        
        return outputs

    def _forward_two_star_mode(self,
                               input_ids: torch.Tensor,
                               star_a_features: torch.Tensor,
                               star_b_features: torch.Tensor,
                               star_a_indices: torch.Tensor,
                               star_b_indices: torch.Tensor,
                               start_pos: int = 0,
                               use_cache: bool = False,
                               cache_rows: Optional[Sequence[int]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for two-star mode using exact feature indices for each star"""
        bsz, seqlen = input_ids.shape
        device = input_ids.device

        # Token embeddings from base model
        token_embeddings = self.base_model.tok_embeddings(input_ids)

        # Project spectral features to K token embeddings for each star
        spec_tokens_a = self.projector_a(star_a_features)  # (B, K, d_model)
        spec_tokens_b = self.projector_b(star_b_features)  # (B, K, d_model)

        spec_tokens_a = spec_tokens_a.to(dtype=token_embeddings.dtype)
        spec_tokens_b = spec_tokens_b.to(dtype=token_embeddings.dtype)

        star_a_indices = torch.as_tensor(star_a_indices, device=device).long()
        star_b_indices = torch.as_tensor(star_b_indices, device=device).long()

        for b in range(bsz):
            indices_a = star_a_indices[b]
            if indices_a.numel() > 0:
                rel_indices_a = indices_a - start_pos
                keep_mask = (rel_indices_a >= 0) & (rel_indices_a < seqlen)
                rel_indices_a = rel_indices_a[keep_mask]
                if rel_indices_a.numel() > 0:
                    num_tokens_a = min(rel_indices_a.numel(), spec_tokens_a.shape[1])
                    token_embeddings[b, rel_indices_a[:num_tokens_a], :] = spec_tokens_a[b, :num_tokens_a, :]

            indices_b = star_b_indices[b]
            if indices_b.numel() > 0:
                rel_indices_b = indices_b - start_pos
                keep_mask = (rel_indices_b >= 0) & (rel_indices_b < seqlen)
                rel_indices_b = rel_indices_b[keep_mask]
                if rel_indices_b.numel() > 0:
                    num_tokens_b = min(rel_indices_b.numel(), spec_tokens_b.shape[1])
                    token_embeddings[b, rel_indices_b[:num_tokens_b], :] = spec_tokens_b[b, :num_tokens_b, :]

        h = self._transformer_forward(
            token_embeddings,
            start_pos=start_pos,
            use_cache=use_cache,
            cache_rows=cache_rows,
        )
        logits = self.base_model.output(h).float()
        
        # Check for NaN in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Warning: NaN/inf detected in model logits")
            # Replace NaN/inf with zeros
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits), 
                               torch.zeros_like(logits), logits)
        
        outputs = {"logits": logits, "h": h}
        return self._add_common_outputs(h, outputs, batch={}, cfm_targets=None)

    def _forward_no_cache_two_star(self, input_ids: torch.Tensor, 
                                   star_a_features: torch.Tensor, star_b_features: torch.Tensor,
                                   star_a_indices: torch.Tensor, star_b_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._forward_two_star_mode(
            input_ids=input_ids,
            star_a_features=star_a_features,
            star_b_features=star_b_features,
            star_a_indices=star_a_indices,
            star_b_indices=star_b_indices,
            start_pos=0,
            use_cache=False,
        )

    def _attn_no_cache(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                        mask: Optional[torch.Tensor], attention_layer) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        # QKV projections
        xq = attention_layer.wq(x)
        xk = attention_layer.wk(x)
        xv = attention_layer.wv(x)
        # reshape
        xq = xq.view(bsz, seqlen, attention_layer.n_local_heads, attention_layer.head_dim)
        xk = xk.view(bsz, seqlen, attention_layer.n_local_kv_heads, attention_layer.head_dim)
        xv = xv.view(bsz, seqlen, attention_layer.n_local_kv_heads, attention_layer.head_dim)
        # rope
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        # repeat kv if needed
        keys = repeat_kv(xk, attention_layer.n_rep)
        values = repeat_kv(xv, attention_layer.n_rep)
        # attention
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(attention_layer.head_dim)
        if mask is not None:
            scores = scores + mask
        probs = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        # --- HARD DEBUG PATCH (CONDITIONAL) ---
        if getattr(attention_layer, 'store_attention', False):
            try:
                # print(f"DEBUG: Capturing attention in _attn_no_cache for {id(attention_layer)}")
                attention_layer.last_attn_scores = probs.detach().cpu()
            except Exception as e:
                print(f"Error capturing attention: {e}")
        # ------------------------

        out = torch.matmul(probs, values)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return attention_layer.wo(out)

    @torch.no_grad()
    def generate_response_from_batch(self,
                                     batch_data: dict,
                                     batch_idx: int = 0,
                                     tokenizer=None,
                                     max_new_tokens: int = 100,
                                     temperature: float = 0.7,
                                     top_p: float = 0.9,
                                     truncate_answer: bool = False) -> tuple:
        """Greedy/top-p generate using the multi-token model.

        Returns: (generated_text, input_text, target_text, generation_log_probs)
        """
        self.eval()

        device = next(self.parameters()).device
        input_ids = batch_data['input_ids'][batch_idx:batch_idx+1].to(device)
        
        # Determine the mode for this specific sample
        def _normalize_mode(mode_value):
            if mode_value is None:
                return None
            if isinstance(mode_value, torch.Tensor):
                if mode_value.ndim == 0:
                    mode_value = mode_value.item()
                else:
                    mode_value = mode_value.tolist()
            if isinstance(mode_value, (list, tuple)):
                mode_value = mode_value[0]
            if isinstance(mode_value, bytes):
                mode_value = mode_value.decode('utf-8', errors='ignore')
            mode_str = str(mode_value).lower()
            if mode_str in {"two_star", "comparative", "comparison", "pair", "dual"}:
                return "two_star"
            if mode_str in {"single_star", "single", "singlemode", "one_star"}:
                return "single_star"
            # Combined mode is determined per-sample by additional metadata
            return None

        raw_mode = None
        if 'mode' in batch_data and batch_data['mode']:
            try:
                raw_mode = batch_data['mode'][batch_idx]
            except (IndexError, KeyError, TypeError):
                raw_mode = None
        current_mode = _normalize_mode(raw_mode)

        if current_mode is None:
            comp_mask = batch_data.get('mode_mask_comparative')
            if comp_mask is not None:
                try:
                    if isinstance(comp_mask, torch.Tensor):
                        current_mode = "two_star" if bool(comp_mask[batch_idx].item()) else "single_star"
                    else:
                        current_mode = "two_star" if bool(comp_mask[batch_idx]) else "single_star"
                except (IndexError, TypeError):
                    current_mode = None

        if current_mode is None:
            has_two_star_features = (
                batch_data.get('masked_spectra_a') is not None and
                batch_data.get('masked_spectra_b') is not None
            )
            current_mode = "two_star" if has_two_star_features else "single_star"

        input_spectra = batch_data['masked_spectra'][batch_idx:batch_idx+1].to(device)
        feature_start_raw = batch_data['feature_start_indices'][batch_idx]
        if isinstance(feature_start_raw, torch.Tensor):
            feature_start_idx = feature_start_raw.to(device)
        else:
            feature_start_idx = torch.tensor(int(feature_start_raw), device=device, dtype=torch.long)
        answer_start_raw = batch_data['answer_start_indices'][batch_idx]
        answer_start_idx = answer_start_raw.item() if isinstance(answer_start_raw, torch.Tensor) else int(answer_start_raw)

        input_text, target_text = self._extract_text_fields(batch_data, batch_idx)

        # Prompt = features + question (truncate before answer start)
        prompt = input_ids[:, :max(1, min(answer_start_idx, input_ids.shape[1]))].clone()
        gen_logps = []
        gen_ids = []

        # Extract special tokens
        pad_id = getattr(tokenizer, 'pad_id', getattr(tokenizer, 'pad_token_id', 0)) if tokenizer else 0
        eos_id = getattr(tokenizer, 'eos_id', getattr(tokenizer, 'eos_token_id', None)) if tokenizer else None

        def sample_top_p(logits: torch.Tensor) -> int:
            # Check for NaN or inf in logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"Warning: NaN or inf detected in logits, returning fallback token")
                return 0  # Return a safe token ID
            
            # Clamp logits to prevent numerical issues
            logits = torch.clamp(logits, min=-1e4, max=1e4)
            
            if temperature > 0:
                logits = logits / temperature
            
            # Suppress padding token
            if pad_id is not None:
                logits[pad_id] = float('-inf')
            
            # Safe softmax with numerical stability
            logits_max = logits.max()
            logits = logits - logits_max  # Subtract max for numerical stability
            probs = torch.softmax(logits, dim=-1)
            
            # Check for NaN in probabilities
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                print(f"Warning: NaN or inf detected in probabilities, using uniform distribution")
                probs = torch.ones_like(probs) / probs.size(0)
            
            if 0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cdf = torch.cumsum(sorted_probs, dim=-1)
                cutoff = (cdf > top_p).float().argmax().item()
                cutoff = max(1, cutoff)
                sorted_probs = sorted_probs[:cutoff]
                sorted_idx = sorted_idx[:cutoff]
                
                # Ensure probabilities sum to 1 and are valid
                prob_sum = sorted_probs.sum()
                if prob_sum > 0:
                    sorted_probs = sorted_probs / prob_sum
                else:
                    # Fallback to uniform distribution
                    sorted_probs = torch.ones_like(sorted_probs) / len(sorted_probs)
                
                # Additional safety check
                if torch.isnan(sorted_probs).any() or (sorted_probs < 0).any():
                    print(f"Warning: Invalid probabilities detected, using first token")
                    return sorted_idx[0].item()
                
                next_idx = torch.multinomial(sorted_probs, 1).item()
                return sorted_idx[next_idx].item()
            else:
                # Additional safety check for full distribution
                if torch.isnan(probs).any() or (probs < 0).any():
                    print(f"Warning: Invalid probabilities detected, returning first token")
                    return 0
                
                return torch.multinomial(probs, 1).item()

        # Ensure features fed to projector match projector dtype/device
        proj_param = next(self.projector.parameters())
        features_vec = self._encode_latent_features(input_spectra)
        features_vec = features_vec.view(prompt.size(0), -1).to(device=proj_param.device, dtype=proj_param.dtype)

        for _ in range(max_new_tokens):
            out = self._forward_single_mode(prompt, features_vec,
                                            feature_start_idx,
                                            start_pos=0,
                                            use_cache=False,
                                            )
            logits = out['logits'][:, -1, :].squeeze(0)
            # Log prob of chosen token
            if temperature > 0:
                logits_scaled = logits / temperature
            else:
                logits_scaled = logits
            probs = torch.softmax(logits_scaled, dim=-1)
            next_token = sample_top_p(logits)
            gen_ids.append(next_token)
            gen_logps.append(torch.log(probs[next_token]).item())

            # Append and continue
            next_tensor = torch.tensor([[next_token]], device=device, dtype=prompt.dtype)
            prompt = torch.cat([prompt, next_tensor], dim=1)

            # Stop on EOS if available
            if eos_id is not None:
                if next_token == eos_id:
                    break
        
        generated_text = tokenizer.decode(gen_ids) if tokenizer is not None else ''
        return generated_text, input_text, target_text, gen_logps, gen_ids
