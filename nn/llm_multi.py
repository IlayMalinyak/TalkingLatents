import math
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Reuse RoPE utils from the existing implementation
from .llm import apply_rotary_emb, repeat_kv
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


class MultimodalLlamaModelMultiTokens(nn.Module):
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
                 quantiles: List[float] = [0.159, 0.5, 0.841], enable_classification: bool = True):
        super().__init__()
        self.base_model = base_model
        self.fm_model = fm_model
        self.embedding_dim = base_model.params.dim
        print("self.embedding dim: ", self.embedding_dim)
        self.num_spectral_features = int(num_spectral_features)
        self.use_checkpoint = use_checkpoint
        self.mode = mode  # "single_star" or "two_star"
        self.use_cfm = use_cfm
        self.cfm_weight = cfm_weight
        self.predict_stellar_params = predict_stellar_params
        self.stellar_params = stellar_params
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.enable_classification = enable_classification
        
        # For two-star mode, create separate projectors for each star
        # if mode == "two_star":
        self.projector_a = SpectralTokensProjector(
            latent_dim=latent_dim,
            d_model=self.embedding_dim,
            hidden_dim=hidden_dim,
            num_tokens=self.num_spectral_features,
        )
        self.projector_b = SpectralTokensProjector(
            latent_dim=latent_dim,
            d_model=self.embedding_dim,
            hidden_dim=hidden_dim,
            num_tokens=self.num_spectral_features,
        )
    # else:
        self.projector = SpectralTokensProjector(
            latent_dim=latent_dim,
            d_model=self.embedding_dim,
            hidden_dim=hidden_dim,
            num_tokens=self.num_spectral_features,
            )
        if self.use_cfm:
            vocab_size = base_model.params.vocab_size
            self.flow_bridge = SpectralFlowBridge(
                vocab_size=vocab_size,
                feature_dim=latent_dim,
                hidden_dim=hidden_dim
            )
        
        # Initialize stellar parameter predictor
        if self.predict_stellar_params:
            # Small transformer for stellar prediction
            self.stellar_transformer = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=self.embedding_dim,
                    nhead=8,
                    dim_feedforward=self.embedding_dim*2,
                    dropout=0.1,
                    batch_first=True
                ) for _ in range(2)
            ])
            
            self.stellar_predictor = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim//2),
                nn.LayerNorm(self.embedding_dim//2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.embedding_dim//2, len(stellar_params) * self.num_quantiles)
                )
        else:
            self.stellar_predictor = None
        
        # Classification head for comparative questions (STAR_A vs STAR_B)
        if self.enable_classification:
            self.classification_head = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim // 2),
                nn.LayerNorm(self.embedding_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.embedding_dim // 2, 2)  # Binary classification: STAR_A (0) or STAR_B (1)
            )
        else:
            self.classification_head = None
           

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass that handles both single-star and two-star samples in one pass.
        """
        input_ids = batch['input_ids']
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
                
                # Process spectra through FM model if available
                if self.fm_model is not None:
                    self.fm_model.eval()
                    with torch.no_grad():
                        _, _, latent_features = self.fm_model(single_spectra)
                        if latent_features.dim() == 3:
                            latent_features = latent_features.mean(dim=1)
                else:
                    latent_features = single_spectra.float()
                
                # Project to tokens
                proj_param = next(self.projector.parameters())
                latent_features = latent_features.to(device=proj_param.device, dtype=proj_param.dtype)
                spec_tokens = self.projector(latent_features)
                
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
        if 'masked_spectra_a' in batch and batch['masked_spectra_a'] is not None:
            comp_mask = batch.get('mode_mask_comparative', torch.ones(batch_size, dtype=torch.bool, device=device))
            comp_indices = torch.nonzero(comp_mask, as_tuple=False).squeeze(-1)
            
            if comp_indices.numel() > 0:
                comp_spectra_a = batch['masked_spectra_a'].index_select(0, comp_indices)
                comp_spectra_b = batch['masked_spectra_b'].index_select(0, comp_indices)
                comp_indices_a = batch['star_a_feature_indices'].index_select(0, comp_indices)
                comp_indices_b = batch['star_b_feature_indices'].index_select(0, comp_indices)
                
                # Process both stars' spectra
                if self.fm_model is not None:
                    self.fm_model.eval()
                    with torch.no_grad():
                        _, _, latent_a = self.fm_model(comp_spectra_a)
                        _, _, latent_b = self.fm_model(comp_spectra_b)
                        if latent_a.dim() == 3:
                            latent_a = latent_a.mean(dim=1)
                        if latent_b.dim() == 3:
                            latent_b = latent_b.mean(dim=1)
                else:
                    latent_a = comp_spectra_a.float()
                    latent_b = comp_spectra_b.float()
                
                # Project to tokens using same projector
                proj_param = next(self.projector.parameters())
                latent_a = latent_a.to(device=proj_param.device, dtype=proj_param.dtype)
                latent_b = latent_b.to(device=proj_param.device, dtype=proj_param.dtype)
                spec_tokens_a = self.projector(latent_a)
                spec_tokens_b = self.projector(latent_b)
                
                # Insert tokens at exact positions
                for i, global_idx in enumerate(comp_indices):
                    indices_a = comp_indices_a[i]
                    indices_b = comp_indices_b[i]
                    
                    # Insert star A tokens
                    valid_indices_a = indices_a[indices_a < seq_len]
                    if len(valid_indices_a) > 0:
                        num_tokens_a = min(len(valid_indices_a), spec_tokens_a.shape[1])
                        token_embeddings[global_idx, valid_indices_a[:num_tokens_a], :] = spec_tokens_a[i, :num_tokens_a, :].to(token_embeddings.dtype)
                    
                    # Insert star B tokens
                    valid_indices_b = indices_b[indices_b < seq_len]
                    if len(valid_indices_b) > 0:
                        num_tokens_b = min(len(valid_indices_b), spec_tokens_b.shape[1])
                        token_embeddings[global_idx, valid_indices_b[:num_tokens_b], :] = spec_tokens_b[i, :num_tokens_b, :].to(token_embeddings.dtype)
                
                # Store CFM targets (concatenated for two-star)
                cfm_targets.extend([torch.cat([comp_spectra_a[i], comp_spectra_b[i]], dim=-1) for i in range(len(comp_indices))])
        
        # Single transformer forward pass for all samples
        h = self._transformer_forward(token_embeddings)
        logits = self.base_model.output(h).float()
        
        # Check for NaN in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Warning: NaN/inf detected in model logits")
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits), 
                               torch.zeros_like(logits), logits)
        
        outputs = {"logits": logits, "h": h}
        
        # Add CFM loss if enabled
        if self.use_cfm and cfm_targets:
            cfm_param_dtype = next(self.flow_bridge.parameters()).dtype
            hidden_states = h.to(dtype=cfm_param_dtype)
            normalized_hidden = (hidden_states - hidden_states.mean(dim=-1, keepdim=True)) / (hidden_states.std(dim=-1, keepdim=True) + 1e-8)
            
            cfm_losses = []
            for i, target in enumerate(cfm_targets):
                cfm_target = target.to(dtype=cfm_param_dtype)
                cfm_loss = self.flow_bridge.training_step(normalized_hidden[i:i+1], cfm_target.unsqueeze(0))
                cfm_losses.append(cfm_loss)
            
            if cfm_losses:
                outputs['cfm_loss'] = torch.stack(cfm_losses).mean()
        
        # Add stellar parameter predictions if enabled
        if self.stellar_predictor is not None:
            # Convert to FP32 for stellar components (they are kept in FP32 for stability)
            stellar_h = h.float()
            
            # Pass full sequence through small transformer
            for layer in self.stellar_transformer:
                stellar_h = layer(stellar_h)
            
            # Pool the sequence dimension (mean pooling)
            pooled_h = stellar_h.mean(dim=1)
            
            # Final prediction from pooled representation
            stellar_preds = self.stellar_predictor(pooled_h.float())  # Convert to float32 for stellar predictor
            outputs['stellar_predictions'] = stellar_preds
        
        # Add classification predictions for comparative samples
        if self.classification_head is not None:
            # Get comparative mask to identify which samples are comparative
            comp_mask = batch.get('mode_mask_comparative', torch.zeros(batch_size, dtype=torch.bool, device=device))
            
            if comp_mask.any():
                # Get classification head's target dtype for memory efficiency
                if hasattr(self, 'classification_head') and self.classification_head is not None:
                    class_head_dtype = next(self.classification_head.parameters()).dtype
                else:
                    class_head_dtype = h.dtype
                
                # Use classification head's dtype (could be FP16/BF16 for memory efficiency)
                class_h = h.to(dtype=class_head_dtype)
                
                # Pass full sequence through small transformer (reuse stellar transformer)
                if hasattr(self, 'stellar_transformer') and self.stellar_transformer is not None:
                    # Convert to FP32 for stellar transformer (which is kept in FP32), then back
                    class_h_fp32 = class_h.float()
                    for layer in self.stellar_transformer:
                        class_h_fp32 = layer(class_h_fp32)
                    class_h = class_h_fp32.to(dtype=class_head_dtype)
                
                # Pool the sequence dimension (mean pooling)
                pooled_class_h = class_h.mean(dim=1)
                
                # Get classification logits using classification head's native precision
                classification_logits = self.classification_head(pooled_class_h)
                outputs['classification_logits'] = classification_logits
        
        return outputs

    def _transformer_forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Unified transformer forward pass for token embeddings.
        """
        h = token_embeddings
        device = token_embeddings.device
        seqlen = token_embeddings.size(1)
        
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

        def layer_block(h_in, layer):
            # Attention
            h_norm = layer.attention_norm(h_in)
            attn_out = self._attn_no_cache(h_norm, freqs_cis, mask, layer.attention)
            h_mid = h_in + attn_out
            # FFN
            ff_norm = layer.ffn_norm(h_mid)
            return h_mid + layer.feed_forward(ff_norm)

        for layer in self.base_model.layers:
            if self.training and self.use_checkpoint:
                h = checkpoint(layer_block, h, layer, use_reentrant=False)
            else:
                h = layer_block(h, layer)

        h = self.base_model.norm(h)
        return h

    def _forward_no_cache(self, input_ids: torch.Tensor, latent_features: torch.Tensor,
                          feature_start_indices) -> Dict[str, torch.Tensor]:
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
        K = self.num_spectral_features
        for b in range(bsz):
            s = int(fsi[b].item())
            e = s + K
            if 0 <= s and e <= seqlen:
                token_embeddings[b, s:e, :] = spec_tokens[b]
            else:
                # If indices are out of range, fallback to prefix insertion
                token_embeddings[b, :K, :] = spec_tokens[b]

        # Simple transformer forward (no cache), reusing the logic used in your current model
        h = token_embeddings

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

        def layer_block(h_in, layer):
            # Attention
            h_norm = layer.attention_norm(h_in)
            attn_out = self._attn_no_cache(h_norm, freqs_cis, mask, layer.attention)
            h_mid = h_in + attn_out
            # FFN
            ff_norm = layer.ffn_norm(h_mid)
            return h_mid + layer.feed_forward(ff_norm)

        for layer in self.base_model.layers:
            if self.training and self.use_checkpoint:
                h = checkpoint(layer_block, h, layer, use_reentrant=False)
            else:
                h = layer_block(h, layer)

        h = self.base_model.norm(h)
        logits = self.base_model.output(h).float()
        
        # Check for NaN in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Warning: NaN/inf detected in model logits")
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

    def _forward_no_cache_two_star(self, input_ids: torch.Tensor, 
                                   star_a_features: torch.Tensor, star_b_features: torch.Tensor,
                                   star_a_indices: torch.Tensor, star_b_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for two-star mode using exact feature indices for each star"""
        bsz, seqlen = input_ids.shape
        device = input_ids.device

        # Token embeddings from base model
        token_embeddings = self.base_model.tok_embeddings(input_ids)

        # Project spectral features to K token embeddings for each star
        spec_tokens_a = self.projector_a(star_a_features)  # (B, K, d_model)
        spec_tokens_b = self.projector_b(star_b_features)  # (B, K, d_model)

        # Ensure spectral tokens match token embeddings dtype
        spec_tokens_a = spec_tokens_a.to(dtype=token_embeddings.dtype)
        spec_tokens_b = spec_tokens_b.to(dtype=token_embeddings.dtype)

        # Insert the K tokens per sample at exact positions specified by indices
        for b in range(bsz):
            # Replace tokens at star_a_indices positions
            indices_a = star_a_indices[b]  # Should be tensor of K indices
            valid_indices_a = indices_a[indices_a < seqlen]  # Filter out out-of-bounds indices
            if len(valid_indices_a) > 0:
                # Only replace as many tokens as we have valid indices
                num_tokens_a = min(len(valid_indices_a), spec_tokens_a.shape[1])
                token_embeddings[b, valid_indices_a[:num_tokens_a], :] = spec_tokens_a[b, :num_tokens_a, :]

            # Replace tokens at star_b_indices positions  
            indices_b = star_b_indices[b]  # Should be tensor of K indices
            valid_indices_b = indices_b[indices_b < seqlen]  # Filter out out-of-bounds indices
            if len(valid_indices_b) > 0:
                # Only replace as many tokens as we have valid indices
                num_tokens_b = min(len(valid_indices_b), spec_tokens_b.shape[1])
                token_embeddings[b, valid_indices_b[:num_tokens_b], :] = spec_tokens_b[b, :num_tokens_b, :]

        # Simple transformer forward (no cache), reusing the logic used in your current model
        h = token_embeddings

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

        def layer_block(h_in, layer):
            # Attention
            h_norm = layer.attention_norm(h_in)
            attn_out = self._attn_no_cache(h_norm, freqs_cis, mask, layer.attention)
            h_mid = h_in + attn_out
            # FFN
            ff_norm = layer.ffn_norm(h_mid)
            return h_mid + layer.feed_forward(ff_norm)

        for layer in self.base_model.layers:
            if self.training and self.use_checkpoint:
                h = checkpoint(layer_block, h, layer, use_reentrant=False)
            else:
                h = layer_block(h, layer)

        h = self.base_model.norm(h)
        logits = self.base_model.output(h).float()
        
        # Check for NaN in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Warning: NaN/inf detected in model logits")
            # Replace NaN/inf with zeros
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits), 
                               torch.zeros_like(logits), logits)
        
        outputs = {"logits": logits, "h": h}
        
        # Add stellar parameter predictions if enabled
        if self.predict_stellar_params and hasattr(self, 'stellar_predictor'):
            # # Ensure hidden states match predictor dtype
            # h_for_predictor = h.to(dtype=next(self.stellar_predictor.parameters()).dtype)
            
            # # Check for NaN in hidden states before prediction
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
                #         stellar_preds[param] = torch.zeros_like(pred)
            outputs['stellar_predictions'] = stellar_preds
        
        return outputs

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
                                     top_p: float = 0.9) -> tuple:
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

        if current_mode == "two_star":
            star_a_features = batch_data['masked_spectra_a'][batch_idx:batch_idx+1].to(device)
            star_b_features = batch_data['masked_spectra_b'][batch_idx:batch_idx+1].to(device)
            star_a_indices = batch_data['star_a_feature_indices'][batch_idx:batch_idx+1].to(device)
            star_b_indices = batch_data['star_b_feature_indices'][batch_idx:batch_idx+1].to(device)
            answer_start_source = batch_data.get('answer_start_indices', [input_ids.shape[1]])
            answer_start_idx = answer_start_source[batch_idx]
            if isinstance(answer_start_idx, torch.Tensor):
                answer_start_idx = answer_start_idx.item()
        else:
            input_spectra = batch_data['masked_spectra'][batch_idx:batch_idx+1].to(device)
            feature_start_raw = batch_data['feature_start_indices'][batch_idx]
            if isinstance(feature_start_raw, torch.Tensor):
                feature_start_idx = feature_start_raw.to(device)
            else:
                feature_start_idx = torch.tensor(int(feature_start_raw), device=device, dtype=torch.long)
            answer_start_raw = batch_data['answer_start_indices'][batch_idx]
            answer_start_idx = answer_start_raw.item() if isinstance(answer_start_raw, torch.Tensor) else int(answer_start_raw)

        # Handle different batch formats (mixed vs single dataset)
        if 'input_texts' in batch_data:
            # Single dataset format
            input_text = batch_data.get('input_texts', [''])[batch_idx]
            target_text = batch_data.get('target_texts', [''])[batch_idx]
        elif 'metadata' in batch_data and batch_data['metadata']:
            # Mixed dataset format
            meta = batch_data['metadata'][batch_idx]
            if meta and 'raw' in meta:
                input_text = meta.get('input_text', '')
                target_text = meta.get('target_text', '')
            else:
                input_text = ''
                target_text = ''
        else:
            input_text = ''
            target_text = ''

        # Prompt = features + question (truncate before answer start)
        prompt = input_ids[:, :max(1, min(answer_start_idx, input_ids.shape[1]))].clone()
        gen_logps = []
        gen_ids = []

        def sample_top_p(logits: torch.Tensor) -> int:
            # Check for NaN or inf in logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"Warning: NaN or inf detected in logits, returning fallback token")
                return 0  # Return a safe token ID
            
            # Clamp logits to prevent numerical issues
            logits = torch.clamp(logits, min=-1e4, max=1e4)
            
            if temperature > 0:
                logits = logits / temperature
            
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

        # Prepare features based on current sample mode
        if current_mode == "two_star":
            # Ensure features match projector dtype/device
            proj_param_a = next(self.projector_a.parameters())
            proj_param_b = next(self.projector_b.parameters())
            star_a_features = star_a_features.to(device=proj_param_a.device, dtype=proj_param_a.dtype)
            star_b_features = star_b_features.to(device=proj_param_b.device, dtype=proj_param_b.dtype)
        else:
            # Ensure features fed to projector match projector dtype/device
            proj_param = next(self.projector.parameters())
            features_vec = input_spectra.view(prompt.size(0), -1).to(device=proj_param.device, dtype=proj_param.dtype)

        for _ in range(max_new_tokens):
            if current_mode == "two_star":
                out = self._forward_no_cache_two_star(prompt, star_a_features, star_b_features,
                                                      star_a_indices, star_b_indices)
            else:
                out = self._forward_no_cache(prompt, features_vec, feature_start_idx)
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
            if tokenizer is not None and hasattr(tokenizer, 'eos_id'):
                if next_token == getattr(tokenizer, 'eos_id'):
                    break

        generated_text = tokenizer.decode(torch.tensor(gen_ids).cpu().numpy()) if tokenizer is not None else ''
        return generated_text, input_text, target_text, gen_logps
