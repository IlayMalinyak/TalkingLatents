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
            self.param_heads[param] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)
            )
    
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
        Returns:
            Dict of predicted stellar parameters
        """
        if hidden_states.dim() == 3:
            # Pool across sequence dimension (mean pooling)
            hidden_states = hidden_states.mean(dim=1)  # [batch_size, hidden_dim]
        
        predictions = {}
        for param in self.stellar_params:
            raw_pred = self.param_heads[param](hidden_states).squeeze(-1)  # [batch_size]
            
            # Apply reasonable bounds to prevent extreme values
            if param == 'Teff':
                # Temperature: reasonable range 2000-10000K
                predictions[param] = torch.clamp(raw_pred, min=2000.0, max=10000.0)
            elif param == 'logg':
                # Surface gravity: reasonable range 0.0-6.0
                predictions[param] = torch.clamp(raw_pred, min=0.0, max=6.0)
            elif param == 'FeH':
                # Metallicity: reasonable range -5.0 to +1.0
                predictions[param] = torch.clamp(raw_pred, min=-5.0, max=1.0)
            else:
                # For any other parameters, apply general bounds
                predictions[param] = torch.clamp(raw_pred, min=-100.0, max=100.0)
        
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
                 predict_stellar_params: bool = True, stellar_params: List[str] = ['Teff', 'logg', 'FeH']):
        super().__init__()
        self.base_model = base_model
        self.fm_model = fm_model
        self.embedding_dim = base_model.params.dim
        self.num_spectral_features = int(num_spectral_features)
        self.use_checkpoint = use_checkpoint
        self.mode = mode  # "single_star" or "two_star"
        self.use_cfm = use_cfm
        self.cfm_weight = cfm_weight
        self.predict_stellar_params = predict_stellar_params
        self.stellar_params = stellar_params
        
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
            self.stellar_predictor = StellarParameterPredictor(
                hidden_dim=self.embedding_dim,
                stellar_params=self.stellar_params
            )

    def forward(self,
                input_ids: torch.Tensor,
                input_spectra: torch.Tensor,
                special_token_positions: torch.Tensor = None,
                star_a_spectra: torch.Tensor = None,
                star_b_spectra: torch.Tensor = None,
                star_a_indices: torch.Tensor = None,
                star_b_indices: torch.Tensor = None,
                start_pos: int = 0,
                ) -> Dict[str, torch.Tensor]:
        # Handle different modes
        if self.mode == "combined":
            # Combined mode should be handled by forward_mixed method
            raise ValueError("Combined mode should use forward_mixed() method instead of forward()")
        elif self.mode == "two_star":
            # Two-star mode: use separate features for each star
            if star_a_spectra is None or star_b_spectra is None:
                raise ValueError("star_a_features and star_b_features must be provided in two_star mode")
            
            # # Ensure features match projector dtype/device
            # proj_param_a = next(self.projector_a.parameters())
            # proj_param_b = next(self.projector_b.parameters())
            # star_a_spectra = star_a_spectra.to(device=proj_param_a.device, dtype=proj_param_a.dtype)
            # star_b_spectra = star_b_spectra.to(device=proj_param_b.device, dtype=proj_param_b.dtype)
                        
            return self._forward_no_cache_two_star(input_ids, star_a_spectra, star_b_spectra, 
                                                   star_a_indices, star_b_indices)
        else:
            # Single-star mode: original behavior
            if special_token_positions is None:
                raise ValueError("special_token_positions must be provided in single_star mode")
            
            # Derive latent features
            if self.fm_model is not None:
                self.fm_model.eval()
                with torch.no_grad():
                    # Expect fm_model to return (reg_out,ssl_out,features)
                    _, _, latent_features = self.fm_model(input_spectra)
                    # If multi-stage, collapse across stages
                    if latent_features.dim() == 3:  # (B, S, D)
                        latent_features = latent_features.mean(dim=1)
            else:
                latent_features = input_spectra.float()  # (B, D)
                
                # Handle case where input_spectra was concatenated for two-star mode
                # but we're in single-star mode (e.g., during evaluation)
                if latent_features.size(-1) == 4096:
                    # Split concatenated features and use first half
                    latent_features = latent_features[:, :2048]
            # Ensure latent features match projector dtype/device
            proj_param = next(self.projector.parameters())
            latent_features = latent_features.to(device=proj_param.device, dtype=proj_param.dtype)
            outputs = self._forward_no_cache(input_ids, latent_features, special_token_positions)
            # Compute CFM loss if in training mode
            if self.use_cfm:
                # Use hidden states for CFM (richer representation than logits)
                hidden_states = outputs['h']
                
                # Ensure hidden states match CFM bridge dtype (fix dtype mismatch)
                cfm_param_dtype = next(self.flow_bridge.parameters()).dtype
                hidden_states = hidden_states.to(dtype=cfm_param_dtype)
                
                # Get original features for CFM target
                if self.mode == "two_star":
                    # Concatenate both star features for two-star mode
                    cfm_target = torch.cat([star_a_spectra, star_b_spectra], dim=-1)
                else:
                    cfm_target = input_spectra
                
                # Ensure CFM target also matches dtype
                cfm_target = cfm_target.to(dtype=cfm_param_dtype)
                
                # Compute CFM loss using hidden states
                cfm_loss = self.flow_bridge.training_step(hidden_states, cfm_target)
                outputs['cfm_loss'] = cfm_loss
            return outputs

    def forward_mixed(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for mixed batches containing both single-star and two-star samples.
        
        Args:
            batch: Mixed batch from MixedStellarQADataset with mode masks and appropriate data
            
        Returns:
            Dict containing logits, hidden states, and optional CFM loss
        """
        input_ids = batch['input_ids']
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize output tensors
        full_logits = None
        full_hidden = None
        cfm_losses = []
        
        def _allocate_outputs(sample_outputs):
            nonlocal full_logits, full_hidden
            if full_logits is None:
                full_logits = torch.zeros(
                    (batch_size,) + sample_outputs['logits'].shape[1:],
                    dtype=sample_outputs['logits'].dtype,
                    device=device,
                )
                full_hidden = torch.zeros(
                    (batch_size,) + sample_outputs['h'].shape[1:],
                    dtype=sample_outputs['h'].dtype,
                    device=device,
                )
        
        # Get mode masks
        single_mask = batch['mode_mask_single']
        comp_mask = batch['mode_mask_comparative']
        
        # Process single-star samples
        single_indices = torch.nonzero(single_mask, as_tuple=False).squeeze(-1)
        if single_indices.numel() > 0:
            single_input_ids = input_ids.index_select(0, single_indices)
            single_spectra = batch['masked_spectra'].index_select(0, single_indices) if batch['masked_spectra'] is not None else None
            single_positions = batch['feature_start_indices'].index_select(0, single_indices)
            
            if single_spectra is not None:
                # Temporarily set mode to single_star for processing
                original_mode = self.mode
                self.mode = "single_star"
                single_outputs = self.forward(
                    input_ids=single_input_ids,
                    input_spectra=single_spectra,
                    special_token_positions=single_positions,
                )
                self.mode = original_mode
                
                _allocate_outputs(single_outputs)
                full_logits.index_copy_(0, single_indices, single_outputs['logits'])
                full_hidden.index_copy_(0, single_indices, single_outputs['h'])
                
                if 'cfm_loss' in single_outputs:
                    cfm_losses.append(single_outputs['cfm_loss'])
        
        # Process comparative samples
        comp_indices = torch.nonzero(comp_mask, as_tuple=False).squeeze(-1)
        if comp_indices.numel() > 0:
            comp_input_ids = input_ids.index_select(0, comp_indices)
            comp_spectra_a = batch['masked_spectra_a'].index_select(0, comp_indices) if batch['masked_spectra_a'] is not None else None
            comp_spectra_b = batch['masked_spectra_b'].index_select(0, comp_indices) if batch['masked_spectra_b'] is not None else None
            comp_indices_a = batch['star_a_feature_indices'].index_select(0, comp_indices)
            comp_indices_b = batch['star_b_feature_indices'].index_select(0, comp_indices)
            
            if comp_spectra_a is not None and comp_spectra_b is not None:
                # Temporarily set mode to two_star for processing
                original_mode = self.mode
                self.mode = "two_star"
                comp_outputs = self.forward(
                    input_ids=comp_input_ids,
                    input_spectra=None,  # Not used in two-star mode
                    star_a_spectra=comp_spectra_a,
                    star_b_spectra=comp_spectra_b,
                    star_a_indices=comp_indices_a,
                    star_b_indices=comp_indices_b,
                )
                self.mode = original_mode
                
                if full_logits is None:
                    _allocate_outputs(comp_outputs)
                
                full_logits.index_copy_(0, comp_indices, comp_outputs['logits'])
                full_hidden.index_copy_(0, comp_indices, comp_outputs['h'])
                
                if 'cfm_loss' in comp_outputs:
                    cfm_losses.append(comp_outputs['cfm_loss'])
        
        # Combine outputs
        outputs = {
            'logits': full_logits,
            'h': full_hidden,
        }
        
        # Add CFM loss if available
        if cfm_losses:
            outputs['cfm_loss'] = torch.stack(cfm_losses).mean()
        
        # Add stellar parameter predictions if enabled
        if self.predict_stellar_params and hasattr(self, 'stellar_predictor'):
            # Ensure hidden states match predictor dtype
            h_for_predictor = full_hidden.to(dtype=next(self.stellar_predictor.parameters()).dtype)
            stellar_preds = self.stellar_predictor(h_for_predictor)  # Use combined hidden states
            outputs['stellar_predictions'] = stellar_preds
        
        return outputs

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
            h_for_predictor = h.to(dtype=next(self.stellar_predictor.parameters()).dtype)
            
            # Check for NaN in hidden states before prediction
            if torch.isnan(h_for_predictor).any() or torch.isinf(h_for_predictor).any():
                print("Warning: NaN/inf detected in hidden states for stellar prediction")
                # Create dummy predictions to avoid breaking the forward pass
                stellar_preds = {param: torch.zeros(h.size(0), device=h.device, dtype=h.dtype) 
                               for param in self.stellar_predictor.stellar_params}
            else:
                stellar_preds = self.stellar_predictor(h_for_predictor)  # Use hidden states
                
                # Check for NaN in stellar predictions
                for param, pred in stellar_preds.items():
                    if torch.isnan(pred).any() or torch.isinf(pred).any():
                        print(f"Warning: NaN/inf detected in stellar prediction for {param}")
                        stellar_preds[param] = torch.zeros_like(pred)
            
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
            # Ensure hidden states match predictor dtype
            h_for_predictor = h.to(dtype=next(self.stellar_predictor.parameters()).dtype)
            
            # Check for NaN in hidden states before prediction
            if torch.isnan(h_for_predictor).any() or torch.isinf(h_for_predictor).any():
                print("Warning: NaN/inf detected in hidden states for stellar prediction")
                # Create dummy predictions to avoid breaking the forward pass
                stellar_preds = {param: torch.zeros(h.size(0), device=h.device, dtype=h.dtype) 
                               for param in self.stellar_predictor.stellar_params}
            else:
                stellar_preds = self.stellar_predictor(h_for_predictor)  # Use hidden states
                
                # Check for NaN in stellar predictions
                for param, pred in stellar_preds.items():
                    if torch.isnan(pred).any() or torch.isinf(pred).any():
                        print(f"Warning: NaN/inf detected in stellar prediction for {param}")
                        stellar_preds[param] = torch.zeros_like(pred)
            
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
        
        # Handle different data structures for single vs two-star modes
        # For mixed datasets, determine the mode for this specific sample
        if self.mode == "combined":
            # Check if this sample is single or comparative
            if 'mode' in batch_data and batch_data['mode']:
                current_mode = batch_data['mode'][batch_idx]
            else:
                # Fallback: check if it has comparative features
                current_mode = "two_star" if ('masked_spectra_a' in batch_data and batch_data['masked_spectra_a'] is not None) else "single_star"
        else:
            current_mode = self.mode
            
        if current_mode == "two_star":
            star_a_features = batch_data['masked_spectra_a'][batch_idx:batch_idx+1].to(device)
            star_b_features = batch_data['masked_spectra_b'][batch_idx:batch_idx+1].to(device)
            star_a_indices = batch_data['star_a_feature_indices'][batch_idx:batch_idx+1].to(device)
            star_b_indices = batch_data['star_b_feature_indices'][batch_idx:batch_idx+1].to(device)
            answer_start_idx = batch_data.get('answer_start_indices', [input_ids.shape[1]])[batch_idx]
            if isinstance(answer_start_idx, torch.Tensor):
                answer_start_idx = answer_start_idx.item()
        else:
            input_spectra = batch_data['masked_spectra'][batch_idx:batch_idx+1].to(device)
            feature_start_idx = batch_data['feature_start_indices'][batch_idx].to(device)
            answer_start_idx = batch_data['answer_start_indices'][batch_idx].item()

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
                # Temporarily set mode for forward pass
                original_mode = self.mode
                self.mode = "two_star"
                out = self._forward_no_cache_two_star(prompt, star_a_features, star_b_features,
                                                      star_a_indices, star_b_indices)
                self.mode = original_mode
            else:
                # Temporarily set mode for forward pass
                original_mode = self.mode
                self.mode = "single_star"
                out = self._forward_no_cache(prompt, features_vec, feature_start_idx)
                self.mode = original_mode
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
