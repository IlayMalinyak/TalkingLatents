import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.perceiver_decoder import SpectralDecoder
from nn.perceiver_io import PerceiverEncoder, CrossAttentionBlock
from nn.positional_encodings import build_position_indices
from nn.DualFormer.dual_attention import DualFormerForJointEmbedding
from llama3.llama.model import precompute_freqs_cis


class LateFusionModel(nn.Module):
    """
    Perceiver IO style late-fusion network with optional CLIP alignment.

    Mode A (joint fusion):
        - Concatenate text and spectral tokens → Perceiver encoder
        - Decode spectra + expose LLM prefix embeddings

    Mode B (contrastive):
        - Encode text-only and spectra-only streams → pooled embeddings
        - InfoNCE loss for retrieval alignment
    """

    def __init__(self, llm_model, spectral_model, config: Dict[str, Any]):
        super().__init__()
        self.llm_model = llm_model
        
        # Explicitly enable gradient checkpointing if requested
        if config.get("gradient_checkpointing", True):
            # Try HF method
            if hasattr(self.llm_model, "gradient_checkpointing_enable"):
                self.llm_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                if hasattr(self.llm_model, "config"):
                    self.llm_model.config.use_cache = False
            
            # If wrapped in DDP or other wrappers, try to access base model
            elif hasattr(self.llm_model, "module"):
                if hasattr(self.llm_model.module, "gradient_checkpointing_enable"):
                    self.llm_model.module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    if hasattr(self.llm_model.module, "config"):
                         self.llm_model.module.config.use_cache = False

        self.spectral_model = spectral_model
        self.config = config

        self.d_model = config["d_model"]
        self.d_llm = config["d_llm"]
        self.d_align = config["d_align"]
        self.num_latents = config["M_latent"]
        self.spectral_target_dim = config.get("spectral_target_dim", config["spectral_feature_dim"])
        self.d_spec = config["spectral_feature_dim"]
        self.spectral_token_count = int(config.get("spectral_token_count", 1))
        self.llm_prefix_len = int(config.get("llm_prefix_len", 0)) # 0 means use all latents
        self.loss_weights = config.get(
            "loss_weights", {"reconstruction": 1.0, "contrastive": 1.0, "ce": 1.0}
        )
        # Cycle-fusion CE branch enable flag
        self.enable_cycle_ce = bool(config.get("enable_cycle_ce", False))
        self.modality_dropout_cfg = config.get(
            "modality_dropout",
            {"enabled": False, "drop_text_prob": 0.0, "drop_spectra_prob": 0.0},
        )

        self._freeze_backbones()

        self.text_adapter = self._build_adapter(config["llm_hidden_dim"], self.d_model)
        self.spectral_adapter = self._build_adapter(
            config["spectral_feature_dim"], self.d_model
        )
        self.spectral_feat_norm = nn.LayerNorm(self.d_spec)
        if self.spectral_token_count > 1:
            self.spectral_token_proj = nn.Sequential(
                nn.LayerNorm(config["spectral_feature_dim"]),
                nn.Linear(
                    config["spectral_feature_dim"],
                    self.spectral_token_count * config["spectral_feature_dim"],
                ),
            )
        else:
            self.spectral_token_proj = None
        self.modality_embeddings = nn.Parameter(torch.randn(2, self.d_model))

        self.perceiver = PerceiverEncoder(
            d_model=self.d_model,
            num_latents=self.num_latents,
            num_latent_blocks=config["num_latent_blocks"],
            num_heads=config["num_heads"],
            ffn_mult=config.get("ffn_mult", 4),
            dropout=config.get("dropout", 0.1),
            attn_dropout=config.get("attn_dropout", 0.0),
            rope_theta=config.get("rope_theta", 10000.0),
        )

        # Unified Perceiver Decoder (Cross-Attention)
        self.perceiver_decoder = CrossAttentionBlock(
            dim=self.d_model,
            num_heads=config.get("num_heads", 8),
            ffn_mult=config.get("ffn_mult", 4),
            dropout=config.get("dropout", 0.1),
            attn_dropout=config.get("attn_dropout", 0.0),
            rope_theta=config.get("rope_theta", 10000.0),
        )

        # Learnable Queries
        self.spectral_queries = nn.Parameter(torch.randn(1, self.spectral_target_dim, self.d_model))
        self.prefix_queries = nn.Parameter(torch.randn(1, self.llm_prefix_len, self.d_model))
        
        # Feature Reconstruction Output Head (Latents -> Features)
        # Input: Pooled Feature Latents [B, d_model]
        # Output: Feature Vector [B, d_spec]
        self.feature_reconstruction_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_spec)
        )
        
        # LLM Prefix: d_model -> d_llm
        self.llm_output_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.d_model * 2, self.d_llm),
            nn.LayerNorm(self.d_llm),
        )

        # Description Reconstruction Head (Latents -> LLM Prefix Embeddings)
        # Input: Pooled LLM Latents [B, d_model]
        # Output: LLM Prefix Embeddings [B, d_llm]
        self.desc_reconstruction_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.d_model * 2, self.d_llm),
            nn.LayerNorm(self.d_llm),
        )

        # Positional Indices for queries (fixed/cached)
        total_queries = self.spectral_target_dim + self.llm_prefix_len
        self.decoder_query_positions = nn.Parameter(
            build_position_indices(total_queries), requires_grad=False
        )

        # Output Heads
        # Spectrum: d_model -> 1 (or 1D target length if queries handle spatial)
        # We assume queries correspond to wavelength bins, so 1 scalar per query
        # Raw Spectra Reconstruction Head: d_model -> raw_spectra_dim
        # Reconstructs the raw input spectra (intensity per bin)
        self.raw_spectra_dim = config.get("raw_spectra_dim", 4096)
        self.raw_spectra_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.d_model, self.raw_spectra_dim)
        )
        
        # Stellar parameter regression & injection
        num_stellar_params = config.get("num_stellar_params", 3)
        self.stellar_regressor = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.d_model // 2, num_stellar_params),
        )
        
        # Project predicted params back to d_model space
        self.param_projector = nn.Sequential(
            nn.Linear(num_stellar_params, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # self.alignment_pool = nn.Sequential(
        #     nn.LayerNorm(self.d_model),
        #     nn.Linear(self.d_model, projector_hidden),
        #     nn.GELU(),
        #     nn.Dropout(projector_dropout),
        # )
        # self.alignment_head = nn.Sequential(
        #     nn.LayerNorm(projector_hidden),
        #     nn.Linear(projector_hidden, self.d_align),
        # )
        # self.alignment_dropout = nn.Dropout(projector_dropout)

        # temperature_init = config.get("temperature_init", 0.07)
        # if config.get("learn_temperature", True):
        #     self.temperature = nn.Parameter(torch.tensor(temperature_init))
        # else:
        #     self.register_buffer(
        #         "temperature", torch.tensor(temperature_init), persistent=False
        #     )


        self._ensure_perceiver_fp32()

    def _llama_forward_from_embeds(self, inputs_embeds: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable LLaMA forward from precomputed input embeddings.
        Returns (hidden, logits)."""
        base = getattr(self.llm_model, "base_model", self.llm_model)
        hidden = inputs_embeds
        seqlen = hidden.size(1)
        device = hidden.device

        # Causal mask
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1).to(hidden.dtype)

        # Combine with padding mask if provided
        if attn_mask is not None:
            # attn_mask is (B, T) with 1=valid, 0=pad
            # We want 0 for valid, -inf for pad
            pad_mask = torch.zeros_like(attn_mask, dtype=hidden.dtype)
            pad_mask.masked_fill_(attn_mask == 0, float("-inf"))
            # Reshape to (B, 1, 1, T) to broadcast over heads and query positions
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(1)
            
            if mask is None:
                mask = pad_mask
            else:
                # Combine causal (T, T) and padding (B, 1, 1, T)
                # Result: (B, 1, T, T)
                mask = mask.unsqueeze(0).unsqueeze(0) + pad_mask
        # Rotary frequencies (ensure capacity)
        start_pos = 0
        if hasattr(base, 'params') and hasattr(base.params, 'dim') and hasattr(base.params, 'n_heads'):
            head_dim = base.params.dim // base.params.n_heads
        else:
            # Fallback: infer from existing cache
            head_dim = base.freqs_cis.shape[1]
        if seqlen > base.freqs_cis.shape[0]:
            new_len = max(seqlen, base.freqs_cis.shape[0] * 2)
            base.freqs_cis = precompute_freqs_cis(head_dim, new_len, getattr(base.params, 'rope_theta', 10000.0)).to(base.freqs_cis.device)
        base.freqs_cis = base.freqs_cis.to(device)
        freqs_cis = base.freqs_cis[start_pos : start_pos + seqlen]
        # Ensure KV-cache capacity matches (batch, seqlen) - Not needed if use_cache=False
        
        # Transformer layers
        for layer in base.layers:
            # We bypass cache for training to avoid memory leaks/accumulating history in buffers
            hidden = layer(hidden, start_pos, freqs_cis, mask, use_cache=False)
        hidden = base.norm(hidden)
        logits = base.output(hidden).float()
        # Detach KV caches to avoid holding computation graphs across steps
        for layer in base.layers:
            attn = getattr(layer, 'attention', None)
            if attn is None:
                continue
            if hasattr(attn, 'cache_k') and isinstance(attn.cache_k, torch.Tensor):
                attn.cache_k = attn.cache_k.detach()
            if hasattr(attn, 'cache_v') and isinstance(attn.cache_v, torch.Tensor):
                attn.cache_v = attn.cache_v.detach()
        return hidden, logits

    def _hf_forward_from_embeds(self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable HF forward from embeddings; returns (hidden, logits)."""
        model = getattr(self.llm_model, "base_model", self.llm_model)
        kwargs = {
            "inputs_embeds": inputs_embeds,
            "use_cache": False,
            "output_hidden_states": True,
        }
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        outputs = model(**kwargs)
        hidden = getattr(outputs, "last_hidden_state", outputs[0])
        logits = getattr(outputs, "logits", None)
        if logits is None:
            lm_head = getattr(model, "lm_head", None)
            if lm_head is None:
                raise ValueError("HF model output lacks logits and no lm_head is available.")
            logits = lm_head(hidden)
        return hidden, logits.float()

    def _ensure_perceiver_fp32(self) -> None:
        """Keep Perceiver stack and adapters in float32 for stability."""
        modules_fp32 = [
            self.text_adapter,
            self.spectral_adapter,
            self.spectral_token_proj,
            self.perceiver,
            self.perceiver_decoder,
            self.raw_spectra_head,
            self.llm_output_head,
            self.feature_reconstruction_head, # Added
            self.desc_reconstruction_head, # Added
            self.stellar_regressor,
            self.param_projector,
        ]
        for module in modules_fp32:
            if module is None:
                continue
            module.float()

    def _apply_modality_dropout(
        self,
        text_embeddings: torch.Tensor,
        text_mask: torch.Tensor,
        spectral_embeddings: torch.Tensor,
        spectral_mask: torch.Tensor,
    ):
        cfg = self.modality_dropout_cfg or {}
        if not (self.training and cfg.get("enabled", False)):
            return text_embeddings, text_mask, spectral_embeddings, spectral_mask, "none"

        device = text_embeddings.device
        drop_text_prob = float(cfg.get("drop_text_prob", 0.0))
        drop_spec_prob = float(cfg.get("drop_spectra_prob", 0.0))
    
        drop_text = torch.rand(1, device=device).item() < drop_text_prob
        drop_spec = torch.rand(1, device=device).item() < drop_spec_prob

        if drop_text and drop_spec:
            if torch.rand(1, device=device).item() < 0.5:
                drop_text = False
            else:
                drop_spec = False

        dropped = "none"
        if drop_text:
            text_embeddings = torch.zeros_like(text_embeddings)
            text_mask = torch.zeros_like(text_mask, dtype=torch.bool)
            dropped = "text"
        if drop_spec:
            spectral_embeddings = torch.zeros_like(spectral_embeddings)
            spectral_mask = torch.zeros_like(spectral_mask, dtype=torch.bool)
            dropped = "spectra"

        return text_embeddings, text_mask, spectral_embeddings, spectral_mask, dropped

    @staticmethod
    def _safe_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(tensor):
            return tensor
        tensor = torch.nan_to_num(tensor)
        if tensor.dtype in (torch.float16, torch.bfloat16):
            tensor = tensor.float()
        return tensor

    def _freeze_backbones(self) -> None:
        if self.config.get("freeze_llm", True):
            for param in self.llm_model.parameters():
                param.requires_grad = False
        if self.config.get("freeze_spectral", True) and self.spectral_model is not None:
            for param in self.spectral_model.parameters():
                param.requires_grad = False

    @staticmethod
    def _build_adapter(input_dim: int, output_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def _encode_text_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return final transformer hidden states from the LLM backbone."""
        model = self.llm_model.base_model if hasattr(self.llm_model, "base_model") else self.llm_model
        requires_grad = not self.config.get("freeze_llm", True)
        ctx = torch.enable_grad() if requires_grad else torch.no_grad()
        start_pos = 0
        seqlen = input_ids.size(1)
        with ctx:
            hidden = model.tok_embeddings(input_ids)
            model.freqs_cis = model.freqs_cis.to(hidden.device)
            freqs_cis = model.freqs_cis[start_pos : start_pos + seqlen]

            mask = None
            if seqlen > 1:
                mask = torch.full((seqlen, seqlen), float("-inf"), device=input_ids.device)
                mask = torch.triu(mask, diagonal=1).to(hidden.dtype)

            # Ensure KV-cache capacity matches (batch, seqlen)
            # This is critical because LLaMA's forward pass writes to cache
            bsz = input_ids.size(0)
            if hasattr(model, 'layers'):
                for layer in model.layers:
                    attn = getattr(layer, 'attention', None)
                    if attn is None:
                        continue
                    cache_k = getattr(attn, 'cache_k', None)
                    cache_v = getattr(attn, 'cache_v', None)
                    if cache_k is None or cache_v is None:
                        continue
                    
                    need_b = bsz
                    need_t = seqlen
                    cur_b = cache_k.shape[0]
                    cur_t = cache_k.shape[1]
                    
                    if need_b > cur_b or need_t > cur_t:
                        # Resize if needed (expand dimensions)
                        new_b = max(need_b, cur_b)
                        new_t = max(need_t, cur_t)
                        new_shape = (new_b, new_t, cache_k.shape[2], cache_k.shape[3])
                        
                        new_k = cache_k.new_zeros(new_shape)
                        new_v = cache_v.new_zeros(new_shape)
                        
                        # Copy existing content
                        if cur_b > 0 and cur_t > 0:
                            new_k[:cur_b, :cur_t] = cache_k
                            new_v[:cur_b, :cur_t] = cache_v
                            
                        attn.cache_k = new_k
                        attn.cache_v = new_v

            for layer in model.layers:
                hidden = layer(hidden, start_pos, freqs_cis, mask)
            hidden = model.norm(hidden)

            # Detach KV caches to avoid holding computation graphs across steps
            for layer in model.layers:
                attn = getattr(layer, 'attention', None)
                if attn is None:
                    continue
                if hasattr(attn, 'cache_k') and isinstance(attn.cache_k, torch.Tensor):
                    attn.cache_k = attn.cache_k.detach()
                if hasattr(attn, 'cache_v') and isinstance(attn.cache_v, torch.Tensor):
                    attn.cache_v = attn.cache_v.detach()
        return hidden

    def _get_spectral_input(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        for key in ("spectral_data", "masked_spectra", "features"):
            tensor = batch.get(key, None)
            if tensor is not None:
                return tensor
        raise ValueError("Batch is missing spectral inputs ('spectral_data' or similar key).")

    def extract_spectral_features(self, spectral_data: torch.Tensor) -> torch.Tensor:
        if self.spectral_model is None:
            print("spectral model is None! using input as features")
            features = spectral_data
        else:
            ctx = torch.no_grad() if self.config.get("freeze_spectral", True) else torch.enable_grad()
            with ctx:
                # Force float32 for spectral model to avoid NaNs
                with torch.cuda.amp.autocast(enabled=False):
                    outputs = self.spectral_model(spectral_data.float())
            # Expect tuple (logits, aux, latent_features)
            if isinstance(outputs, tuple):
                features = outputs[-1]
            else:
                features = outputs

        if features.dim() == 2:
            features = features.unsqueeze(1)
            
        features = self._safe_tensor(features)
        
        # Apply normalization to the features
        features = self.spectral_feat_norm(features)
        
        return features

    def _expand_spectral_tokens(self, spectral_features: torch.Tensor) -> torch.Tensor:
        if self.spectral_token_proj is None:
            return spectral_features
        seq_len = spectral_features.size(1)
        if seq_len > 1:
            return spectral_features
        pooled = spectral_features.mean(dim=1)
        expanded = self.spectral_token_proj(pooled)
        expanded = expanded.view(spectral_features.size(0), self.spectral_token_count, -1)
        return expanded

    def _prepare_spectral_targets(self, spectral_features: torch.Tensor) -> torch.Tensor:
        """
        Build regression targets in the spectral latent space (spectral_features).
        Spectral features may contain a sequence dimension; we pool to [B, feature_dim].
        """
        if spectral_features.dim() == 3:
            targets = spectral_features.mean(dim=1)
        elif spectral_features.dim() == 2:
            targets = spectral_features
        else:
            raise ValueError("Spectral features must have rank 2 or 3.")

        targets = targets.float()
        if targets.size(1) != self.spectral_target_dim:
            raise ValueError(
                f"Spectral target dim mismatch: expected {self.spectral_target_dim}, "
                f"got {targets.size(1)}. Set 'spectral_target_dim' in config if needed."
            )
        return targets

    def _build_joint_sequence(
        self,
        text_embeddings: torch.Tensor,
        spectral_embeddings: torch.Tensor,
        text_mask: torch.Tensor,
        spectral_mask: torch.Tensor,
    ):
        seq_text = text_embeddings.size(1)
        seq_spec = spectral_embeddings.size(1)
        device = text_embeddings.device

        text_positions = torch.arange(seq_text, device=device, dtype=torch.long)
        spec_positions = torch.arange(seq_spec, device=device, dtype=torch.long) + seq_text
        joint_positions = torch.cat([text_positions, spec_positions], dim=0)

        inputs = torch.cat([text_embeddings, spectral_embeddings], dim=1)
        mask = torch.cat([text_mask, spectral_mask], dim=1)
        return inputs, mask, joint_positions

    def _pool_alignment(self, latents: torch.Tensor, branch: str) -> torch.Tensor:
        pooled = latents.mean(dim=1)
        pooled = self._safe_tensor(self.alignment_pool(pooled))
        proj = self._safe_tensor(self.alignment_head(pooled))
        proj = self.alignment_dropout(proj)
        proj = F.normalize(proj, p=2, dim=-1)
        if self.stop_gradient_branch == branch and self.training:
            proj = proj.detach()
        return proj


    def _compute_cycle_ce_loss(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute cycle-consistency cross-entropy loss."""
        if not (self.enable_cycle_ce and ("followup_input_ids" in batch) and ("followup_labels" in batch)):
            return torch.tensor(0.0, device=device)

        full_ids = batch["followup_input_ids"].to(device)
        full_labels = batch["followup_labels"].to(device)
        
        prefix = outputs.get("prefix_embeddings", None)
        if prefix is None:
            return torch.tensor(0.0, device=device)
        
        # Token embeddings from LLM
        base_llm = getattr(self.llm_model, "base_model", self.llm_model)
        tok_embeds = base_llm.tok_embeddings(full_ids)
        
        # Concatenate prefix embeddings
        inputs_embeds = torch.cat([prefix.to(device, dtype=tok_embeds.dtype), tok_embeds], dim=1)
        
        # Build attention mask: 1s for prefix + non-pad for ids
        pad_id = 0
        if hasattr(base_llm, 'config') and hasattr(base_llm.config, 'pad_token_id') and base_llm.config.pad_token_id is not None:
            pad_id = base_llm.config.pad_token_id
        
        attn_mask_ids = (full_ids != pad_id).long()
        prefix_mask = torch.ones(prefix.size(0), prefix.size(1), device=device, dtype=torch.long)
        attention_mask = torch.cat([prefix_mask, attn_mask_ids], dim=1)
        
        # Choose backend path
        if hasattr(base_llm, "get_input_embeddings"):
            hidden2, logits2 = self._hf_forward_from_embeds(inputs_embeds, attention_mask)
        else:
            hidden2, logits2 = self._llama_forward_from_embeds(inputs_embeds, attention_mask)
        
        # Build labels: prepend -100 for prefix
        B, Ltot, V = logits2.shape
        prefix_len = prefix.size(1)
        
        # Prepend -100s for prefix to the labels
        prefix_labels = torch.full((B, prefix_len), -100, device=device, dtype=torch.long)
        labels = torch.cat([prefix_labels, full_labels], dim=1)
        
        # Sanitize labels: mask invalid or padding ids to -100
        # Resolve model vocab size
        vocab_size = None
        base_params = getattr(base_llm, 'params', None)
        if base_params is not None:
            vocab_size = getattr(base_params, 'vocab_size', None)
        if vocab_size is None and hasattr(base_llm, 'config'):
            vocab_size = getattr(base_llm.config, 'vocab_size', None)
        if vocab_size is None:
            vocab_size = V
            
        invalid = (labels < 0) | (labels >= vocab_size)
        if pad_id is not None:
            invalid = invalid | (labels == pad_id)
        labels[invalid] = -100
        
        # Standard shifted CE
        shift_logits = logits2[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1), ignore_index=-100)
        return ce_loss

    def _compute_desc_reconstruction_loss(
        self,
        batch: Dict[str, torch.Tensor],
        desc_prefix_emb: torch.Tensor, # [B, 1, d_llm]
        device: torch.device,
    ) -> torch.Tensor:
        """Compute cross-entropy loss for description reconstruction."""
        if not ("followup_input_ids" in batch and "followup_labels" in batch):
            return torch.tensor(0.0, device=device)

        full_ids = batch["followup_input_ids"].to(device)
        full_labels = batch["followup_labels"].to(device)
        
        # Token embeddings from LLM
        base_llm = getattr(self.llm_model, "base_model", self.llm_model)
        tok_embeds = base_llm.tok_embeddings(full_ids) # [B, L, d_llm]
        
        # Concatenate prefix embeddings
        # desc_prefix_emb is [B, 1, d_llm], tok_embeds is [B, L, d_llm]
        # DEBUG: Check stats of prefix embeddings
        if torch.isnan(desc_prefix_emb).any() or torch.isinf(desc_prefix_emb).any() or desc_prefix_emb.abs().max() > 100:
             # Use a cleaner print to restrict spam
             if not hasattr(self, "_last_printed_stats_step") or self._last_printed_stats_step != batch.get("step_count", -1):
                print(f"DEBUG: desc_prefix_emb stats | min={desc_prefix_emb.min().item():.2f} | max={desc_prefix_emb.max().item():.2f} | mean={desc_prefix_emb.mean().item():.2f}")
                if torch.isnan(desc_prefix_emb).any():
                    print("DEBUG: desc_prefix_emb contains NaNs!")
        
        inputs_embeds = torch.cat([desc_prefix_emb.to(device, dtype=tok_embeds.dtype), tok_embeds], dim=1)
        
        # Build attention mask: 1s for prefix + non-pad for ids
        pad_id = 0
        if hasattr(base_llm, 'config') and hasattr(base_llm.config, 'pad_token_id') and base_llm.config.pad_token_id is not None:
            pad_id = base_llm.config.pad_token_id
        
        attn_mask_ids = (full_ids != pad_id).long()
        prefix_mask = torch.ones(desc_prefix_emb.size(0), desc_prefix_emb.size(1), device=device, dtype=torch.long)
        attention_mask = torch.cat([prefix_mask, attn_mask_ids], dim=1)
        
        # Choose backend path
        if hasattr(base_llm, "get_input_embeddings"):
            hidden2, logits2 = self._hf_forward_from_embeds(inputs_embeds, attention_mask)
        else:
            hidden2, logits2 = self._llama_forward_from_embeds(inputs_embeds, attention_mask)
        
        # Build labels: prepend -100 for prefix
        B, Ltot, V = logits2.shape
        prefix_len = desc_prefix_emb.size(1)
        
        # Prepend -100s for prefix to the labels
        prefix_labels = torch.full((B, prefix_len), -100, device=device, dtype=torch.long)
        labels = torch.cat([prefix_labels, full_labels], dim=1)
        
        # Sanitize labels: mask invalid or padding ids to -100
        vocab_size = None
        base_params = getattr(base_llm, 'params', None)
        if base_params is not None:
            vocab_size = getattr(base_params, 'vocab_size', None)
        if vocab_size is None and hasattr(base_llm, 'config'):
            vocab_size = getattr(base_llm.config, 'vocab_size', None)
        if vocab_size is None:
            vocab_size = V
            
        invalid = (labels < 0) | (labels >= vocab_size)
        if pad_id is not None:
            invalid = invalid | (labels == pad_id)
        labels[invalid] = -100
        
        # Standard shifted CE
        shift_logits = logits2[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1), ignore_index=-100)
        return ce_loss


    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)
        text_mask = attention_mask.bool()

        text_embeddings = self._encode_text_inputs(input_ids).to(device)
        text_embeddings = self._safe_tensor(text_embeddings)
        text_embeddings = self.text_adapter(text_embeddings)
        text_embeddings = self._safe_tensor(text_embeddings)
        text_embeddings = (
            text_embeddings
            + self.modality_embeddings[0].view(1, 1, -1)
        )

        spectral_input = self._get_spectral_input(batch).to(device) # raw spectra
        spectral_input = self._safe_tensor(spectral_input)
        spectral_features = self.extract_spectral_features(spectral_input.to(device)) # B,1,d_spec
        spectral_tokens = self._expand_spectral_tokens(spectral_features) # B,num_tokens,d_spec
        spectral_tokens = self._safe_tensor(spectral_tokens)
        spectral_embeddings = self.spectral_adapter(spectral_tokens) # B,num_tokens,d_latent
        spectral_embeddings = self._safe_tensor(spectral_embeddings)
        spectral_embeddings = (
            spectral_embeddings
            + self.modality_embeddings[1].view(1, 1, -1)
        )
        spectral_mask = torch.ones(
            spectral_embeddings.size(0),
            spectral_embeddings.size(1),
            device=device,
            dtype=torch.bool,
        )

        features_targets = spectral_features # B, d_spec

        outputs: Dict[str, torch.Tensor] = {}
        total_loss = torch.zeros(1, device=device)
        dropped_flag = "none"

        if self.config.get("use_joint_encode", True):
            fusion_text = text_embeddings
            fusion_text_mask = text_mask
            fusion_spec = spectral_embeddings
            fusion_spec_mask = spectral_mask

            (
                fusion_text,
                fusion_text_mask,
                fusion_spec,
                fusion_spec_mask,
                dropped_flag,
            ) = self._apply_modality_dropout(
                fusion_text, fusion_text_mask, fusion_spec, fusion_spec_mask
            )
            fusion_inputs, fusion_mask, fusion_positions = self._build_joint_sequence(
                fusion_text, fusion_spec, fusion_text_mask, fusion_spec_mask
            )
            
            # 1. ENCODE
            latents = self.perceiver(fusion_inputs, fusion_mask, fusion_positions)
            latents = self._safe_tensor(latents)
            outputs["latents"] = latents

            # 2. REGRESS (The "Side Quest")
            # Splitting Latents per User Request:
            # Token 0: Params Regression
            # Tokens 1 to 1+prefix_len: LLM Prefix
            # Tokens 1+prefix_len to End: Features
            
            # Param Token
            latents_reg = latents[:, 0]  # [B, d_model]
            stellar_pred = self.stellar_regressor(latents_reg)  # [B, 3]
            outputs["stellar_prediction"] = stellar_pred

            # LLM Split
            # The splitting of lm tokens is self.llm_prefix_len as per user request
            # latents_llm = latents[:, 1 : 1 + self.llm_prefix_len]
            # latents_feat = latents[:, 1 + self.llm_prefix_len :]

            # DEBUG: Check Perceiver output stats
            # print(f"DEBUG: Latents LLM Stats | Mean: {latents_llm.mean().item():.3f} | Std: {latents_llm.std().item():.3f} | Min: {latents_llm.min().item():.3f} | Max: {latents_llm.max().item():.3f}")
            # if torch.isnan(latents_llm).any():
            #     print("DEBUG: CRITICAL - NaNs in Latents LLM!")

            # --- B. Description Reconstruction Head ---
            # Generate LLM prefix [B, prefix_len, d_llm] from Latents [B, prefix_len, d_model]
            
            # desc_prefix_emb = self.desc_reconstruction_head(latents_llm) # [B, prefix_len, d_llm]
            
            # DEBUG: Check Projected stats
            # print(f"DEBUG: Prefix Emb Stats | Mean: {desc_prefix_emb.mean().item():.3f} | Std: {desc_prefix_emb.std().item():.3f} | Min: {desc_prefix_emb.min().item():.3f} | Max: {desc_prefix_emb.max().item():.3f}")
            # if torch.isnan(desc_prefix_emb).any():
            #     print("DEBUG: CRITICAL - NaNs in Prefix Embeddings!")
            
            # Removed Clamp as per user request
            # desc_prefix_emb = torch.clamp(desc_prefix_emb, min=-10.0, max=10.0)

            # --- A. Feature Reconstruction Head ---
            # Pool feature latents: [B, N_feat, D] -> [B, D]
            # feat_rec_loss = torch.tensor(0.0, device=device)
            # if latents_feat.size(1) > 0:
            #     pool_feat = latents_feat.mean(dim=1)
            #     rec_features = self.feature_reconstruction_head(pool_feat)
            #     outputs["reconstructed_features"] = rec_features
                
                # Loss Logic: Train if Spectra Dropped (or Both present)
                # If dropped_flag == 'spectra': Input was text only -> Reconstruct features from text
                # If dropped_flag == 'text': Input was spectra only -> Don't train feature reconstruction? (Masked)
                # If dropped_flag == 'none': Both present -> Train? (User said "opposite dropout")
                # Strict interpretation: 
                # Text/None dropped (Spec input available) -> Don't train or Weight=0?
                # Spectra dropped (Text input available) -> Train.
                
                # Check user request: "when text is dropped the features loss is masked and vice versa"
                # Text Dropped (Input=Spectra) -> Features Loss Masked (Don't train)
                # Spectra Dropped (Input=Text) -> Features Loss Active (Train)
                
                # Wait, "features loss masked" means we DON'T compute it?
                # If Input=Spectra, we have perfect features input. Predicting them is trivial auto-encoding.
                # If Input=Text, we don't have features. Predicting them is Cross-Modal Generation.
                # So we train when Spectra is Dropped.
            
                # Fix broadcasting shape mismatch: features_targets is [B, 1, D], rec_features is [B, D]
            target_squeezed = features_targets.squeeze(1) if features_targets.dim() == 3 and features_targets.size(1) == 1 else features_targets
                
            # feat_rec_loss = F.mse_loss(rec_features, target_squeezed, reduction="mean")
            feat_rec_loss =  torch.tensor([0.0], device=target_squeezed.device)
            outputs["feat_rec_loss"] = feat_rec_loss
                
            #     if dropped_flag == "spectra":
            #          feat_rec_loss = raw_feat_loss
            #     elif dropped_flag == "none":
            #          # Optional: Train on full data too?
            #          feat_rec_loss = raw_feat_loss
            #     else: 
            #          # Text dropped (Input=Spec). Trivial task? Or strictly masked?
            #          # "when text is dropped the features loss is masked"
            #          feat_rec_loss = torch.tensor(0.0, device=device)
                
                # outputs["feat_rec_loss"] = feat_rec_loss
            # else:
            #     outputs["feat_rec_loss"] = torch.tensor(0.0, device=device)

            # --- B. Description Reconstruction Head ---
            # Pool LLM latents: [B, N_llm, D] -> [B, D]
            # desc_loss = torch.tensor(0.0, device=device)
            # if latents_llm.size(1) > 0:
            # pool_llm = latents_llm.mean(dim=1)
            # Head produces [B, d_llm] -> Unsqueeze to [B, 1, d_llm]
            # desc_prefix_emb = self.desc_reconstruction_head(pool_llm).unsqueeze(1)
            # outputs["desc_prefix"] = desc_prefix_emb
            
            # desc_loss = self._compute_desc_reconstruction_loss(batch, desc_prefix_emb, device)
            desc_loss = torch.tensor([0.0], device=target_squeezed.device)
            outputs["desc_loss"] = desc_loss
            # 3. CONDITION (The Injection)
            # Project predicted params back to d_model space
            param_emb = self.param_projector(stellar_pred).unsqueeze(1) # [B, 1, d_model]
            
            # Concatenate to latents: [B, M, D] + [B, 1, D] -> [B, M+1, D]
            # These are the Keys/Values for the decoder
            cond_latents = torch.cat([latents, param_emb], dim=1)
            
            # Position indices for the condition tokens (latents + 1 param token)
            # We already have latent_positions associated with 'latents' inside the encoder,
            # but CrossAttentionBlock expects kv_positions matching cond_latents.
            # Encoder latents have positions 0..M-1. Param token gets position M.
            bsz = latents.size(0)
            M = latents.size(1)
            # M+1 positions: 0 to M
            cond_positions = torch.arange(M + 1, device=device, dtype=torch.long)
            
            # 4. DECODE
            # Prepare Queries: [B, N+K, D]
            # Spectral queries (N) + Prefix queries (K)
            full_queries = torch.cat([
                self.spectral_queries,
                self.prefix_queries
            ], dim=1).expand(bsz, -1, -1)
            
            # Query positions (cached)
            query_positions = self.decoder_query_positions.to(device)

            # Cross-Attend: Queries attend to Conditioned Latents
            decoder_out = self.perceiver_decoder(
                latents=full_queries,     # Q
                inputs=cond_latents,      # K, V
                input_mask=None,
                latent_positions=query_positions,   # Pos for Q
                input_positions=cond_positions      # Pos for K, V
            )
            decoder_out = self._safe_tensor(decoder_out)

        # --- DECODER OUTPUTS (Split) ---
        # decoder_out is [B, spectral_len + prefix_len, d_model]
        
        # 1. Raw Spectra Reconstruction
        spec_out_emb = decoder_out[:, :self.spectral_target_dim, :]
        raw_spectra_pred = self.raw_spectra_head(spec_out_emb.mean(1))
        # If spectral_input is [B, L_in] and prediction is [B, L_pred], we might have a mismatch if L_in != L_pred
        # But self.spectral_target_dim should match L_in or be interpolated?
        # Assuming for now spectral_target_dim == Input Length.
        
        recon_loss = torch.tensor(0.0, device=device)
        # Only compute if we have spectral data
        if "spectra" in batch:
             # Ensure target matches prediction shape
             target = batch["spectra"].float()
             if target.shape != raw_spectra_pred.shape:
                  # Try viewing/reshaping if number of elements match
                   if target.numel() == raw_spectra_pred.numel():
                        target = target.view_as(raw_spectra_pred)
                   else:
                        print(f"Warning: transformation mismatch. Target {target.shape}, Pred {raw_spectra_pred.shape}")
                        
             recon_loss = F.mse_loss(raw_spectra_pred, target.squeeze(), reduction="mean")
             
        outputs["spectral_reconstruction"] = raw_spectra_pred
        outputs["reconstruction_loss"] = recon_loss

        # 2. LLM Prefix (Follow-up)
        # Split second part of decoder output
        prefix_out_emb = decoder_out[:, self.spectral_target_dim:, :]
        prefix_embeddings = self.llm_output_head(prefix_out_emb)
        outputs["prefix_embeddings"] = prefix_embeddings

        # Stellar parameter loss
        stellar_params_true = batch['y_numeric'].to(device)
        # Clamp predictions to prevent extreme values before loss computation
        stellar_pred_clamped = torch.clamp(stellar_pred, min=-10.0, max=10.0)
        param_loss = F.mse_loss(stellar_pred_clamped, stellar_params_true, reduction="mean")
        # Check for NaN in param_loss before adding to total
        if torch.isnan(param_loss) or torch.isinf(param_loss):
            param_loss = torch.tensor(0.0, device=device)
        outputs["stellar_targets"] = stellar_params_true
        outputs["param_loss"] = param_loss

        # Check reconstruction loss for NaN/Inf
        if torch.isnan(recon_loss) or torch.isinf(recon_loss):
            recon_loss = torch.tensor(0.0, device=device)

        # Aggregate all losses
        # 1. Raw Spectra Reconstruction (recon_loss)
        # 2. Param Regression (param_loss)
        # 3. Feature Reconstruction (feat_rec_loss)
        # 4. Description Reconstruction (desc_loss)
        # 5. Cycle Consistency CE (ce_loss)

        total_loss = (
            self.loss_weights.get("reconstruction", 1.0) * recon_loss
            + self.loss_weights.get("param_regression", 1.0) * param_loss
            + self.loss_weights.get("feature_reconstruction", 1.0) * feat_rec_loss
            + self.loss_weights.get("description_reconstruction", 1.0) * desc_loss
        )

        # Optional: Cycle-fusion CE branch using follow-up QA
        ce_loss = self._compute_cycle_ce_loss(batch, outputs, device)
        outputs["ce_loss"] = ce_loss # Always store even if 0
        
        if ce_loss > 0:
            total_loss = total_loss + self.loss_weights.get("ce", 1.0) * ce_loss

        outputs["total_loss"] = total_loss
        outputs["dropped_modality"] = dropped_flag
        return outputs

    @torch.no_grad()
    def generate_response_from_batch(
        self,
        batch_data: Dict[str, Any],
        batch_idx: int = 0,
        tokenizer=None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[str, str, str, List[float]]:
        """
        Generate text response for a single sample in the batch.
        Returns: (generated_text, input_text, target_text, generation_log_probs)
        """
        self.eval()
        device = next(self.parameters()).device
        
        # 1. Extract inputs for this sample
        # We need to reconstruct a mini-batch of size 1 for the forward pass components
        mini_batch = {}
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                mini_batch[k] = v[batch_idx : batch_idx + 1].to(device)
            elif isinstance(v, list):
                mini_batch[k] = [v[batch_idx]]
            else:
                mini_batch[k] = v
                
        # 2. Get Prefix Embeddings
        # We can reuse the forward logic to get latents and then prefix
        input_ids = mini_batch["input_ids"]
        attention_mask = mini_batch.get("attention_mask", torch.ones_like(input_ids))
        text_mask = attention_mask.bool()

        text_embeddings = self._encode_text_inputs(input_ids)
        text_embeddings = self.text_adapter(text_embeddings)
        text_embeddings = text_embeddings + self.modality_embeddings[0].view(1, 1, -1)

        spectral_input = self._get_spectral_input(mini_batch).to(device)
        spectral_features = self.extract_spectral_features(spectral_input)
        spectral_tokens = self._expand_spectral_tokens(spectral_features)
        spectral_embeddings = self.spectral_adapter(spectral_tokens)
        spectral_embeddings = spectral_embeddings + self.modality_embeddings[1].view(1, 1, -1)
        
        spectral_mask = torch.ones(
            spectral_embeddings.size(0),
            spectral_embeddings.size(1),
            device=device,
            dtype=torch.bool,
        )

        # Joint sequence
        fusion_inputs, fusion_mask, fusion_positions = self._build_joint_sequence(
            text_embeddings, spectral_embeddings, text_mask, spectral_mask
        )
        
        # 3. Prepare Prompt (Follow-up Question)
        latents = self.perceiver(fusion_inputs, fusion_mask, fusion_positions)
        
        # Regress
        latents_reg = latents[:, 0]
        stellar_pred = self.stellar_regressor(latents_reg)
        
        # Inject
        param_emb = self.param_projector(stellar_pred).unsqueeze(1)
        cond_latents = torch.cat([latents, param_emb], dim=1)
        bsz = latents.size(0)
        M = latents.size(1)
        cond_positions = torch.arange(M + 1, device=device, dtype=torch.long)
        
        # Decode
        full_queries = torch.cat([
            self.spectral_queries,
            self.prefix_queries
        ], dim=1).expand(bsz, -1, -1)
        query_positions = self.decoder_query_positions.to(device)
        cond_mask = torch.ones(bsz, M + 1, device=device, dtype=torch.bool)
        
        decoder_out = self.perceiver_decoder(
            latents=full_queries,
            inputs=cond_latents,
            input_mask=cond_mask,
            latent_positions=query_positions,
            input_positions=cond_positions
        )
        
        # Split -> Prefix
        spectral_len = self.spectral_queries.size(1)
        prefix_out_emb = decoder_out[:, spectral_len:, :]
        prefix_embeddings = self.llm_output_head(prefix_out_emb)


        
        # 3. Prepare Prompt (Follow-up Question)
        # Check if we have followup data
        if "followup_input_ids" in mini_batch:
            f_ids = mini_batch["followup_input_ids"][0] # (L,)
            f_labels = mini_batch["followup_labels"][0] # (L,)
            
            # Find start of answer (first non-negative label)
            # If all are -100, it might be test mode (no answer), so use full sequence as prompt
            answer_starts = (f_labels != -100).nonzero()
            if answer_starts.numel() > 0:
                answer_start_idx = answer_starts[0].item()
                prompt_ids = f_ids[:answer_start_idx]
                target_ids = f_ids[answer_start_idx:]
            else:
                # No answer labels, assume full sequence is prompt (or check for padding)
                # Strip padding from the end
                pad_id = 0
                if tokenizer:
                    pad_id = getattr(tokenizer, 'pad_id', 0)
                
                non_pad = (f_ids != pad_id).nonzero()
                if non_pad.numel() > 0:
                    last_idx = non_pad[-1].item()
                    prompt_ids = f_ids[:last_idx+1]
                else:
                    prompt_ids = f_ids # All padding?
                target_ids = torch.tensor([], device=device, dtype=torch.long)
        else:
            # Fallback to main input_ids if no followups (unlikely for this task)
            prompt_ids = input_ids[0]
            target_ids = torch.tensor([], device=device, dtype=torch.long)

        # 4. Generation Loop with Hard Textual Injection
        
        # Format predicted parameters as text
        # Denormalize using dataset_interpert bounds:
        # Teff: [3000, 7500], logg: [0, 5], FeH: [-3, 0.5]
        # x_norm = (x - min) / (max - min) => x = x_norm * (max - min) + min
        
        pred_teff = stellar_pred[0, 0].item() * (7500.0 - 3000.0) + 3000.0
        pred_logg = stellar_pred[0, 1].item() * (5.0 - 0.0) + 0.0
        pred_feh = stellar_pred[0, 2].item() * (0.5 - -3.0) + -3.0
        
        pred_text_prefix = f"Predicted Params: Teff {pred_teff:.0f} K, logg {pred_logg:.2f}, [Fe/H] {pred_feh:.2f}. "
        
        # Prepare tokens for the injection
        if tokenizer:
            prefix_ids_list = tokenizer.encode(pred_text_prefix, bos=False, eos=False)
            # Depending on tokenizer, it might add bos. Strip it if we are mid-sentence or appending.
            # Llama tokenizer usually adds BOS if bos=True. Here we want raw tokens.
            prefix_ids = torch.tensor(prefix_ids_list, device=device, dtype=torch.long)
        else:
            # Fallback if no tokenizer available (shouldn't happen in valid flow)
            prefix_ids = torch.tensor([], device=device, dtype=torch.long)
            
        # Prepend to the prompt
        # prompt_ids includes BOS if original data had it. We insert AFTER BOS if present, or just prepend.
        # Assuming prompt_ids[0] is BOS (128000 for Llama 3).
        has_bos = False
        if prompt_ids.numel() > 0 and tokenizer and prompt_ids[0] == tokenizer.bos_id:
            has_bos = True
            
        if has_bos:
            current_ids = torch.cat([
                prompt_ids[:1],  # BOS
                prefix_ids,      # Injection
                prompt_ids[1:]   # Rest of prompt
            ], dim=0).unsqueeze(0)
        else:
            current_ids = torch.cat([
                prefix_ids,
                prompt_ids
            ], dim=0).unsqueeze(0)
            
        gen_log_probs = []
        
        base_llm = getattr(self.llm_model, "base_model", self.llm_model)
        
        for _ in range(max_new_tokens):
            # Embed current sequence
            current_embeddings = base_llm.tok_embeddings(current_ids)
            
            # Concatenate prefix + current
            full_embeddings = torch.cat([prefix_embeddings, current_embeddings], dim=1)
            
            # Forward pass
            # We use the full sequence forward (inefficient but safe)
            # _llama_forward_from_embeds returns (hidden, logits)
            _, logits = self._llama_forward_from_embeds(full_embeddings)
            
            # Get last token logits
            next_token_logits = logits[0, -1, :]
            
            # Sample
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top_p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
                
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
            gen_log_probs.append(torch.log(probs[next_token]).item())
            
            # Stop if EOS
            if tokenizer and next_token.item() == tokenizer.eos_id:
                break
                
        # 5. Decode
        if tokenizer:
            # Input text (Prompt)
            # Filter out negative IDs (padding/masked) before decoding to avoid OverflowError
            valid_prompt_ids = [t for t in prompt_ids.cpu().tolist() if t >= 0]
            input_text = tokenizer.decode(valid_prompt_ids)
            
            # Target text (True Answer)
            # Filter out -100 if any remain (though we sliced them out) and padding
            target_ids_list = [t for t in target_ids.cpu().tolist() if t != -100 and t != tokenizer.pad_id and t >= 0]
            target_text = tokenizer.decode(target_ids_list)
            
            # Generated text (excluding prompt)
            generated_ids_list = current_ids[0, prompt_ids.size(0):].cpu().tolist()
            # Also ensure generated ids are valid
            generated_ids_list = [t for t in generated_ids_list if t >= 0]
            generated_text = tokenizer.decode(generated_ids_list)
        else:
            input_text = str(prompt_ids.cpu().tolist())
            target_text = str(target_ids.cpu().tolist())
            generated_text = str(current_ids[0, prompt_ids.size(0):].cpu().tolist())
            
        return generated_text, input_text, target_text, gen_log_probs

    @classmethod
    def from_config_file(cls, llm_model, spectral_model, config_path: Union[str, Path]):
        config_path = Path(config_path)
        with open(config_path, "r") as handle:
            config = json.load(handle)
        return cls(llm_model, spectral_model, config)

    def save_config(self, save_path: Union[str, Path]) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as handle:
            json.dump(self.config, handle, indent=2)


def create_default_config_perceiver(
    llm_hidden_dim: int = 4096,
    spectral_feature_dim: int = 2048,
    spectral_length: int = 2048,
) -> Dict[str, Any]:
    return {
        "llm_hidden_dim": llm_hidden_dim,
        "spectral_feature_dim": spectral_feature_dim,
        "spectral_length": spectral_length,
        "spectral_target_dim": spectral_feature_dim,
        "spectral_token_count": 1,
        "d_model": 768,
        "d_align": 512,
        "d_llm": llm_hidden_dim,
        "M_latent": 32,
        "num_latent_blocks": 8,
        "num_heads": 8,
        "ffn_mult": 4,
        "dropout": 0.1,
        "attn_dropout": 0.0,
        "spectral_decode_mode": "film_1d",
        "freeze_llm": True,
        "freeze_spectral": True,
        "use_joint_encode": True,
        "enable_contrastive": True,
        "learn_temperature": True,
        "temperature_init": 0.07,
        "rope_theta": 10000.0,
        "loss_weights": {"reconstruction": 1.0, "contrastive": 1.0},
        "modality_dropout": {
            "enabled": False,
            "drop_text_prob": 0.0,
            "drop_spectra_prob": 0.0,
        },
        "contrastive": {
            "loss_type": "info_nce",
            "queue_size": 2048,
            "stop_gradient_branch": "none",
            "projection_hidden_dim": 1024,
            "projection_dropout": 0.1,
            "vicreg_invariance_weight": 25.0,
            "vicreg_variance_weight": 25.0,
            "vicreg_covariance_weight": 1.0,
            "vicreg_variance_epsilon": 1e-4,
            "dualformer": {
                "embed_dim": 768,
                "projection_dim": 512,
                "input_dim": 768,
                "output_dim": 512,
                "num_layers": 2,
                "num_heads": 8,
                "ffn_dim": 1536,
                "dropout": 0.1,
                "attention_dropout": 0.0,
                "activation": "gelu",
                "bidirectional": True,
                "norm_first": True,
                "use_positional_encoding": True,
                "max_seq_len": 2048,
                "pooling": "mean",
                "use_cls_token": False,
                "use_prediction_head": False,
                "latent_dim": 0,
                "attention_type": "cross",
                "projection_type": "transpose",
                "covariance_weight": 1.0,
                "duality_weight": 1.0,
            },
        },
    }


def create_default_config(
    llm_hidden_dim: int = 4096,
    spectral_feature_dim: int = 2048,
    spectral_length: int = 2048,
) -> Dict[str, Any]:
    """Backward-compatible alias."""
    return create_default_config_perceiver(llm_hidden_dim, spectral_feature_dim, spectral_length)


if __name__ == "__main__":
    cfg = create_default_config_perceiver()
    config_path = Path(__file__).parent / "late_fusion_config.json"
    config_path.write_text(json.dumps(cfg, indent=2))
    print(f"✓ Default Perceiver config saved to {config_path}")
