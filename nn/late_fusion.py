import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.perceiver_decoder import SpectralDecoder
from nn.perceiver_io import PerceiverEncoder
from nn.DualFormer.dual_attention import DualFormerForJointEmbedding


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
        self.spectral_model = spectral_model
        self.config = config

        self.d_model = config["d_model"]
        self.d_llm = config["d_llm"]
        self.d_align = config["d_align"]
        self.num_latents = config["M_latent"]
        self.spectral_target_dim = config.get("spectral_target_dim", config["spectral_feature_dim"])
        self.spectral_token_count = int(config.get("spectral_token_count", 1))
        self.loss_weights = config.get(
            "loss_weights", {"reconstruction": 1.0, "contrastive": 1.0}
        )
        self.modality_dropout_cfg = config.get(
            "modality_dropout",
            {"enabled": False, "drop_text_prob": 0.0, "drop_spectra_prob": 0.0},
        )
        contrastive_cfg = config.get("contrastive", {})
        self.contrastive_loss_type = (
            contrastive_cfg.get("loss_type", "info_nce") or "info_nce"
        ).lower()
        self.contrastive_queue_size = int(contrastive_cfg.get("queue_size", 0))
        self.stop_gradient_branch = (
            contrastive_cfg.get("stop_gradient_branch", "none") or "none"
        ).lower()
        projector_hidden = int(contrastive_cfg.get("projection_hidden_dim", self.d_model))
        projector_dropout = float(
            contrastive_cfg.get("projection_dropout", config.get("dropout", 0.1))
        )
        self.vicreg_lambda = float(contrastive_cfg.get("vicreg_invariance_weight", 25.0))
        self.vicreg_mu = float(contrastive_cfg.get("vicreg_variance_weight", 25.0))
        self.vicreg_nu = float(contrastive_cfg.get("vicreg_covariance_weight", 1.0))
        self.vicreg_variance_eps = float(
            contrastive_cfg.get("vicreg_variance_epsilon", 1e-4)
        )
        dualformer_cfg = contrastive_cfg.get("dualformer", {})
        self.dualformer_head = None
        self.dualformer_cov_weight = 0.0
        self.dualformer_duality_weight = 0.0
        if self.contrastive_loss_type == "dualformer":
            dualformer_embed_dim = dualformer_cfg.get("embed_dim", self.d_model)
            projection_dim = dualformer_cfg.get("projection_dim", self.d_align)
            dualformer_kwargs = {
                "input_dim": dualformer_cfg.get("input_dim", self.d_model),
                "output_dim": dualformer_cfg.get("output_dim", self.d_align),
                "num_layers": dualformer_cfg.get("num_layers", 2),
                "num_heads": dualformer_cfg.get("num_heads", config.get("num_heads", 8)),
                "ffn_dim": dualformer_cfg.get("ffn_dim", dualformer_embed_dim * 4),
                "dropout": dualformer_cfg.get("dropout", config.get("dropout", 0.1)),
                "attention_dropout": dualformer_cfg.get("attention_dropout", config.get("attn_dropout", 0.0)),
                "activation": dualformer_cfg.get("activation", "gelu"),
                "bidirectional": dualformer_cfg.get("bidirectional", True),
                "norm_first": dualformer_cfg.get("norm_first", True),
                "use_positional_encoding": dualformer_cfg.get("use_positional_encoding", True),
                "max_seq_len": dualformer_cfg.get("max_seq_len", 2048),
                "pooling": dualformer_cfg.get("pooling", "mean"),
                "use_cls_token": dualformer_cfg.get("use_cls_token", False),
                "use_prediction_head": dualformer_cfg.get("use_prediction_head", False),
                "latent_dim": dualformer_cfg.get("latent_dim", 0),
                "attention_type": dualformer_cfg.get("attention_type", "cross"),
                "projection_type": dualformer_cfg.get("projection_type", "transpose"),
            }
            self.dualformer_head = DualFormerForJointEmbedding(
                embed_dim=dualformer_embed_dim,
                projection_dim=projection_dim,
                **dualformer_kwargs,
            )
            self.dualformer_cov_weight = float(dualformer_cfg.get("covariance_weight", 1.0))
            self.dualformer_duality_weight = float(dualformer_cfg.get("duality_weight", 1.0))

        self._freeze_backbones()

        self.text_adapter = self._build_adapter(config["llm_hidden_dim"], self.d_model)
        self.spectral_adapter = self._build_adapter(
            config["spectral_feature_dim"], self.d_model
        )
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

        self.spectral_decoder = SpectralDecoder(
            mode=config.get("spectral_decode_mode", "film_1d"),
            d_model=self.d_model,
            spectral_length=self.spectral_target_dim,
            num_heads=config.get("num_heads", 8),
            dropout=config.get("dropout", 0.1),
            attn_dropout=config.get("attn_dropout", 0.0),
        )

        self.llm_adapter = nn.Linear(self.d_model, self.d_llm)
        self.alignment_pool = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, projector_hidden),
            nn.GELU(),
            nn.Dropout(projector_dropout),
        )
        self.alignment_head = nn.Sequential(
            nn.LayerNorm(projector_hidden),
            nn.Linear(projector_hidden, self.d_align),
        )
        self.alignment_dropout = nn.Dropout(projector_dropout)

        temperature_init = config.get("temperature_init", 0.07)
        if config.get("learn_temperature", True):
            self.temperature = nn.Parameter(torch.tensor(temperature_init))
        else:
            self.register_buffer(
                "temperature", torch.tensor(temperature_init), persistent=False
            )

        if self.contrastive_queue_size > 0 and self.contrastive_loss_type == "info_nce":
            self.register_buffer(
                "contrastive_queue_text",
                torch.zeros(self.contrastive_queue_size, self.d_align),
            )
            self.register_buffer(
                "contrastive_queue_spec",
                torch.zeros(self.contrastive_queue_size, self.d_align),
            )
            self.register_buffer(
                "contrastive_queue_ptr", torch.zeros(1, dtype=torch.long)
            )
            self.register_buffer(
                "contrastive_queue_filled", torch.zeros(1, dtype=torch.long)
            )
        else:
            self.contrastive_queue_text = None
            self.contrastive_queue_spec = None
            self.contrastive_queue_ptr = None
            self.contrastive_queue_filled = None

        self._ensure_perceiver_fp32()

    def _ensure_perceiver_fp32(self) -> None:
        """Keep Perceiver stack and adapters in float32 for stability."""
        modules_fp32 = [
            self.text_adapter,
            self.spectral_adapter,
            self.spectral_token_proj,
            self.perceiver,
            self.spectral_decoder,
            self.llm_adapter,
            self.alignment_pool,
            self.alignment_head,
            self.dualformer_head,
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

            for layer in model.layers:
                hidden = layer(hidden, start_pos, freqs_cis, mask)
            hidden = model.norm(hidden)
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
                outputs = self.spectral_model(spectral_data)
            # Expect tuple (logits, aux, latent_features)
            if isinstance(outputs, tuple):
                features = outputs[-1]
            else:
                features = outputs

        if features.dim() == 2:
            features = features.unsqueeze(1)
        return self._safe_tensor(features)

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

    def _get_contrastive_queue(self):
        if (
            self.contrastive_loss_type != "info_nce"
            or self.contrastive_queue_size <= 0
            or self.contrastive_queue_text is None
        ):
            return None, None
        filled = int(self.contrastive_queue_filled.item())
        if filled <= 0:
            return None, None
        return (
            self.contrastive_queue_text[:filled].detach(),
            self.contrastive_queue_spec[:filled].detach(),
        )

    def _update_contrastive_queue(self, g_text: torch.Tensor, g_spec: torch.Tensor) -> None:
        if (
            self.contrastive_loss_type != "info_nce"
            or self.contrastive_queue_size <= 0
            or self.contrastive_queue_text is None
        ):
            return
        g_text = g_text.detach()
        g_spec = g_spec.detach()
        batch_size = g_text.size(0)
        max_size = self.contrastive_queue_size
        ptr = int(self.contrastive_queue_ptr.item())

        end = ptr + batch_size
        if end <= max_size:
            self.contrastive_queue_text[ptr:end] = g_text
            self.contrastive_queue_spec[ptr:end] = g_spec
        else:
            first = max_size - ptr
            if first > 0:
                self.contrastive_queue_text[ptr:] = g_text[:first]
                self.contrastive_queue_spec[ptr:] = g_spec[:first]
            remainder = batch_size - first
            if remainder > 0:
                self.contrastive_queue_text[:remainder] = g_text[first:]
                self.contrastive_queue_spec[:remainder] = g_spec[first:]

        ptr = (ptr + batch_size) % max_size
        self.contrastive_queue_ptr[0] = ptr
        filled = int(self.contrastive_queue_filled.item())
        filled = min(max_size, filled + batch_size)
        self.contrastive_queue_filled[0] = filled

    def _vicreg_variance(self, z: torch.Tensor) -> torch.Tensor:
        if z.size(0) <= 1:
            return torch.zeros(1, device=z.device, dtype=z.dtype)
        std = torch.sqrt(z.var(dim=0) + self.vicreg_variance_eps)
        penalty = F.relu(1.0 - std)
        return penalty.mean()

    def _vicreg_covariance(self, z: torch.Tensor) -> torch.Tensor:
        if z.size(0) <= 1:
            return torch.zeros(1, device=z.device, dtype=z.dtype)
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (z.size(0) - 1)
        diag = torch.diagonal(cov)
        cov = cov - torch.diag(diag)
        return cov.pow(2).sum() / z.size(1)

    def _vicreg_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        invariance = F.mse_loss(z1, z2)
        variance = self._vicreg_variance(z1) + self._vicreg_variance(z2)
        covariance = self._vicreg_covariance(z1) + self._vicreg_covariance(z2)
        return (
            self.vicreg_lambda * invariance
            + self.vicreg_mu * variance
            + self.vicreg_nu * covariance
        )

    @staticmethod
    def _off_diagonal(tensor: torch.Tensor) -> torch.Tensor:
        dim = tensor.size(0)
        return tensor.flatten()[:-1].view(dim - 1, dim + 1)[:, 1:].reshape(-1)

    def _cov_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)
        if y.dim() > 2:
            batch_size = y.shape[0]
            y = y.view(batch_size, -1)
        batch_size, num_features = x.shape
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        denom = max(batch_size - 1, 1)
        cov_x = (x.T @ x) / denom
        cov_y = (y.T @ y) / denom
        cov_xy = (x.T @ y) / denom
        cov_loss = (
            self._off_diagonal(cov_x).pow_(2).sum().div(num_features)
            + self._off_diagonal(cov_y).pow_(2).sum().div(num_features)
            + self._off_diagonal(cov_xy).pow_(2).sum().div(num_features)
        )
        return cov_loss

    @staticmethod
    def _duality_loss(
        proj1: torch.Tensor, proj2: torch.Tensor, emb1: torch.Tensor, emb2: torch.Tensor
    ) -> torch.Tensor:
        left_side = torch.sum(proj1 * emb1, dim=1)
        right_side = torch.sum(emb2 * proj2, dim=1)
        diff = left_side - right_side
        return torch.mean(diff**2)

    def _dualformer_loss(
        self,
        proj_text: torch.Tensor,
        proj_spec: torch.Tensor,
        emb_text: torch.Tensor,
        emb_spec: torch.Tensor,
    ) -> torch.Tensor:
        cov = self._cov_loss(proj_text, proj_spec)
        duality = self._duality_loss(proj_text, proj_spec, emb_text, emb_spec)
        return self.dualformer_cov_weight * cov + self.dualformer_duality_weight * duality

    def _dualformer_encode(
        self,
        text_embeddings: torch.Tensor,
        spectral_embeddings: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        spectral_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.dualformer_head is None:
            raise ValueError("DualFormer head is not initialized.")
        dualformer_outputs = self.dualformer_head.dual_former(
            text_embeddings,
            spectral_embeddings,
            text_mask,
            spectral_mask,
            output_attentions=False,
            output_hidden_states=False,
        )
        if isinstance(dualformer_outputs, dict):
            mod_text = dualformer_outputs["emb1"]
            mod_spec = dualformer_outputs["emb2"]
            proj_text = dualformer_outputs.get("proj1")
            proj_spec = dualformer_outputs.get("proj2")
        else:
            mod_text, mod_spec = dualformer_outputs
            proj_text = None
            proj_spec = None
        if proj_text is None:
            proj_text = self.dualformer_head.proj1(mod_text)
        if proj_spec is None:
            proj_spec = self.dualformer_head.proj2(mod_spec)
        return mod_text, mod_spec, proj_text, proj_spec

    def _alignment_loss(self, g_text: torch.Tensor, g_spec: torch.Tensor) -> torch.Tensor:
        if self.contrastive_loss_type == "vicreg":
            return self._vicreg_loss(g_text, g_spec)
        batch_size = g_text.size(0)
        device = g_text.device
        queue_text, queue_spec = self._get_contrastive_queue()
        spec_bank = g_spec
        text_bank = g_text
        if queue_spec is not None:
            spec_bank = torch.cat([spec_bank, queue_spec.to(device)], dim=0)
        if queue_text is not None:
            text_bank = torch.cat([text_bank, queue_text.to(device)], dim=0)

        logits_t2s = torch.matmul(g_text, spec_bank.T) / self.temperature
        logits_s2t = torch.matmul(g_spec, text_bank.T) / self.temperature
        labels = torch.arange(batch_size, device=device)
        loss_t2s = F.cross_entropy(logits_t2s, labels)
        loss_s2t = F.cross_entropy(logits_s2t, labels)
        return 0.5 * (loss_t2s + loss_s2t)

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

        spectral_targets = self._safe_tensor(self._prepare_spectral_targets(spectral_features))

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
            latents = self.perceiver(fusion_inputs, fusion_mask, fusion_positions)
            latents = self._safe_tensor(latents)
            outputs["latents"] = latents

            spectral_pred = self._safe_tensor(self.spectral_decoder(latents))
            recon_loss = F.mse_loss(spectral_pred, spectral_targets, reduction="mean")
            outputs["spectral_reconstruction"] = spectral_pred
            outputs["spectral_targets"] = spectral_targets
            outputs["reconstruction_loss"] = recon_loss

            prefix_embeddings = self._safe_tensor(self.llm_adapter(latents))
            outputs["prefix_embeddings"] = prefix_embeddings

            total_loss = total_loss + self.loss_weights.get("reconstruction", 1.0) * recon_loss

        if self.config.get("enable_alignment", True):
            if self.contrastive_loss_type == "dualformer":
                if self.dualformer_head is None:
                    raise ValueError("DualFormer loss selected but dualformer_head is not initialized.")
                padding_text = text_mask.float() if text_mask is not None else None
                padding_spec = spectral_mask.float() if spectral_mask is not None else None
                df_emb_text, df_emb_spec, df_proj_text, df_proj_spec = self._dualformer_encode(
                    text_embeddings,
                    spectral_embeddings,
                    padding_text,
                    padding_spec,
                )
                df_emb_text = self._safe_tensor(df_emb_text)
                df_emb_spec = self._safe_tensor(df_emb_spec)
                df_proj_text = self._safe_tensor(df_proj_text)
                df_proj_spec = self._safe_tensor(df_proj_spec)
                clip_loss = self._dualformer_loss(df_proj_text, df_proj_spec, df_emb_text, df_emb_spec)
                outputs["dualformer_text_emb"] = df_emb_text
                outputs["dualformer_spec_emb"] = df_emb_spec
                outputs["dualformer_text_proj"] = df_proj_text
                outputs["dualformer_spec_proj"] = df_proj_spec
            else:
                seq_text = text_embeddings.size(1)
                seq_spec = spectral_embeddings.size(1)
                text_positions = torch.arange(seq_text, device=device, dtype=torch.long)
                spec_positions = torch.arange(seq_spec, device=device, dtype=torch.long)

                text_latents = self._safe_tensor(
                    self.perceiver(text_embeddings, text_mask, text_positions)
                )
                spec_latents = self._safe_tensor(
                    self.perceiver(spectral_embeddings, spectral_mask, spec_positions)
                )
                g_text = self._pool_alignment(text_latents, branch="text")
                g_spec = self._pool_alignment(spec_latents, branch="spectra")

                clip_loss = self._alignment_loss(g_text, g_spec)
                if self.training and self.contrastive_loss_type == "info_nce":
                    self._update_contrastive_queue(g_text, g_spec)
                outputs["g_text"] = g_text
                outputs["g_spec"] = g_spec
            outputs["contrastive_loss"] = clip_loss

            total_loss = total_loss + self.loss_weights.get("contrastive", 1.0) * clip_loss

        outputs["total_loss"] = total_loss
        outputs["dropped_modality"] = dropped_flag
        return outputs

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
