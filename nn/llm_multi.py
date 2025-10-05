import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Reuse RoPE utils from the existing implementation
from .llm import apply_rotary_emb, repeat_kv


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


class InverseProjector(nn.Module):
    """
    Map K token embeddings (d_model each) back to the fm latent space (dim=latent_dim).
    Minimal version: mean-pool over K, then a small MLP to latent_dim.
    """

    def __init__(self, d_model: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, token_block: torch.Tensor) -> torch.Tensor:
        # token_block: (B, K, d_model)
        pooled = token_block.mean(dim=1)  # (B, d_model)
        return self.mlp(pooled)           # (B, latent_dim)


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
                 use_checkpoint: bool = True, physics_dim: int = 3, neighbor_temperature: float = 1.0):
        super().__init__()
        self.base_model = base_model
        self.fm_model = fm_model
        self.embedding_dim = base_model.params.dim
        self.num_spectral_features = int(num_spectral_features)
        self.use_checkpoint = use_checkpoint
        self.physics_dim = int(max(0, physics_dim))
        self.neighbor_temperature = float(neighbor_temperature)
        self.projector = SpectralTokensProjector(
            latent_dim=latent_dim,
            d_model=self.embedding_dim,
            hidden_dim=hidden_dim,
            num_tokens=self.num_spectral_features,
        )
        # Minimal inverse mapping head (tokens -> latent)
        self.inverse_projector = InverseProjector(
            d_model=self.embedding_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
        # Text -> latent head (pool question hidden states, then map to latent_dim)
        self.text_to_latent = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.latent_to_physics = None
        if self.physics_dim > 0:
            self.latent_to_physics = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.physics_dim),
            )

    def forward(self,
                input_ids: torch.Tensor,
                input_spectra: torch.Tensor,
                special_token_positions: torch.Tensor,
                start_pos: int = 0,
                question_start_indices: Optional[torch.Tensor] = None,
                answer_start_indices: Optional[torch.Tensor] = None,
                neighbor_latents: Optional[torch.Tensor] = None,
                neighbor_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
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
        # Ensure latent features match projector dtype/device
        proj_param = next(self.projector.parameters())
        latent_features = latent_features.to(device=proj_param.device, dtype=proj_param.dtype)

        return self._forward_no_cache(
            input_ids,
            latent_features,
            special_token_positions,
            question_start_indices=question_start_indices,
            answer_start_indices=answer_start_indices,
            neighbor_latents=neighbor_latents,
            neighbor_mask=neighbor_mask,
        )

    def _forward_no_cache(self, input_ids: torch.Tensor, latent_features: torch.Tensor,
                          feature_start_indices, *,
                          question_start_indices: Optional[torch.Tensor] = None,
                          answer_start_indices: Optional[torch.Tensor] = None,
                          neighbor_latents: Optional[torch.Tensor] = None,
                          neighbor_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        bsz, seqlen = input_ids.shape
        device = input_ids.device

        # Token embeddings from base model
        token_embeddings = self.base_model.tok_embeddings(input_ids)

        # Project spectral features to K token embeddings
        spec_tokens = self.projector(latent_features)  # (B, K, d_model)
        # Optional: reconstruct latent from tokens for auxiliary loss
        latent_recon = self.inverse_projector(spec_tokens)  # (B, latent_dim)

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

        # Mask handled inside _attn_no_cache to accommodate dynamic lengths
        mask = None

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

        # Optional text->latent prediction computed inside forward for DDP correctness
        pred_latent_from_text = None
        if question_start_indices is not None and answer_start_indices is not None:
            try:
                pooled = self.pool_question_hidden(h, question_start_indices, answer_start_indices)
                pred_latent_from_text = self.text_to_latent(pooled)
            except Exception:
                pred_latent_from_text = None

        physics_pred = None
        if pred_latent_from_text is not None and self.latent_to_physics is not None:
            physics_pred = self.latent_to_physics(pred_latent_from_text)

        neighbor_logits = None
        if pred_latent_from_text is not None and neighbor_latents is not None and neighbor_latents.numel() > 0:
            neighbor_latents = neighbor_latents.to(device=latent_features.device, dtype=latent_features.dtype)
            query = F.normalize(pred_latent_from_text, dim=-1)
            neigh = F.normalize(neighbor_latents, dim=-1)
            neighbor_logits = torch.matmul(query.unsqueeze(1), neigh.transpose(1, 2)).squeeze(1)
            if neighbor_mask is not None:
                neighbor_mask_float = neighbor_mask.to(device=neighbor_logits.device, dtype=neighbor_logits.dtype)
                neighbor_logits = neighbor_logits.masked_fill(neighbor_mask_float < 0.5, float('-inf'))
                neighbor_mask = neighbor_mask_float
            if self.neighbor_temperature > 0:
                neighbor_logits = neighbor_logits / self.neighbor_temperature

        return {
            "logits": logits,
            "h": h,
            # For auxiliary invertibility loss (used only if trainer enables it)
            "latent_recon_from_tokens": latent_recon,
            "latent_target": latent_features,
            "pred_latent_from_text": pred_latent_from_text,
            "physics_pred": physics_pred,
            "neighbor_logits": neighbor_logits,
            "neighbor_mask": neighbor_mask,
        }

    def pool_question_hidden(self, h: torch.Tensor, q_start: torch.Tensor, a_start: torch.Tensor) -> torch.Tensor:
        """Mean-pool hidden states over [q_start, a_start) per sample.

        Args:
            h: [B, S, D]
            q_start: [B] long tensor
            a_start: [B] long tensor
        Returns:
            pooled: [B, D]
        """
        B, S, D = h.shape
        pooled = []
        for b in range(B):
            qs = int(q_start[b].item())
            as_ = int(a_start[b].item())
            qs = max(0, min(qs, S-1))
            as_ = max(qs+1, min(as_, S))  # ensure at least one token
            seg = h[b, qs:as_, :]
            pooled.append(seg.mean(dim=0))
        return torch.stack(pooled, dim=0)

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
        return {"logits": logits, "h": h}

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
        qlen, klen = scores.size(-2), scores.size(-1)
        if mask is None:
            causal = torch.full((qlen, klen), float('-inf'), device=scores.device, dtype=scores.dtype)
            causal = torch.triu(causal, diagonal=1)
            scores = scores + causal
        else:
            mask_to_add = mask.to(dtype=scores.dtype, device=scores.device)
            mask_to_add = mask_to_add[..., :qlen, :klen]
            scores = scores + mask_to_add
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
        if self.mode == "two_star":
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

        input_text = batch_data.get('input_texts', [''])[batch_idx]
        target_text = batch_data.get('target_texts', [''])[batch_idx]

        # Prompt = features + question (truncate before answer start)
        prompt = input_ids[:, :max(1, min(answer_start_idx, input_ids.shape[1]))].clone()
        gen_logps = []
        gen_ids = []

        def sample_top_p(logits: torch.Tensor) -> int:
            if temperature > 0:
                logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            if 0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cdf = torch.cumsum(sorted_probs, dim=-1)
                cutoff = (cdf > top_p).float().argmax().item()
                cutoff = max(1, cutoff)
                sorted_probs = sorted_probs[:cutoff]
                sorted_idx = sorted_idx[:cutoff]
                sorted_probs = sorted_probs / sorted_probs.sum()
                next_idx = torch.multinomial(sorted_probs, 1).item()
                return sorted_idx[next_idx].item()
            else:
                return torch.multinomial(probs, 1).item()

        # Ensure features fed to projector match projector dtype/device
        proj_param = next(self.projector.parameters())
        features_vec = input_spectra.view(prompt.size(0), -1).to(device=proj_param.device, dtype=proj_param.dtype)
        neighbor_latents = None
        neighbor_mask = None
        if 'neighbor_latents' in batch_data and batch_data['neighbor_latents'].numel() > 0:
            neighbor_latents = batch_data['neighbor_latents'][batch_idx:batch_idx+1].to(device)
            neighbor_mask = batch_data.get('neighbor_mask')
            if neighbor_mask is not None and neighbor_mask.numel() > 0:
                neighbor_mask = neighbor_mask[batch_idx:batch_idx+1].to(device)

        for _ in range(max_new_tokens):
            out = self._forward_no_cache(
                prompt,
                features_vec,
                feature_start_idx,
                neighbor_latents=neighbor_latents,
                neighbor_mask=neighbor_mask,
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
            if tokenizer is not None and hasattr(tokenizer, 'eos_id'):
                if next_token == getattr(tokenizer, 'eos_id'):
                    break

        generated_text = tokenizer.decode(torch.tensor(gen_ids).cpu().numpy()) if tokenizer is not None else ''
        return generated_text, input_text, target_text, gen_logps
