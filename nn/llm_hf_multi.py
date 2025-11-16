from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from .llm_multi import MultimodalBackboneBase


class HuggingFaceMultimodalModel(MultimodalBackboneBase):
    """Multimodal wrapper for Hugging Face causal LMs (e.g., Qwen)."""

    def __init__(self, base_model, fm_model, *args, **kwargs):
        super().__init__(base_model=base_model, fm_model=fm_model, *args, **kwargs)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        token_embeddings, cfm_targets = self._prepare_embeddings(batch)
        attention_mask = batch.get('attention_mask', None)
        hf_kwargs = {
            "inputs_embeds": token_embeddings,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "use_cache": False,
        }
        hf_kwargs = {k: v for k, v in hf_kwargs.items() if v is not None}
        outputs = self.base_model(**hf_kwargs)
        hidden_states = getattr(outputs, "last_hidden_state", outputs[0])
        logits = getattr(outputs, "logits", None)
        if logits is None:
            lm_head = getattr(self.base_model, "lm_head", None)
            if lm_head is None:
                raise ValueError("HF model output lacks logits and no lm_head is available.")
            logits = lm_head(hidden_states)
        logits = logits.float()
        out_dict = {"logits": logits, "h": hidden_states}
        return self._add_common_outputs(hidden_states, out_dict, batch, cfm_targets)

    def _prepare_embeddings(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        input_ids = batch['input_ids']
        embed_module = self.base_model.get_input_embeddings()

        # Replace negative placeholders (e.g., -100) with a valid token ID for embedding lookup.
        # Prefer model pad token; fall back to EOS; finally 0.
        pad_id = None
        config = getattr(self.base_model, 'config', None)
        if config is not None:
            pad_id = getattr(config, 'pad_token_id', None)
            if pad_id is None:
                pad_id = getattr(config, 'eos_token_id', None)
        if pad_id is None:
            pad_id = 0

        if torch.is_tensor(input_ids):
            input_ids = input_ids.clone()
            neg_mask = input_ids < 0
            if torch.any(neg_mask):
                input_ids[neg_mask] = int(pad_id)

        token_embeddings = embed_module(input_ids)
        token_embeddings = token_embeddings.clone()
        seq_len = token_embeddings.size(1)
        cfm_targets: List[torch.Tensor] = []

        device = token_embeddings.device
        
        def clamp_start(start: int) -> int:
            if start is None:
                return 0
            if start < 0:
                return 0
            max_start = max(0, seq_len - self.num_spectral_features)
            if start > max_start:
                return max_start
            return start
        debug = os.environ.get('TL_DEBUG', '0') == '1'

        def _safe_index_select_rows(t: torch.Tensor, idx_list: List[int], label: str) -> Tuple[torch.Tensor, List[int]]:
            if t is None:
                return t, []
            bsz = t.size(0)
            valid = [i for i in idx_list if 0 <= i < bsz]
            if debug and len(valid) != len(idx_list):
                print(f"[HF-MM] filtered invalid {label} rows: total={len(idx_list)} valid={len(valid)} (tensor rows={bsz})", flush=True)
            if len(valid) == 0:
                return t[:0], []
            idx_tensor = torch.tensor(valid, dtype=torch.long, device=t.device)
            return t.index_select(0, idx_tensor), valid

        # Single-star samples
        if 'masked_spectra' in batch and batch['masked_spectra'] is not None:
            single_mask = batch.get('mode_mask_single', torch.ones(token_embeddings.size(0), dtype=torch.bool, device=device))
            present_single = batch.get('masked_spectra_present', None)
            if present_single is not None:
                # ensure same device and dtype
                present_single = present_single.to(device=single_mask.device, dtype=torch.bool)
                single_mask = single_mask & present_single
            # Use CPU for nonzero to avoid device assert cascades; then back to list
            single_indices_list = torch.nonzero(single_mask.detach().to('cpu'), as_tuple=False).squeeze(-1).tolist()
            if len(single_indices_list) > 0:
                # Safe index_select for spectra and positions
                single_spectra, valid_rows = _safe_index_select_rows(batch['masked_spectra'], single_indices_list, 'single')
                if len(valid_rows) == 0:
                    pass
                else:
                    pos_tensor = batch['feature_start_indices']
                    pos_sel, _ = _safe_index_select_rows(pos_tensor, valid_rows, 'single-pos')
                    single_positions = pos_sel.to('cpu')
                spec_tokens = self._project_spectra(single_spectra, self.projector)
                spec_tokens = spec_tokens.to(dtype=token_embeddings.dtype)
                for i, global_idx in enumerate(valid_rows):
                    start_pos = clamp_start(int(single_positions[i].item()))
                    end_pos = min(seq_len, start_pos + self.num_spectral_features)
                    token_embeddings[global_idx, start_pos:end_pos, :] = spec_tokens[i, :end_pos-start_pos, :]
                cfm_targets.extend([single_spectra[i] for i in range(len(valid_rows))])

        # Comparative samples
        if 'masked_spectra_a' in batch and batch['masked_spectra_a'] is not None:
            comp_mask = batch.get('mode_mask_comparative', torch.ones(token_embeddings.size(0), dtype=torch.bool, device=device))
            present_a = batch.get('masked_spectra_a_present', None)
            present_b = batch.get('masked_spectra_b_present', None)
            if present_a is not None:
                comp_mask = comp_mask & present_a.to(device=comp_mask.device, dtype=torch.bool)
            if present_b is not None:
                comp_mask = comp_mask & present_b.to(device=comp_mask.device, dtype=torch.bool)
            comp_indices_list = torch.nonzero(comp_mask.detach().to('cpu'), as_tuple=False).squeeze(-1).tolist()
            if len(comp_indices_list) > 0:
                comp_spectra_a, valid_rows = _safe_index_select_rows(batch['masked_spectra_a'], comp_indices_list, 'comp-a')
                comp_spectra_b, _ = _safe_index_select_rows(batch['masked_spectra_b'], valid_rows, 'comp-b')
                comp_indices_a, _ = _safe_index_select_rows(batch['star_a_feature_indices'], valid_rows, 'comp-a-idx')
                comp_indices_b, _ = _safe_index_select_rows(batch['star_b_feature_indices'], valid_rows, 'comp-b-idx')
                spec_tokens_a = self._project_spectra(comp_spectra_a, self.projector)
                spec_tokens_b = self._project_spectra(comp_spectra_b, self.projector)
                spec_tokens_a = spec_tokens_a.to(dtype=token_embeddings.dtype)
                spec_tokens_b = spec_tokens_b.to(dtype=token_embeddings.dtype)
                for i, global_idx in enumerate(valid_rows):
                    # Move index lists to CPU for safe filtering and use python ints when indexing
                    indices_a_cpu = comp_indices_a[i].to('cpu', dtype=torch.long)
                    indices_b_cpu = comp_indices_b[i].to('cpu', dtype=torch.long)
                    valid_indices_a = indices_a_cpu[(indices_a_cpu >= 0) & (indices_a_cpu < seq_len)]
                    if len(valid_indices_a) > 0:
                        num_tokens_a = min(len(valid_indices_a), spec_tokens_a.shape[1])
                        rows_a = [int(x) for x in valid_indices_a[:num_tokens_a].tolist()]
                        token_embeddings[global_idx, rows_a, :] = spec_tokens_a[i, :num_tokens_a, :]
                    valid_indices_b = indices_b_cpu[(indices_b_cpu >= 0) & (indices_b_cpu < seq_len)]
                    if len(valid_indices_b) > 0:
                        num_tokens_b = min(len(valid_indices_b), spec_tokens_b.shape[1])
                        rows_b = [int(x) for x in valid_indices_b[:num_tokens_b].tolist()]
                        token_embeddings[global_idx, rows_b, :] = spec_tokens_b[i, :num_tokens_b, :]
                cfm_targets.extend([torch.cat([comp_spectra_a[i], comp_spectra_b[i]], dim=-1) for i in range(len(valid_rows))])

        return token_embeddings, cfm_targets

    # ------------------------------------------------------------------ #
    # HF wrappers for inference-time chunked decoding compatibility
    # (match LLaMA interface used by follow_up_inference.py)
    # ------------------------------------------------------------------ #
    def _get_safe_pad_id(self) -> int:
        config = getattr(self.base_model, 'config', None)
        pad_id = None
        if config is not None:
            pad_id = getattr(config, 'pad_token_id', None)
            if pad_id is None:
                pad_id = getattr(config, 'eos_token_id', None)
        return 0 if pad_id is None else int(pad_id)

    def _embed_and_sanitize(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, int]:
        pad_id = self._get_safe_pad_id()
        if torch.is_tensor(input_ids):
            input_ids = input_ids.clone()
            neg_mask = input_ids < 0
            if torch.any(neg_mask):
                input_ids[neg_mask] = pad_id
        embed_module = self.base_model.get_input_embeddings()
        token_embeddings = embed_module(input_ids)
        return token_embeddings, pad_id

    @torch.no_grad()
    def _forward_single_mode(
        self,
        input_ids: torch.Tensor,
        latent_features: torch.Tensor,
        feature_start_indices,
        start_pos: int = 0,
        cache_rows=None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        token_embeddings, pad_id = self._embed_and_sanitize(input_ids)
        seq_len = token_embeddings.size(1)

        # Project spectral features to K token embeddings
        spec_tokens = self.projector(latent_features)
        spec_tokens = spec_tokens.to(dtype=token_embeddings.dtype, device=token_embeddings.device)

        # Normalize feature_start_indices to a 1D tensor of length bsz
        bsz = token_embeddings.size(0)
        if feature_start_indices is None:
            fsi = torch.zeros(bsz, dtype=torch.long, device=token_embeddings.device)
        elif isinstance(feature_start_indices, torch.Tensor):
            if feature_start_indices.dim() == 0:
                fsi = feature_start_indices.view(1).repeat(bsz).to(device=token_embeddings.device, dtype=torch.long)
            elif feature_start_indices.dim() == 1 and feature_start_indices.numel() == bsz:
                fsi = feature_start_indices.to(device=token_embeddings.device, dtype=torch.long)
            else:
                fsi = torch.zeros(bsz, dtype=torch.long, device=token_embeddings.device)
        else:
            fsi = torch.full((bsz,), int(feature_start_indices), dtype=torch.long, device=token_embeddings.device)

        # Insert K tokens per sample
        K = spec_tokens.shape[1]
        for b in range(bsz):
            start_idx = int(fsi[b].item()) - int(start_pos)
            if start_idx >= seq_len or (start_idx + K) <= 0:
                continue
            ins_start = max(0, start_idx)
            ins_end = min(seq_len, start_idx + K)
            spec_start = ins_start - start_idx
            spec_end = spec_start + (ins_end - ins_start)
            token_embeddings[b, ins_start:ins_end, :] = spec_tokens[b, spec_start:spec_end, :]

        hf_kwargs = {
            "inputs_embeds": token_embeddings,
            "use_cache": False,
            "output_hidden_states": True,
        }
        outputs = self.base_model(**hf_kwargs)
        hidden_states = getattr(outputs, "last_hidden_state", outputs[0])
        logits = getattr(outputs, "logits", None)
        if logits is None:
            lm_head = getattr(self.base_model, "lm_head", None)
            if lm_head is None:
                raise ValueError("HF model output lacks logits and no lm_head is available.")
            logits = lm_head(hidden_states)
        return {"logits": logits.float(), "h": hidden_states}

    @torch.no_grad()
    def _forward_two_star_mode(
        self,
        input_ids: torch.Tensor,
        star_a_features: torch.Tensor,
        star_b_features: torch.Tensor,
        star_a_indices: torch.Tensor,
        star_b_indices: torch.Tensor,
        start_pos: int = 0,
        cache_rows=None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        token_embeddings, pad_id = self._embed_and_sanitize(input_ids)
        seq_len = token_embeddings.size(1)

        spec_tokens_a = self.projector_a(star_a_features).to(dtype=token_embeddings.dtype, device=token_embeddings.device)
        spec_tokens_b = self.projector_b(star_b_features).to(dtype=token_embeddings.dtype, device=token_embeddings.device)

        bsz = token_embeddings.size(0)
        for b in range(bsz):
            idx_a = star_a_indices[b].to('cpu', dtype=torch.long)
            idx_b = star_b_indices[b].to('cpu', dtype=torch.long)
            valid_a = idx_a[(idx_a >= 0) & (idx_a < seq_len)]
            valid_b = idx_b[(idx_b >= 0) & (idx_b < seq_len)]
            if len(valid_a) > 0:
                n_a = min(len(valid_a), spec_tokens_a.shape[1])
                rows_a = [int(x) for x in valid_a[:n_a].tolist()]
                token_embeddings[b, rows_a, :] = spec_tokens_a[b, :n_a, :]
            if len(valid_b) > 0:
                n_b = min(len(valid_b), spec_tokens_b.shape[1])
                rows_b = [int(x) for x in valid_b[:n_b].tolist()]
                token_embeddings[b, rows_b, :] = spec_tokens_b[b, :n_b, :]

        hf_kwargs = {
            "inputs_embeds": token_embeddings,
            "use_cache": False,
            "output_hidden_states": True,
        }
        outputs = self.base_model(**hf_kwargs)
        hidden_states = getattr(outputs, "last_hidden_state", outputs[0])
        logits = getattr(outputs, "logits", None)
        if logits is None:
            lm_head = getattr(self.base_model, "lm_head", None)
            if lm_head is None:
                raise ValueError("HF model output lacks logits and no lm_head is available.")
            logits = lm_head(hidden_states)
        return {"logits": logits.float(), "h": hidden_states}

    def _slice_batch(self, batch_data: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        sliced = {}
        for key, value in batch_data.items():
            if torch.is_tensor(value):
                sliced[key] = value[batch_idx:batch_idx+1]
            elif isinstance(value, list):
                sliced[key] = [value[batch_idx]]
            else:
                sliced[key] = value
        return sliced

    @torch.no_grad()
    def generate_response_from_batch(self,
                                     batch_data: dict,
                                     batch_idx: int = 0,
                                     tokenizer=None,
                                     max_new_tokens: int = 100,
                                     temperature: float = 0.7,
                                     top_p: float = 0.9) -> tuple:
        self.eval()
        device = next(self.parameters()).device
        sample_batch = self._slice_batch(batch_data, batch_idx)
        token_embeddings, _ = self._prepare_embeddings(sample_batch)
        attention_mask = sample_batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        token_embeddings = token_embeddings.to(device)

        pad_token_id = getattr(tokenizer, 'pad_id', None)
        if pad_token_id is None:
            pad_token_id = getattr(self.base_model.config, 'pad_token_id', None)
        if pad_token_id is None:
            pad_token_id = getattr(self.base_model.config, 'eos_token_id', None)

        gen_kwargs = {
            "inputs_embeds": token_embeddings,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "temperature": max(temperature, 0.0),
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": pad_token_id,
        }
        sequences = self.base_model.generate(**gen_kwargs)
        prompt_len = token_embeddings.shape[1]
        generated_ids = sequences[:, prompt_len:]
        if tokenizer is not None and generated_ids.numel() > 0:
            generated_text = tokenizer.decode(generated_ids[0].tolist())
        else:
            generated_text = ""

        input_text, target_text = self._extract_text_fields(batch_data, batch_idx)
        return generated_text, input_text, target_text, []
