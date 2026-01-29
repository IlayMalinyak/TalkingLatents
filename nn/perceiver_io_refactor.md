# Refactor Guide: Perceiver IO + Contrastive Alignment for Spectra–Text Modelling

This document describes the architecture and training strategy for transforming the current late-fusion system (`nn/late_fusion.py`) into a full **Perceiver IO multimodal backbone**, optionally enhanced with **CLIP-style contrastive alignment**.

The goal of this hybrid approach is to achieve:

- strong **cross-modal fusion** (Perceiver IO)
- strong **global alignment** (CLIP loss)
- bidirectional generation (text↔spectra)
- smooth, structured latent geometry
- retrieval and interpolation

---

## ✅ Core Architectural Principles

The Perceiver IO architecture is built around:

1. **Input Adapters**  
   Convert raw modality embeddings to a unified dimension `d_model` via a light linear projection per modality (`text_in → d_model`, `spec_in → d_model`).

2. **RoPE Positional Encoding (Required)**  
   Apply Rotary Positional Embeddings to Q/K vectors to encode order smoothly and continuously:
   - Apply RoPE in **cross-attention** (latent→inputs) and **latent self-attention** blocks.
   - Position indices:
     - text tokens: `0..L-1`
     - spectral tokens: `0..K-1`
     - latent tokens: `0..M-1` (artificial index or learned table)
   - RoPE replaces additive positional embeddings.

3. **Latent Array**  
   A compact, learned set of latent tokens (`M × d_model`) that act as a bottleneck and fusion space. `M` kept small (32–64) for efficiency.

4. **Cross-Attention Encode**  
   Latent tokens attend to serialized multimodal inputs (concatenate sequences with modality embeddings).

5. **Latent Self-Attention Stack**  
   Deep Transformer blocks over the latent tokens refine the joint representation.

6. **Output Heads**  
   Decoders conditioned on latents produce:
   - **Spectral reconstruction** (1D decoder; FiLM conditioning or query-cross-attn)
   - **LLM prefix embeddings** (linear adapter to `d_llm`)
   - **Global alignment embeddings** (pooled latent → MLP → normalized)

---

## ✅ Two Training Modes (Important Distinction)

The hybrid supports two **conceptually different** training modes that you will *both* use.

### Mode A — Perceiver IO Joint Fusion (Token-level cross-modal mixing)

**Inputs:**  
`[Z_text; Z_spec] → InputAdapters → RoPE → PerceiverIO(latent)`

**Loss:**  
```text
L_A = w_spec * MSE(decoded_spectra, original_spectra)
    + w_text * CE(decoded_text, original_text)   # optional, if text teacher available
```

**What alignment does this produce?**  
- ✅ **Implicit/functional alignment** via a *shared latent bottleneck*  
- ✅ Cross-modal reasoning & conditional generation (text↔spectra)  
- ✅ Interpolation support and missing-modality reconstruction

**Why it works:**  
- Cross-attention injects dependencies between modalities into the same latent tokens  
- Both reconstruction losses backprop through that same bottleneck  
- The latent capacity constraint (small `M`) forces feature sharing

> Result: alignment emerges **without** any explicit contrastive term.

---

### Mode B — CLIP-style Contrastive Alignment (Separate unimodal passes)

**Inputs:**  
- `Z_text → PerceiverEncoder (no spec) → pooled g_text`
- `Z_spec → PerceiverEncoder (no text) → pooled g_spec`

**Loss:**  
```text
L_B = InfoNCE(g_text, g_spec)   # symmetric, with learnable temperature τ
```

**What alignment does this produce?**  
- ✅ **Global/geometric alignment** for retrieval and cosine-similarity structure  
- ✅ Retrieval-ready embeddings and semantic proximity in the pooled space  
- ❌ No reconstruction, no token-level mixing

> Important: For this branch you **must not** fuse modalities; feed each modality alone.

---

### Why combine A + B?

They are **complementary**:

| Capability | From Mode A (Fusion) | From Mode B (CLIP) |
|---|---|---|
| Reconstruction | ✅ | ❌ |
| Cross-modal generation | ✅ | ❌ |
| Token-level interaction | ✅ | ❌ |
| Retrieval (ANN) | ⚠️ weak | ✅ strong |
| Cosine-geometry structure | ⚠️ weak | ✅ strong |
| Interpolation | ✅ semantic | ✅ geometric |

**Total loss:** `L_total = L_A + L_B` (with weights).

---

## ✅ Joint Fusion vs CLIP Alignment (Conceptual)

- **Joint Fusion** actually **mixes** tokens via attention inside one latent space.  
- **CLIP Alignment** only **aligns** two *separately encoded* embeddings.

Even if you always have paired (text, spectra) at training time, CLIP *by itself* never fuses them; Perceiver IO *does*. Use both to get fusion **and** retrieval.

---

## ✅ Outputs & Shapes

- **LLM Adapter:** `latents [B,M,d_model] → prefix [B,M,d_llm]`  
  (Used externally for frozen LLM captioning loss; not computed inside this module.)

- **Spectral Decoder:**  
  - **FiLM-1D (default):** pooled latent `[B,d_model]` conditions a 1D decoder to produce `[B,K]`. Efficient for large `K`.
  - **Query-CrossAttn (optional):** learn `Q_spec [K,d_model]`, cross-attend to latent, then project → `[B,K]`. Heavier for large `K`.

- **Alignment Head:** pooled latent → MLP → `g_text/g_spec ∈ [B, d_align]` (L2-normalized).

---

## ✅ Training Strategy

1) **Stage 1 — Perceiver-only (Fusion)**
   - Freeze LLM & spectral encoders
   - Train `L_A` (reconstruction); optionally a small teacher text decoder if available

2) **Stage 2 — Add Contrastive (Retrieval)**
   - Enable `L_B` with symmetric InfoNCE
   - Still keep encoders frozen; tune Perceiver latent core, decoder, and alignment head

3) **Stage 3 — LLM Alignment (Captioning)**
   - Freeze core; train *only* the `LLMAdapter` linear map with external autoregressive CE
   - Optionally LoRA on cross-attn keys/queries for light refinement

---

## ✅ Implementation Plan (Files)

```
nn/
  positional_encodings.py   # RoPE helper (Q/K rotation)
  perceiver_io.py           # latent core: cross-attn encode + latent self-attn stack
  perceiver_decoder.py      # spectral decoder heads (film_1d / query_crossattn)
  late_fusion.py            # wrapper integrating adapters, encoders, heads, losses
```

---

## ✅ API / Forward Contract

`forward(batch: Dict[str, Tensor]) -> Dict[str, Tensor]`

**Batch keys:**
- `'input_ids'`, `'attention_mask'` for text
- `'spectral_data'` (shape `[B,K]` or `[B,K,C]`) for spectra

**Returns (subset depending on mode):**
```python
{
  "latents": Tensor[B,M,d_model],             # optional (debug)
  "g_text": Tensor[B,d_align],                # for retrieval
  "g_spec": Tensor[B,d_align],                # for retrieval
  "contrastive_loss": Tensor[],               # scalar
  "prefix_embeddings": Tensor[B,M,d_llm],     # for external LLM CE
  "spectral_reconstruction": Tensor[B,K],     # or [B,K,C]
  "spectral_targets": Tensor[B,K],            # or [B,K,C]
}
```

---

## ✅ Configuration Defaults

Provide `create_default_config_perceiver()` with sensible defaults:

```jsonc
{
  "d_model": 768,
  "d_align": 512,
  "d_llm": 4096,
  "M_latent": 32,
  "num_latent_blocks": 8,
  "num_heads": 8,
  "ffn_mult": 4,
  "dropout": 0.1,
  "attn_dropout": 0.0,
  "pos_encoding": "rope",
  "spectral_decode_mode": "film_1d",  // or "query_crossattn"
  "freeze_llm": true,
  "freeze_spectral": true,
  "use_joint_encode": true,
  "learn_temperature": true,
  "temperature_init": 0.07,
  "recon_loss": "mse",
  "iclm_warmup": false
}
```

---

## ✅ Practical Notes & Tips

- **Masks:** pass `attention_mask` into cross-attn key padding mask; spectra default to full-ones unless masked data.
- **Efficiency:** keep `M` small; avoid per-position queries for very large `K` unless needed.
- **RoPE:** implement once in the attention module; keep everything `batch_first=True`.
- **Interleaving:** you can randomize whether to run joint fusion, contrastive, or both on a given batch to balance gradients.
- **Diagnostics:** track retrieval@k, cycle consistency (text→spec→text), spectral MAE, and latent smoothness (finite differences).

---

## ✅ Minimal Pseudocode (Combined Loss)

```python
# --- Fusion pass (Mode A) ---
Z_text, Z_spec = adapters(..., rope=True)
latent_joint = perceiver_encode(Z_text, Z_spec)   # concatenated inputs → latent
spec_pred = spectral_decoder(latent_joint)        # [B,K]
L_A = mse(spec_pred, spec_gt)
# (Optional) text teacher: L_A += w_text * CE(text_pred, text_tgt)

# --- Contrastive passes (Mode B) ---
latent_text = perceiver_encode(Z_text)            # text-only
latent_spec = perceiver_encode(Z_spec)            # spec-only
g_text = align_head(latent_text)                  # L2-normalized
g_spec = align_head(latent_spec)                  # L2-normalized
L_B = clip_loss(g_text, g_spec, tau)

# --- Captioning (external) ---
prefix = llm_adapter(latent_joint)                # [B,M,d_llm]
# external: L_LM = autoregressive_CE(frozen_LLM, prefix, tgt_tokens)

L_total = wA * L_A + wB * L_B
L_total.backward(); optimizer.step()
```

---

## ✅ Summary

- **Mode A (Fusion)** supplies *functional alignment*: cross-modal reasoning, reconstruction, and generation through a **shared latent bottleneck**.
- **Mode B (Contrastive)** supplies *geometric alignment*: retrieval-ready, cosine-meaningful global embeddings.
- **RoPE** improves stability and interpolation across both attention types.
- The combined approach gives you a single, powerful, and efficient multimodal foundation for spectra↔text.

---

**Next steps:**  
I can generate class stubs for `positional_encodings.py` (RoPE), `perceiver_io.py`, and `perceiver_decoder.py`, wired into your `nn/late_fusion.py` wrapper if you want a runnable skeleton.
