# Refactor Plan: Qwen/HF Backend Enablement

## Current State

- **Model loader**: `src/simple_questions.py` now dispatches `_load_llm_model` based on `--llm_backend`. LLaMA checkpoints still load via the existing Meta implementation; `--llm_backend hf` loads `AutoModelForCausalLM` but the rest of the stack cannot yet use it.
- **Tokenizer/data pipeline**: `src/tokenizer_adapter.py` plus dataset changes allow either Meta's SentencePiece tokenizer or a Hugging Face tokenizer to drive all dataloaders. Mixed/comparative loaders already receive the adapter.
- **Multimodal wrapper**: `nn/llm_multi.py` is tightly bound to Meta's LLaMA internals (directly accesses `base_model.tok_embeddings`, `freqs_cis`, layer caches, generation helpers, etc.). No HF-aware variant exists.
- **Training/inference scripts**: `src/simple_questions_multitok.py` and downstream utilities keep invoking `MultimodalLlamaModelMultiTokens`. A guard currently blocks `--llm_backend hf` to avoid deeper failures.
- **Dependencies**: `transformers`/`accelerate`/`bitsandbytes` are not installed by default; no HF-specific env handling exists in Slurm scripts.

## Refactor Steps

### 1. Dependency + Config Preparation *(finished)*
- ✅ Added `requirements_qwen.txt` plus `docs/qwen_backend_setup.md` so the HF/Qwen dependency stack and cache env vars are spelled out for cluster runs.
- ✅ Extended `parse_args` in `src/simple_questions_multitok.py`, tokenizer plumbing, and `cluster/train_simple_questions_multitok.sbatch` to accept/pass HF-specific knobs (backend alias, model id, quantization, device map, trust_remote_code, cache dir).

### 2. Backend-Agnostic Model Loading *(finished)*
- ✅ `_load_hf_llm_model` now handles 4-bit/8-bit BitsAndBytes loading, shared cache/device-map controls, per-device memory caps, and automatically falls back to CPU when CUDA isn’t accessible.
- ✅ Added `LLMBackendConfig` (plus `ensure_backend_config`) so dataset/model builders—and downstream inference scripts—receive a consistent view of backend metadata for checkpointing/logging.

### 3. Hugging Face Multimodal Wrapper *(finished)*
- ✅ `nn/llm_multi.py` now exposes `MultimodalBackboneBase`, letting the LLaMA adapter reuse shared spectral projectors + auxiliary heads while the new `nn/llm_hf_multi.py::HuggingFaceMultimodalModel` injects tokens via `inputs_embeds` and calls HF forward/generate APIs.
- ✅ HF adapter mirrors the single/two-star token replacement logic, honors attention masks, and implements `generate_response_from_batch` using `model.generate`, so downstream scripts can decode answers without custom cache hacks.
- ✅ All builder/inference entrypoints route through `LLMBackendConfig`, choosing the appropriate adapter transparently while keeping CFM, stellar, classification, and feature heads consistent across backends.

### 4. Training Loop Integration *(finished)*
- ✅ `build_model_multitok` now selects the correct adapter, skips unsafe dtype casts for HF models, applies gradient-checkpointing guards (auto-disabling it for multi-GPU or quantized runs), and keeps auxiliary heads on the requested device even when the HF backbone uses a device map.
- ✅ Trainer instances carry a serialized backend config so resume checkpoints, LoRA freezes, and optimizer parameter groups know which layers belong to the frozen backbone vs. the multimodal heads.

### 5. Checkpointing & Resume Support *(finished)*
- ✅ `_build_resume_checkpoint` stores the backend metadata alongside optimizer/scheduler/scaler states, and `prepare_training_with_resume` surfaces mismatches so we don’t accidentally load HF weights into a LLaMA wrapper.
- ✅ `save_config` and every training script now persist the backend block (`backend_config`) so downstream inference knows exactly which tokenizer/model/quantization combo produced the checkpoint.

### 6. Inference & Follow-Up Scripts *(finished)*
- ✅ `src/inference.py`, `src/follow_up_inference.py`, and the rebuilt `src/baseline_follow_up.py` all call `build_model_multitok`, share the tokenizer adapter, and inject backend metadata into their JSON payloads for reproducibility.
- ✅ Baseline follow-up evaluation no longer maintains a separate Qwen code path—it reuses the multimodal wrapper, so HF/LLaMA parity is guaranteed.

### 7. Slurm / CLI Updates *(finished)*
- ✅ `cluster/train_simple_questions_multitok.sbatch` and `cluster/eval_validation_samples_multitok.sbatch` expose `LLM_BACKEND` plus the HF knobs (model id, quantization, cache dir, device map) so switching backends on the cluster is a one-line environment override.

### 8. Validation & Testing *(finished)*
- ✅ Added `tests/test_multimodal_backends.py` with dummy HF/LLaMA models that exercise forward passes and generation on both adapters (and wired it into CI via `pytest`).
- ✅ The new smoke tests run automatically (`pytest tests/test_multimodal_backends.py`) and keep regressions localized without touching the full checkpoints.

### 9. Documentation *(finished)*
- ✅ `docs/qwen_backend_setup.md` now covers inference/follow-up usage and references the new smoke tests so users know how to validate an install quickly.
- ✅ Backends recorded in JSON outputs + training configs make it trivial to diagnose quantization/device-map issues called out in the doc.

## Recommended Sequencing
1. Dependencies + CLI flags (Steps 1-2).
2. Implement HF multimodal wrapper + adapter classes (Step 3).
3. Integrate with training loop, checkpointing, and inference (Steps 4-6).
4. Update Slurm scripts and documentation (Steps 7 & 9).
5. Validation/testing (Step 8) throughout to catch regressions early.

Each step should ship in its own PR or commit to keep reviews manageable and to avoid destabilizing existing LLaMA workflows.
