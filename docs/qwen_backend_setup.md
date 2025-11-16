# Qwen / Hugging Face Backend Setup

This project can now swap the original Meta LLaMA backbone for a Hugging Face
model such as Qwen. Before enabling the HF backend, make sure the following
packages are installed (see `requirements_qwen.txt` for pinned versions):

```
pip install -r requirements_qwen.txt
```

## Cluster-Friendly Caches

Large HF checkpoints are downloaded into a cache. Configure it once in your
Slurm script or shell to avoid repeated downloads and to keep the cache on
high-throughput storage:

```bash
export HF_HOME=${HF_HOME:-/data/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
```

You can also override the cache per run through the new `--hf_cache_dir`
argument that was added to `simple_questions_multitok.py`.

## CLI Flags for HF/Qwen

Every training/inference entry point now understands the following HF-specific
flags (pass them only when `--llm_backend qwen`/`--llm_backend hf`):

- `--hf_model_name`: Hugging Face repo id or local path (e.g. `Qwen/Qwen2.5-7B-Instruct`)
- `--hf_quantization`: Choose between `none`, `8bit`, or `4bit` BitsAndBytes loading
- `--hf_device_map`: Device placement string/dict (e.g. `auto`, `cuda:0`, or JSON)
- `--hf_trust_remote_code`: Allow custom modeling code when a model repo requires it
- `--hf_cache_dir`: Optional override for the cache directory above
- `--hf_max_memory_gb`: (Optional) soft cap per device, forwarded to the loader in a later step

These options are also parameterized via environment variables in
`cluster/train_simple_questions_multitok.sbatch` so they can be tuned without
editing the script.

## Inference & Follow-Up Scripts

`src/inference.py`, `src/follow_up_inference.py`, and
`src/baseline_follow_up.py` understand the same backend flags. When running the
evaluation utilities, pass `--llm_backend hf` (or `qwen`) together with the HF
arguments above so the builders can instantiate the correct wrapper. The JSON
artifacts emitted by these scripts now record the backend metadata under the
`metadata.backend` field to keep comparisons between LLaMA and Qwen runs
traceable.

## Smoke Tests

Minimal unit tests live in `tests/test_multimodal_backends.py`. They instantiate
both the LLaMA-style adapter and the Hugging Face adapter with lightweight dummy
models to verify that multimodal token injection and text generation work
without touching the heavy checkpoints. Run them via:

```bash
pytest tests/test_multimodal_backends.py
```

This quickly validates that backend switching works before launching a large
training or inference job.
