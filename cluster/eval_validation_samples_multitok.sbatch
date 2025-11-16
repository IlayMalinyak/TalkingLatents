#!/bin/bash
# Evaluate multi-token model on a random subset (single GPU)

#SBATCH --job-name=tl-eval-multitok
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=30
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH -C v100-32g
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

mkdir -p logs/slurm

module purge
module load pytorch-gpu/py3/2.2.0

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:64"}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# Paths
JSON_FILE=/lustre/fswork/projects/rech/oxl/utl47bv/data/stellar_descriptions_questions_short.json
FEATURES_FILE=/lustre/fswork/projects/rech/oxl/utl47bv/data/features.npy
LLM_PATH=/lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Meta-Llama-3.1-8B/original
LLM_BACKEND=${LLM_BACKEND:-llama}
HF_MODEL_NAME=${HF_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}
HF_QUANTIZATION=${HF_QUANTIZATION:-4bit}
HF_DEVICE_MAP=${HF_DEVICE_MAP:-auto}
HF_TRUST_REMOTE_CODE=${HF_TRUST_REMOTE_CODE:-0}
HF_CACHE_DIR=${HF_CACHE_DIR:-/lustre/fswork/projects/rech/oxl/utl47bv/cache/hf}
HF_MAX_MEMORY_GB=${HF_MAX_MEMORY_GB:-40}
HF_AUTH_TOKEN=${HF_AUTH_TOKEN:-${HF_TOKEN:-}}
export HF_HOME=${HF_HOME:-${HF_CACHE_DIR}}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}

# Eval knobs (override with --export=VAR=VALUE)
OUTDIR=${OUTDIR:-logs/eval_multitok}
EXP=${EXP:-eval_multitok}
NUM_SAMPLES=${NUM_SAMPLES:-15}
TOKS=${TOKS:-4}
MAXLEN=${MAXLEN:-96}
TOKENS=${TOKENS:-100}          # generation length
TEMP=${TEMP:-0.2}
TOPP=${TOPP:-0.8}
SEED=${SEED:-42}
LLM_PREC=${LLM_PREC:-fp16}

mkdir -p "${OUTDIR}"

# You can provide one of:
RESUME=/lustre/fswork/projects/rech/oxl/utl47bv/python/TalkingLatents/logs/train_ddp/2025-09-20-00-51/llm_multitok_resume_best.pth
#   WEIGHTS=/path/to/llm_multitok.pth           (plain weights)

SR_ARGS=(
  --llm_path "${LLM_PATH}"
  --llm_backend "${LLM_BACKEND}"
  --json_file "${JSON_FILE}"
  --features_file "${FEATURES_FILE}"
  --output_dir "${OUTDIR}"
  --exp_name "${EXP}"
  --num_spectral_features "${TOKS}"
  --max_seq_length "${MAXLEN}"
  --num_samples "${NUM_SAMPLES}"
  --max_new_tokens "${TOKENS}"
  --temperature "${TEMP}"
  --top_p "${TOPP}"
  --random_seed "${SEED}"
  --llm_precision "${LLM_PREC}"
)

if [[ "${LLM_BACKEND}" != "llama" ]]; then
  SR_ARGS+=(--hf_model_name "${HF_MODEL_NAME}")
  SR_ARGS+=(--hf_quantization "${HF_QUANTIZATION}")
  if [[ -n "${HF_DEVICE_MAP}" ]]; then
    SR_ARGS+=(--hf_device_map "${HF_DEVICE_MAP}")
  fi
  if [[ -n "${HF_CACHE_DIR}" ]]; then
    SR_ARGS+=(--hf_cache_dir "${HF_CACHE_DIR}")
  fi
  if [[ -n "${HF_MAX_MEMORY_GB}" ]]; then
    SR_ARGS+=(--hf_max_memory_gb "${HF_MAX_MEMORY_GB}")
  fi
  if [[ -n "${HF_AUTH_TOKEN}" ]]; then
    SR_ARGS+=(--hf_auth_token "${HF_AUTH_TOKEN}")
  fi
  case "${HF_TRUST_REMOTE_CODE,,}" in
    1|true|yes)
      SR_ARGS+=(--hf_trust_remote_code)
      ;;
  esac
fi

if [[ -n "${RESUME:-}" ]]; then
  srun --kill-on-bad-exit=1 \
    python -u src/eval_validation_samples_multitok.py \
      "${SR_ARGS[@]}" \
      --resume_path "${RESUME}"
elif [[ -n "${WEIGHTS:-}" ]]; then
  srun --kill-on-bad-exit=1 \
    python -u src/eval_validation_samples_multitok.py \
      "${SR_ARGS[@]}" \
      --weights "${WEIGHTS}"
else
  srun --kill-on-bad-exit=1 \
    python -u src/eval_validation_samples_multitok.py \
      "${SR_ARGS[@]}"
fi
