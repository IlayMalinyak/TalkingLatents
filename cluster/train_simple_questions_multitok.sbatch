#!/bin/bash
#SBATCH --job-name=TalkingLatents
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:2
#SBATCH --qos=basic
#SBATCH --mem=64G


# Data/model paths (edit to your filesystem)
JSON_FILE=/data/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json
COMPARATIVE_JSON_FILE=/data/TalkingLatents/data/dataset/comparative_dataset.json
ADVANCED_JSON_PATH=/data/TalkingLatents/data/dataset/caption_advanced_100k.json
DATASET_TYPE=advanced
MODE=combined
FEATURES_FILE=/data/TalkingLatents/logs/2025-07-29/features.npy
LLM_PATH=/data/.llama/Llama3.1-8B

# Training knobs (override at submit time with --export=VAR=VALUE)
OUTDIR=${OUTDIR:-/data/TalkingLatents/logs}
EXP=${EXP:-llm_multitok}
EPOCHS=${EPOCHS:-15}
BATCH=${BATCH:-8}              # per-GPU
MAXLEN=${MAXLEN:-256}
WORKERS=${WORKERS:-4}
LR=${LR:-1e-4}
WD=${WD:-1e-3}
TOKS=${TOKS:-8}                # number of spectral tokens (K)
SWITCH=${SWITCH:-3}
LAMBDA=${LAMBDA:-1.0}
LLM_BACKEND=${LLM_BACKEND:-llama}
HF_MODEL_NAME=${HF_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}
HF_QUANTIZATION=${HF_QUANTIZATION:-4bit}
HF_DEVICE_MAP=${HF_DEVICE_MAP:-auto}
HF_TRUST_REMOTE_CODE=${HF_TRUST_REMOTE_CODE:-0}
HF_MAX_MEMORY_GB=${HF_MAX_MEMORY_GB:-40}
HF_AUTH_TOKEN=${HF_AUTH_TOKEN:-${HF_TOKEN:-}}
HF_HOME=${HF_HOME:-/data/.cache/huggingface}
export HF_HOME
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
HF_CACHE_DIR=${HF_CACHE_DIR:-$HF_HOME}

PYTHON_CMD=(
  python -u /data/TalkingLatents/src/simple_questions_multitok.py
    --llm_path "${LLM_PATH}"
    --json_file "${JSON_FILE}"
    --comparative_json_file "${COMPARATIVE_JSON_FILE}"
    --advanced_json_file "${ADVANCED_JSON_PATH}"
    --single_dataset_type "${DATASET_TYPE}"
    --mode "${MODE}"
    --features_file "${FEATURES_FILE}"
    --output_dir "${OUTDIR}"
    --exp_name "${EXP}"
    --batch_size "${BATCH}"
    --num_epochs "${EPOCHS}"
    --learning_rate "${LR}"
    --weight_decay "${WD}"
    --max_seq_length "${MAXLEN}"
    --num_workers "${WORKERS}"
    --num_spectral_features "${TOKS}"
    --loss_lambda "${LAMBDA}"
    --llm_precision fp16
    --llm_backend "${LLM_BACKEND}"
    --use_amp
    --gradient_checkpointing
    --enable_classification
    --max_iter -1
)

if [[ "${LLM_BACKEND}" != "llama" ]]; then
  PYTHON_CMD+=(--hf_model_name "${HF_MODEL_NAME}")
  PYTHON_CMD+=(--hf_quantization "${HF_QUANTIZATION}")
  if [[ -n "${HF_DEVICE_MAP}" ]]; then
    PYTHON_CMD+=(--hf_device_map "${HF_DEVICE_MAP}")
  fi
  if [[ -n "${HF_CACHE_DIR}" ]]; then
    PYTHON_CMD+=(--hf_cache_dir "${HF_CACHE_DIR}")
  fi
  if [[ -n "${HF_MAX_MEMORY_GB}" ]]; then
    PYTHON_CMD+=(--hf_max_memory_gb "${HF_MAX_MEMORY_GB}")
  fi
  if [[ -n "${HF_AUTH_TOKEN}" ]]; then
    PYTHON_CMD+=(--hf_auth_token "${HF_AUTH_TOKEN}")
  fi
  case "${HF_TRUST_REMOTE_CODE,,}" in
    1|true|yes)
      PYTHON_CMD+=(--hf_trust_remote_code)
      ;;
  esac
fi

srun --kill-on-bad-exit=1 \
 --container-image="/rg/perets_prj/ilay.kamai/containers/astro2.sqsh" \
     --mem=64G \
     --output=/home/ilay.kamai/athena/logs/TalkingLatents_%j_%s_%t.out \
     --error=/home/ilay.kamai/athena/logs/TalkingLatents_%j_%s_%t.err \
     --no-container-entrypoint \
     --container-mounts=/home/ilay.kamai/work:/data \
  "${PYTHON_CMD[@]}"
