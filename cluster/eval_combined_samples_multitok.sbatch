#!/bin/bash
# Evaluate multitoken model on combined single-star and comparative test samples

#SBATCH --job-name=tl-eval-multitok-combo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=30
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH -C v100-32g
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

mkdir -p logs/slurm

module purge
module load pytorch-gpu/py3/2.2.0

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:64"}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

JSON_FILE=/lustre/fswork/projects/rech/oxl/utl47bv/data/stellar_descriptions_questions_short.json
COMPARATIVE_JSON_FILE=/lustre/fswork/projects/rech/oxl/utl47bv/data/comparative_dataset.json
FEATURES_FILE=/lustre/fswork/projects/rech/oxl/utl47bv/data/features.npy
LLM_PATH=/lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Meta-Llama-3.1-8B/original

OUTDIR=${OUTDIR:-logs/eval_multitok_combined}
EXP=${EXP:-eval_multitok_combined}
TOTAL_SAMPLES=${TOTAL_SAMPLES:-20}
SIMPLE_SAMPLES=${SIMPLE_SAMPLES:-10}
COMP_SAMPLES=${COMP_SAMPLES:-10}
TOKS=${TOKS:-8}
MAXLEN=${MAXLEN:-128}
TOKENS=${TOKENS:-150}
TEMP=${TEMP:-0.2}
TOPP=${TOPP:-0.8}
SEED=${SEED:-42}
LLM_PREC=${LLM_PREC:-fp16}
SPLIT_CACHE_ROOT=${SPLIT_CACHE_ROOT:-logs/split_cache}
ALLOW_SPLIT_REGEN=${ALLOW_SPLIT_REGEN:-0}

mkdir -p "${OUTDIR}"

RESUME=${RESUME:-$(ls -t logs/train_multitok_ddp/*/llm_multitok_resume_best.pth 2>/dev/null | head -1 || true)}
ARGS=(
  --llm_path "${LLM_PATH}"
  --json_file "${JSON_FILE}"
  --comparative_json_file "${COMPARATIVE_JSON_FILE}"
  --features_file "${FEATURES_FILE}"
  --output_dir "${OUTDIR}"
  --exp_name "${EXP}"
  --num_spectral_features "${TOKS}"
  --max_seq_length "${MAXLEN}"
  --num_samples "${TOTAL_SAMPLES}"
  --num_simple_samples "${SIMPLE_SAMPLES}"
  --num_comparative_samples "${COMP_SAMPLES}"
  --max_new_tokens "${TOKENS}"
  --temperature "${TEMP}"
  --top_p "${TOPP}"
  --random_seed "${SEED}"
  --llm_precision "${LLM_PREC}"
  --split_cache_root "${SPLIT_CACHE_ROOT}"
)

if [[ "${ALLOW_SPLIT_REGEN}" == "1" ]]; then
  ARGS+=(--allow_new_splits)
fi

if [[ -n "${RESUME}" && -f "${RESUME}" ]]; then
  echo "Using resume checkpoint: ${RESUME}"
  srun --kill-on-bad-exit=1 \
    python -u src/eval_combined_samples_multitok.py \
      "${ARGS[@]}" \
      --resume_path "${RESUME}"
else
  echo "No resume checkpoint found; run will evaluate with randomly initialized weights."
  srun --kill-on-bad-exit=1 \
    python -u src/eval_combined_samples_multitok.py \
      "${ARGS[@]}"
fi
