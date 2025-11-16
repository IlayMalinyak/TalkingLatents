#!/bin/bash
#SBATCH --job-name=TL_infer_ablate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=basic
#SBATCH --mem=64G

set -euo pipefail

BASE_LOG_DIR=${BASE_LOG_DIR:-/data/TalkingLatents/logs}
LLM_ROOT=${LLM_ROOT:-/data/.llama}
EXPERIMENTS=(
  abl_single_lambda1
  abl_single_lambda0p5
  abl_combined_lambda1
  abl_combined_lambda0p5
  abl_advanced_lambda1
  abl_advanced_lambda0p5
)

echo "Running inference for experiments: ${EXPERIMENTS[*]}"

for EXP in "${EXPERIMENTS[@]}"; do
  echo "=== Processing ${EXP} ==="
  CKPT_PATH=$(find "${BASE_LOG_DIR}" -maxdepth 2 -type f -name "${EXP}.pth" | sort | tail -n 1 || true)
  if [[ -z "${CKPT_PATH}" ]]; then
    echo "Warning: checkpoint for ${EXP} not found under ${BASE_LOG_DIR}; skipping."
    continue
  fi

  CKPT_DIR=$(dirname "${CKPT_PATH}")
  OUT_DIR="${CKPT_DIR}/inference_${EXP}"

  echo "Checkpoint: ${CKPT_PATH}"
  echo "Output dir: ${OUT_DIR}"

  srun --kill-on-bad-exit=1 \
    --container-image="/rg/perets_prj/ilay.kamai/containers/astro2.sqsh" \
    --mem=64G \
    --output=/home/ilay.kamai/athena/logs/TalkingLatents_infer_%j_%s_%t.out \
    --error=/home/ilay.kamai/athena/logs/TalkingLatents_infer_%j_%s_%t.err \
    --no-container-entrypoint \
    --container-mounts=/home/ilay.kamai/work:/data \
    python -u /data/TalkingLatents/src/inference.py \
      --checkpoint_path "${CKPT_PATH}" \
      --output_dir "${OUT_DIR}" \
      --interpolation_num_pairs 10 \
      --interpolation_min_teff_diff 800 \
      --llm_root "${LLM_ROOT}"
done

echo "Inference sweep complete."
