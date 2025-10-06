#!/bin/bash
#SBATCH --job-name=TalkingLatents
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:2
#SBATCH --qos=basic
#SBATCH --mem=64G
#SBATCH --hint=nomultithread

# Allocator settings optimized for multi-GPU memory efficiency
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:32,expandable_segments:True"}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export OMP_NUM_THREADS=2
export CUDA_LAUNCH_BLOCKING=0

# Data/model paths (edit to your filesystem)
JSON_FILE=/data/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json
COMPARATIVE_JSON_FILE=/data/TalkingLatents/data/dataset/comparative_dataset.json
MODE=combined
FEATURES_FILE=/data/TalkingLatents/logs/2025-07-29/features.npy
LLM_PATH=/data/.llama/Llama3.1-8B

# Training knobs (override at submit time with --export=VAR=VALUE)
OUTDIR=${OUTDIR:-logs/train_ddp}
EXP=${EXP:-llm_multitok}
EPOCHS=${EPOCHS:-20}
BATCH=${BATCH:-4}              # per-GPU
MAXLEN=${MAXLEN:-128}
WORKERS=${WORKERS:-4}
LR=${LR:-1e-4}
WD=${WD:-1e-3}
TOKS=${TOKS:-4}                # number of spectral tokens (K)
SWITCH=${SWITCH:-3}

mkdir -p "${OUTDIR}"

srun --kill-on-bad-exit=1 \
 --container-image="/rg/perets_prj/ilay.kamai/containers/astro.sqsh" \
     --mem=64G \
     --output=/home/ilay.kamai/athena/logs/TalkingLatents_%j_%s_%t.out \
     --error=/home/ilay.kamai/athena/logs/TalkingLatents_%j_%s_%t.err \
     --no-container-entrypoint \
     --container-mounts=/home/ilay.kamai/work:/data \
  python -u /data/TalkingLatents/src/simple_questions_multitok.py \
    --llm_path "${LLM_PATH}" \
    --json_file "${JSON_FILE}" \
    --comparative_json_file "${COMPARATIVE_JSON_FILE}" \
    --mode "${MODE}" \
    --switch_epoch "${SWITCH}" \
    --features_file "${FEATURES_FILE}" \
    --output_dir "${OUTDIR}" \
    --exp_name "${EXP}" \
    --batch_size "${BATCH}" \
    --num_epochs "${EPOCHS}" \
    --learning_rate "${LR}" \
    --weight_decay "${WD}" \
    --max_seq_length "${MAXLEN}" \
    --num_workers "${WORKERS}" \
    --num_spectral_features "${TOKS}" \
    --llm_precision fp16 \
    --use_amp \
    --gradient_checkpointing \
    --max_iter -1
