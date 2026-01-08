#!/bin/bash
#SBATCH --job-name=TL_qwen_inference
#SBATCH --nodes=1
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G                    # Increased from 128G to handle 32B model loading
#SBATCH --partition=h200-shared
#SBATCH --qos=12h_4g
#SBATCH -t 12:00:00

set -euo pipefail

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500

export APPTAINER_BIND="$PRJ_WORKSPACE"

# export TMPDIR=work/tmp
# export TRANSFORMERS_CACHE=work/.cache/hf
# export PIP_CACHE_DIR=work/.cache/pip
# mkdir -p $TMPDIR $TRANSFORMERS_CACHE $PIP_CACHE_DIR

# Use workspace for cache to ensure it's accessible inside container
export HF_HOME=/rg/perets_prj/ilay.kamai/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE $HUGGINGFACE_HUB_CACHE

# Base paths

JSON_FILE=/home/ilay.kamai/work/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json
JSON_FOLLOWUP=/home/ilay.kamai/work/TalkingLatents/data/dataset/stellar_descriptions_questions.json

OUTDIR=${OUTDIR:-/home/ilay.kamai/work/TalkingLatents/logs}

# Training hyper-parameters

EXP=${EXP:-qwen32_fup_lora_6}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-8}
MAXLEN=${MAXLEN:-256}
WORKERS=${WORKERS:-4}
LR=${LR:-1e-4}
WD=${WD:-1e-3}
TOKS=${TOKS:-8}
LAMBDA=${LAMBDA:-1.0}


srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    --output=/home/ilay.kamai/athena/logs/%x_%j_%N_%t.out \
  --error=/home/ilay.kamai/athena/logs/%x_%j_%N_%t.err \
  --container=/apps/apptainer/images/pytorch/latest/torch2.7-tf2.20-cu128.sif \
  python -u /home/ilay.kamai/work/TalkingLatents/src/follow_up_inference.py \
  --checkpoint_path /home/ilay.kamai/work/TalkingLatents/logs/regular_loss1p0_2026-01-03-16-07/fup_qwen32_lora_6_resume_best.pth \
  --max_samples 100
