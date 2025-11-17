#!/bin/bash
#SBATCH --job-name=TL_fup_qwen
#SBATCH --nodes=1
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=l40s-shared
#SBATCH --qos=72h_8g
#SBATCH -t 72:00:00

set -euo pipefail

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500

export APPTAINER_BIND="$PRJ_WORKSPACE:/data"

# export TMPDIR=work/tmp
# export TRANSFORMERS_CACHE=work/.cache/hf
# export PIP_CACHE_DIR=work/.cache/pip
# mkdir -p $TMPDIR $TRANSFORMERS_CACHE $PIP_CACHE_DIR

export HF_HOME=${HF_HOME:-/data/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}

# Base paths

JSON_FILE=/data/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json
OUTDIR=${OUTDIR:-/data/TalkingLatents/logs}

# Training hyper-parameters

EXP=${EXP:-abl_single_qwen32_4bit_no_fup}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-8}
MAXLEN=${MAXLEN:-128}
WORKERS=${WORKERS:-4}
LR=${LR:-1e-4}
WD=${WD:-1e-3}
TOKS=${TOKS:-8}
LAMBDA=${LAMBDA:-1.0}


srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    --output=/home/ilay.kamai/athena/logs/%x_%j_%N_%t.out \
  --error=/home/ilay.kamai/athena/logs/%x_%j_%N_%t.err \
  --container=/rg/perets_prj/ilay.kamai/containers/torch2.7-tf2.20-cu128.sif \
  python -u /data/TalkingLatents/src/simple_questions_multitok.py \
  --llm_backend qwen \
  --hf_model_name Qwen/Qwen2.5-32B-Instruct \
  --hf_quantization 4bit \
  --hf_max_memory_gb 38 \
  --hf_device_map auto \
  --hf_trust_remote_code \
  --hf_max_memory_gb 38 \
  --json_file "${JSON_FILE}" \
  --mode single_star \
  --single_dataset_type regular \
  --output_dir "${OUTDIR}" \
  --exp_name "${EXP}" \
  --batch_size "${BATCH}" \
  --num_epochs "${EPOCHS}" \
  --learning_rate "${LR}" \
  --weight_decay "${WD}" \
  --max_seq_length "${MAXLEN}" \
  --num_workers "${WORKERS}" \
  --num_spectral_features "${TOKS}" \
  --loss_lambda "${LAMBDA}" \
  --llm_precision bf16 \
  --use_amp \
  --gradient_checkpointing \
  --disable_classification \
  --enable_followup_augmentation \
  --followup_prob 0.7 \
  --max_followup_turns 2
