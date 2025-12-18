#!/bin/bash
#SBATCH --job-name=TL_fup_1b
#SBATCH --nodes=1
#SBATCH --ntasks=4                   
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=l40s-shared
#SBATCH --qos=72h_8g
#SBATCH -t 72:00:00

set -euo pipefail

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500

export APPTAINER_BIND="$PRJ_WORKSPACE"

JSON_FILE=/home/ilay.kamai/work/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json
JSON_FOLLOWUP=/home/ilay.kamai/work/TalkingLatents/data/dataset/stellar_descriptions_questions.json

OUTDIR=${OUTDIR:-/home/ilay.kamai/work/TalkingLatents/logs}
LLM_PATH=/home/ilay.kamai/work/.llama/Llama3.2-1B

# Training hyper-parameters
EXP=${EXP:-fup_single_none_lora}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-8}
MAXLEN=${MAXLEN:-256}
WORKERS=${WORKERS:-4}
LR=${LR:-1e-4}
WD=${WD:-1e-3}
TOKS=${TOKS:-8}
LAMBDA=${LAMBDA:-1.0}

srun --ntasks=$SLURM_NNODES --ntasks-per-node=4 \
    --output=/home/ilay.kamai/athena/logs/%x_%j_%N_%t.out \
  --error=/home/ilay.kamai/athena/logs/%x_%j_%N_%t.err \
  --container=/rg/perets_prj/ilay.kamai/containers/torch2.7-tf2.20-cu128.sif \
  python -u /home/ilay.kamai/work/TalkingLatents/src/simple_questions_multitok.py \
        --llm_path "${LLM_PATH}" \
        --json_file "${JSON_FILE}" \
        --followup_json_file "${JSON_FOLLOWUP}" \
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
        --llm_precision fp16 \
        --use_amp \
        --gradient_checkpointing \
        --disable_classification \
        --enable_followup_augmentation  \
        --followup_prob 0.7  \
        --max_followup_turns 2 \
    