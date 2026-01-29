#!/bin/bash
#SBATCH --job-name=TL_single_l05
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:2
#SBATCH --qos=basic
#SBATCH --mem=64G

# Base paths
JSON_FILE=/data/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json
OUTDIR=${OUTDIR:-/data/TalkingLatents/logs}
LLM_PATH=/data/.llama/Llama3.1-8B

# Training hyper-parameters
EXP=${EXP:-abl_single_lambda0p5}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-8}
MAXLEN=${MAXLEN:-256}
WORKERS=${WORKERS:-4}
LR=${LR:-1e-4}
WD=${WD:-1e-3}
TOKS=${TOKS:-8}
LAMBDA=${LAMBDA:-0.5}

srun --kill-on-bad-exit=1 \
  --container-image="/rg/perets_prj/ilay.kamai/containers/astro2.sqsh" \
  --mem=64G \
  --output=/home/ilay.kamai/athena/logs/TalkingLatents_%j_%s_%t.out \
  --error=/home/ilay.kamai/athena/logs/TalkingLatents_%j_%s_%t.err \
  --no-container-entrypoint \
  --container-mounts=/home/ilay.kamai/work:/data \
  python -u /data/TalkingLatents/src/simple_questions_multitok.py \
    --llm_path "${LLM_PATH}" \
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
    --llm_precision fp16 \
    --use_amp \
    --gradient_checkpointing \
    --disable_classification \
    --predict_stellar_params \
    --max_iter -1 \
