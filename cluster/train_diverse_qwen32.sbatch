#!/bin/bash
#SBATCH --job-name=TL_diverse_qwen32
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:4
#SBATCH --qos=basic
#SBATCH --mem=180G

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=$((29500 + ${SLURM_JOB_ID} % 1000))

export HF_HOME=/rg/perets_prj/ilay.kamai/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE $HUGGINGFACE_HUB_CACHE

OUTDIR=${OUTDIR:-/home/ilay.kamai/work/TalkingLatents/logs}
LLM_PATH=/home/ilay.kamai/work/.llama/Llama3.1-8B

# Training hyper-parameters
EXP=${EXP:-diverse_qa}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-8}
MAXLEN=${MAXLEN:-256}
WORKERS=${WORKERS:-4}
LR=${LR:-1e-4}
WD=${WD:-1e-3}
TOKS=${TOKS:-8}
LAMBDA=${LAMBDA:-1.0}

srun --kill-on-bad-exit=1 \
  --container-image='/home/ilay.kamai/work/containers/astro2.sqsh' \
  --mem=64G \
  --output=/home/ilay.kamai/athena/logs/%x_%j_%s_%t.out \
  --error=/home/ilay.kamai/athena/logs/%x_%j_%s_%t.err \
  --no-container-entrypoint \
  --container-mounts='/home/ilay.kamai/work:/home/ilay.kamai/work' \
    python -u /home/ilay.kamai/work/TalkingLatents/src/train_diverse_qa.py \
        --llm_backend qwen \
        --hf_model_name Qwen/Qwen2.5-32B-Instruct \
        --hf_quantization 4bit \
        --hf_max_memory_gb 30 \
        --hf_device_map '{"": 0}' \
        --hf_trust_remote_code \
        --mode single_star \
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
        --enable_followup_augmentation  \
        --followup_prob 1.0  \
        --max_followup_turns 2