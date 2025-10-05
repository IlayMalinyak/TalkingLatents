#!/bin/bash
# Train multi-token model on topology-aware Q/A with neighbour + physics supervision

#SBATCH --job-name=tl-llm-ddp-topology
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2           # 2 ranks = 2 GPUs
#SBATCH --gres=gpu:2                  # request 2 GPUs
#SBATCH --cpus-per-task=20            # tune for your dataloader
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH -C v100-32g
# Logs
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

mkdir -p logs/slurm

# Environment
module purge
module load pytorch-gpu/py3/2.2.0

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:64"}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# Data/model paths (override with --export=VAR=VALUE)
JSON_FILE=${JSON_FILE:-/lustre/fswork/projects/rech/oxl/utl47bv/data/stellar_descriptions_questions_short.json}
FEATURES_FILE=${FEATURES_FILE:-/lustre/fswork/projects/rech/oxl/utl47bv/data/features.npy}
NEIGHBOR_CACHE=${NEIGHBOR_CACHE:-/lustre/fswork/projects/rech/oxl/utl47bv/cache/topology_neighbors.npz}
LLM_PATH=${LLM_PATH:-/lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Meta-Llama-3.1-8B/original}

# Training knobs (override at submission time)
OUTDIR=${OUTDIR:-logs/train_topology}
EXP=${EXP:-llm_multitok_topology}
EPOCHS=${EPOCHS:-10}
BATCH=${BATCH:-2}              # per-GPU
MAXLEN=${MAXLEN:-128}
WORKERS=${WORKERS:-8}
LR=${LR:-5e-5}
WD=${WD:-1e-3}
TOKS=${TOKS:-4}                # number of spectral tokens (K)
NEIGHBORS=${NEIGHBORS:-8}      # neighbour samples per anchor
LAMBDA_FEAT=${LAMBDA_FEAT:-0.00}
LAMBDA_TEXT=${LAMBDA_TEXT:-0.00}
LAMBDA_RETR=${LAMBDA_RETR:-0.0}
LAMBDA_PHYS=${LAMBDA_PHYS:-0.0}
MAX_ITER=${MAX_ITER:--1}

mkdir -p "${OUTDIR}"

echo "Using topology JSON: ${JSON_FILE}"
echo "Neighbour cache: ${NEIGHBOR_CACHE}"

srun --kill-on-bad-exit=1 \
  python -u src/simple_questions_multitok.py \
    --llm_path "${LLM_PATH}" \
    --json_file "${JSON_FILE}" \
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
    --num_neighbor_samples "${NEIGHBORS}" \
    --neighbor_cache "${NEIGHBOR_CACHE}" \
    --neighbor_metric euclidean \
    --physics_keys Teff logg FeH \
    --llm_precision fp16 \
    --use_amp \
    --gradient_checkpointing \
    --lambda_feat "${LAMBDA_FEAT}" \
    --lambda_text "${LAMBDA_TEXT}" \
    --lambda_retrieval "${LAMBDA_RETR}" \
    --lambda_physics "${LAMBDA_PHYS}" \
    --max_iter "${MAX_ITER}"
