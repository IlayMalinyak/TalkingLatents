#!/bin/bash
#SBATCH --job-name=TL_late_fusion
#SBATCH --nodes=1
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --partition=h200-shared
#SBATCH --qos=24h_4g
#SBATCH --time=24:00:00

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=$((29500 + ${SLURM_JOB_ID} % 1000))

export APPTAINER_BIND="$PRJ_WORKSPACE"

# NCCL timeout settings to prevent hanging
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1200
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=10000
export NCCL_DEBUG=INFO

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
     --output=/home/ilay.kamai/athena/logs/%x_%j_%N_%t.out \
     --error=/home/ilay.kamai/athena/logs/%x_%j_%N_%t.err \
     --container=/rg/perets_prj/ilay.kamai/containers/torch2.7-tf2.20-cu128.sif \
     python -m torch.distributed.run \
       --nnodes=${SLURM_NNODES:-1} \
       --nproc_per_node=${SLURM_GPUS_ON_NODE:-4} \
       --node_rank=${SLURM_NODEID:-0} \
       --master_addr=${MASTER_ADDR} \
       --master_port=${MASTER_PORT} \
       /home/ilay.kamai/work/TalkingLatents/src/late_fusion.py \
