#!/bin/bash
#SBATCH --job-name=TL_late_fusion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=basic
#SBATCH --begin=now

# Set up distributed training environment
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Use job-specific port to avoid clashes between simultaneous torchrun jobs
export MASTER_PORT=$((12000 + SLURM_JOB_ID % 20000))
# CRITICAL: Set WORLD_SIZE to total number of processes
export WORLD_SIZE=$SLURM_NTASKS
export GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-1}
export NNODES=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}

srun --container-image="/rg/perets_prj/ilay.kamai/containers/astro2.sqsh" \
     --mpi=pmix \
     --output=/home/ilay.kamai/athena/logs/%x_%j_%s_%t.out \
     --error=/home/ilay.kamai/athena/logs/%x_%j_%s_%t.err \
     --no-container-entrypoint \
     --container-mounts=/home/ilay.kamai/work:/data \
     torchrun \
        --nproc_per_node=$GPUS_PER_NODE \
        --nnodes=$NNODES \
        --node_rank=$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        /data/TalkingLatents/src/late_fusion.py
