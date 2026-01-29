#!/bin/bash
#SBATCH --job-name=abl_tokens
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=l40s-shared
#SBATCH --qos=24h_4g
#SBATCH -t 24:00:00
#SBATCH --output=/home/ilay.kamai/athena/logs/ablation_tokens_%j.out
#SBATCH --error=/home/ilay.kamai/athena/logs/ablation_tokens_%j.err

set -euo pipefail

echo "=================================="
echo "Ablation Study: num_spectral_features"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================="

export APPTAINER_BIND="$PRJ_WORKSPACE"

# Set master address for distributed training (even though we use single GPU)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=$(expr 29500 + $(echo -n $SLURM_JOBID | tail -c 4))

srun --ntasks=1 --nodes=1 --cpus-per-task=32 \
    --output=/home/ilay.kamai/athena/logs/ablation_tokens_%j.out \
    --error=/home/ilay.kamai/athena/logs/ablation_tokens_%j.err \
    --container=/apps/apptainer/images/pytorch/latest/torch2.7-tf2.20-cu128.sif \
    torchrun --nproc_per_node=$SLURM_NTASKS \
        --nnodes=1 \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        --rdzv_backend=c10d \
        /home/ilay.kamai/work/TalkingLatents/src/ablation_exp.py

echo "=================================="
echo "Ablation study completed!"
echo "End time: $(date)"
echo "=================================="
