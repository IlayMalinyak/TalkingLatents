#!/bin/bash
#SBATCH --job-name=caption_diverse
#SBATCH --nodes=1
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --partition=a100-public
#SBATCH --qos=24h_4g
#SBATCH -t 24:00:00

set -euo pipefail

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500

export APPTAINER_BIND="$PRJ_WORKSPACE"

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    --output=/home/ilay.kamai/athena/logs/%x_%j_%N_%t.out \
  --error=/home/ilay.kamai/athena/logs/%x_%j_%N_%t.err \
  --container=/rg/perets_prj/ilay.kamai/containers/torch2.7-tf2.20-cu128.sif \
  python -u /home/ilay.kamai/work/TalkingLatents/data/caption_diverse.py
        