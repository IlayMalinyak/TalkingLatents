#!/bin/bash
#SBATCH --job-name=TL_ablation
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:2
#SBATCH --qos=basic
#SBATCH --mem=64G

srun --kill-on-bad-exit=1 \
  --container-image="/rg/perets_prj/ilay.kamai/containers/astro2.sqsh" \
  --mem=64G \
  --output=/home/ilay.kamai/athena/logs/%x_%j_%s_%t.out \
  --error=/home/ilay.kamai/athena/logs/%x_%j_%s_%t.err \
  --no-container-entrypoint \
  --container-mounts=/home/ilay.kamai/work:/data \
  python -u /data/TalkingLatents/src/training_ablation.py