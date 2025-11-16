#!/bin/bash
#SBATCH --job-name=cap_adv_100k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --qos=basic

INPUT=/data/TalkingLatents/logs/2025-07-29/info.csv
OUTPUT=/data/TalkingLatents/data/dataset/caption_advanced_100k.json

srun --kill-on-bad-exit=1 \
 --container-image="/rg/perets_prj/ilay.kamai/containers/astro2.sqsh" \
 --mem=64G \
 --output=/home/ilay.kamai/athena/logs/caption_advanced_%j_%s_%t.out \
 --error=/home/ilay.kamai/athena/logs/caption_advanced_%j_%s_%t.err \
 --no-container-entrypoint \
 --container-mounts=/home/ilay.kamai/work:/data \
 python -u /data/TalkingLatents/data/caption_advanced.py \
   --input "${INPUT}" \
   --output "${OUTPUT}" \
   --limit 100000 \
   --processes 16
