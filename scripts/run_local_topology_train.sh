#!/bin/bash
# Quick local smoke test for topology-aware training on a small subset.
# Assumes you're in the repo root and have activated the tfenv23 conda env.

set -euo pipefail

DATA_JSON="data/stellar_topology_questions_small.json"
META_JSON="data/stellar_topology_neighbors_small.json"

if [ ! -f "$DATA_JSON" ]; then
  echo "Generating small topology dataset..."
  python3 scripts/make_topology_questions.py \
    --src data/stellar_descriptions_questions_short.json \
    --features data/features.npy \
    --out "$DATA_JSON" \
    --metadata_out "$META_JSON" \
    --k 4 --seed 0 --max_items 1000 --chain_steps 2
fi

python3 src/simple_questions_multitok.py \
  --json_file "$DATA_JSON" \
  --features_file data/features.npy \
  --exp_name local_topology_test \
  --output_dir logs/local_topology \
  --batch_size 1 \
  --num_epochs 1 \
  --max_iter 50 \
  --learning_rate 5e-5 \
  --num_spectral_features 4 \
  --num_neighbor_samples 6 \
  --lambda_feat 0.05 \
  --lambda_text 0.05 \
  --lambda_retrieval 0.3 \
  --lambda_physics 0.1 \
  --physics_keys Teff logg FeH \
  --use_dummy_llm \
  --use_amp \
  --gradient_checkpointing \
  --train True
