#!/bin/bash

  # Qwen 32B follow-up training on a single 40GB-class GPU using 4-bit quantization

  # Derived from cluster/followup_train.sbatch with HF/Qwen replacements

  #SBATCH --job-name=TL_fup_qwen32
  #SBATCH --nodes=1
  #SBATCH --ntasks=1
  #SBATCH --ntasks-per-node=1
  #SBATCH --cpus-per-task=8
  #SBATCH --gpus-per-node=a100:2
  #SBATCH --qos=basic

  # Optional: Hugging Face caches on fast storage

  export HF_HOME=${HF_HOME:-/data/.cache/huggingface}
  export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
  export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
  export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}

  # Base paths

  JSON_FILE=/data/TalkingLatents/data/dataset/stellar_descriptions_questions.json
  OUTDIR=${OUTDIR:-/data/TalkingLatents/logs}

  # Training hyper-parameters

  EXP=${EXP:-abl_single_qwen32_4bit}
  EPOCHS=${EPOCHS:-30}
  BATCH=${BATCH:-8}
  MAXLEN=${MAXLEN:-256}
  WORKERS=${WORKERS:-4}
  LR=${LR:-1e-4}
  WD=${WD:-1e-3}
  TOKS=${TOKS:-8}
  LAMBDA=${LAMBDA:-1.0}

  srun --kill-on-bad-exit=1 \
  --container-image="/rg/perets_prj/ilay.kamai/containers/astro2.sqsh" \
  --mem=64G \
  --output=/home/ilay.kamai/athena/logs/%x_%j_%s_%t.out \
  --error=/home/ilay.kamai/athena/logs/%x_%j_%s_%t.err \
  --no-container-entrypoint \
  --container-mounts=/home/ilay.kamai/work:/data \
  python -u /data/TalkingLatents/src/simple_questions_multitok.py \
  --llm_backend qwen \
  --hf_model_name Qwen/Qwen2.5-32B-Instruct \
  --hf_quantization 4bit \
  --hf_max_memory_gb 38 \
  --hf_device_map auto \
  --hf_trust_remote_code \
  --hf_max_memory_gb 38 \
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
  --llm_precision bf16 \
  --use_amp \
  --gradient_checkpointing \
  --disable_classification \
  --enable_followup_augmentation \
  --followup_prob 0.7 \
  --max_followup_turns 2