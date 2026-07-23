#!/usr/bin/env bash
#
# Stage 2: fine-tune BGE-M3 on the data produced by `scitopic train`.
#
# This is a thin wrapper over FlagEmbedding's m3 encoder fine-tuning entry point.
# Adjust CUDA_VISIBLE_DEVICES / --nproc_per_node to match your hardware, and
# --query_max_len / --passage_max_len to the values suggested by `scitopic train`.
#
# Usage:
#   bash scripts/finetune.sh
#
set -euo pipefail

export OMP_NUM_THREADS=1

# --- Configurable paths --------------------------------------------------------
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-2}"
MODEL_PATH="${MODEL_PATH:-pretrained_model/bge-m3}"
TRAIN_DATA="${TRAIN_DATA:-output/fine_tune/AI-DM/finetune_data.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-finetune_result}"
QUERY_MAX_LEN="${QUERY_MAX_LEN:-64}"
PASSAGE_MAX_LEN="${PASSAGE_MAX_LEN:-64}"
# ------------------------------------------------------------------------------

CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --nproc_per_node "${NPROC}" \
  -m FlagEmbedding.finetune.embedder.encoder_only.m3 \
  --model_name_or_path "${MODEL_PATH}" \
  --train_data "${TRAIN_DATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --cache_dir ./cache/model \
  --cache_path ./cache/data \
  --train_group_size 8 \
  --query_max_len "${QUERY_MAX_LEN}" \
  --passage_max_len "${PASSAGE_MAX_LEN}" \
  --pad_to_multiple_of 8 \
  --knowledge_distillation True \
  --same_dataset_within_batch True \
  --small_threshold 0 \
  --drop_threshold 0 \
  --learning_rate 1e-5 \
  --fp16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --dataloader_drop_last True \
  --warmup_ratio 0.1 \
  --logging_steps 1 \
  --save_steps 1000 \
  --negatives_cross_device \
  --temperature 0.02 \
  --sentence_pooling_method cls \
  --normalize_embeddings True \
  --kd_loss_type m3_kd_loss \
  --unified_finetuning True \
  --use_self_distill True \
  --fix_encoder False \
  --self_distill_start_step 0
