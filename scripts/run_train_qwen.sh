#!/usr/bin/env bash
set -euo pipefail

# Example:
# CUDA_VISIBLE_DEVICES=0 bash scripts/run_train_qwen.sh

# Fail before downloading the model if this machine cannot run CUDA training.
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Training requires an NVIDIA GPU with CUDA support.")
PY

# Use the curated modern-input/original-output pairs, not automatic clause deletion.
python scripts/prepare_curated_weicheng_dataset.py

python train_qwen_lora.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --train_file example_data/qianzhongshu_rewrite_train.jsonl \
  --eval_file example_data/qianzhongshu_rewrite_eval.jsonl \
  --output_dir saved_models/qwen2_5_qian_rewrite_lora \
  --max_length 768 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --save_steps 10 \
  --logging_steps 5 \
  --eval_steps 10

