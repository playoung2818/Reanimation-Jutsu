#!/usr/bin/env bash
set -euo pipefail

# Requires a CUDA-capable GPU and the dependencies in requirements_qwen.txt.
python train_qwen_lora.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --train_file example_data/qianzhongshu_train.jsonl \
  --eval_file example_data/qianzhongshu_eval.jsonl \
  --output_dir saved_models/qwen2_5_weicheng_lora \
  --max_length 1024 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --save_steps 100 \
  --logging_steps 10
