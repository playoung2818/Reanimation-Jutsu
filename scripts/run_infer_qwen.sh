#!/usr/bin/env bash
set -euo pipefail

# Example:
# CUDA_VISIBLE_DEVICES=0 bash scripts/run_infer_qwen.sh

python inference_qwen_lora.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --lora_path saved_models/qwen2_5_qian_rewrite_lora \
  --instruction 请将下面这句话或短段文字改写成钱钟书《围城》式的中文。必须保持原意，不要回答问题，不要解释，不要补充新事实。语言应有讽刺、机智、比喻和冷峭的观察。 \
  --input_path test_data/test.txt \
  --output_path test_data/output_qwen.txt
