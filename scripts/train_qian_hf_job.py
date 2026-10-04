# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "accelerate==1.13.0",
#   "bitsandbytes>=0.49,<0.50",
#   "datasets==4.8.4",
#   "huggingface-hub>=1.10,<2",
#   "peft==0.18.1",
#   "torch==2.8.0",
#   "transformers==5.5.3",
# ]
# ///
"""Train a private Qwen rewrite adapter from pinned curated dataset files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATASET_ID = "Playoung2818/qianzhongshu-curated-rewrite-data"
MODEL_ID = "Playoung2818/qianzhongshu-qwen2.5-7b-curated-lora-v1"


def build_prompt(instruction: str, inp: str) -> str:
    # Keep the same format as the local trainer and the Space's rewrite tab.
    return f"### 指令:\n{instruction}\n\n### 输入:\n{inp}\n\n### 输出:\n"


def encode_row(row: dict[str, str], tokenizer, max_length: int = 768) -> dict[str, list[int]]:
    """Train only on the unchanged output and its stop token, never the prompt."""
    for field in ("instruction", "input", "output"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"Missing or empty {field}")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no end-of-sequence token")
    prompt_ids = tokenizer.encode(build_prompt(row["instruction"], row["input"]), add_special_tokens=False)
    output_ids = tokenizer.encode(row["output"], add_special_tokens=False) + [tokenizer.eos_token_id]
    input_ids = prompt_ids + output_ids
    if len(input_ids) > max_length:
        raise ValueError(f"Example has {len(input_ids)} tokens, exceeds {max_length}; refusing to truncate the original")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + output_ids,
    }


def check_splits(dataset) -> None:
    if len(dataset["train"]) != 285 or len(dataset["validation"]) != 25:
        raise ValueError("Expected exactly 285 training and 25 validation rows")
    seen_inputs: set[str] = set()
    seen_outputs: set[str] = set()
    for split in ("train", "validation"):
        for row in dataset[split]:
            if set(row) != {"instruction", "input", "output"}:
                raise ValueError("Dataset must have instruction/input/output fields only")
            if row["input"] in seen_inputs or row["output"] in seen_outputs:
                raise ValueError("Duplicate input or output across dataset splits")
            seen_inputs.add(row["input"])
            seen_outputs.add(row["output"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dataset-revision", required=True, help="Immutable dataset commit SHA")
    parser.add_argument("--base-revision", required=True, help="Immutable base model commit SHA")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-dir", default="qian-curated-lora")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed,
    )

    token = os.environ["HF_TOKEN"]
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires an NVIDIA GPU with CUDA support")
    set_seed(2818)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Dataset: {args.dataset_id}@{args.dataset_revision}", flush=True)
    print(f"Output adapter: {args.model_id}", flush=True)
    files = {
        split: hf_hub_download(
            args.dataset_id, filename, repo_type="dataset",
            revision=args.dataset_revision, token=token,
        )
        for split, filename in [("train", "train.jsonl"), ("validation", "validation.jsonl")]
    }
    dataset = load_dataset("json", data_files=files)
    check_splits(dataset)
    print("Verified 285 training rows and 25 validation rows", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=args.base_revision, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded = dataset.map(
        lambda row: encode_row(row, tokenizer),
        remove_columns=dataset["train"].column_names,
    )
    print(f"Maximum example length: {max(len(row['input_ids']) for split in encoded.values() for row in split)} tokens", flush=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=args.base_revision, token=token,
        dtype=dtype, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
        ),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    trainer = Trainer(
        model=model, processing_class=tokenizer,
        train_dataset=encoded["train"], eval_dataset=encoded["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8),
        args=TrainingArguments(
            output_dir=args.output_dir, num_train_epochs=args.epochs,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=16, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=2e-4, warmup_ratio=0.03, lr_scheduler_type="cosine",
            bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
            logging_steps=5, logging_first_step=True,
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
            optim="adamw_torch", seed=2818, data_seed=2818,
            report_to="none", push_to_hub=False,
        ),
    )
    result = trainer.train()
    metrics = trainer.evaluate()
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}", flush=True)
    print(f"Validation metrics: {metrics}", flush=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.state.save_to_json(str(Path(args.output_dir) / "trainer_state.json"))
    manifest = {
        "base_model": BASE_MODEL, "base_revision": args.base_revision,
        "dataset_id": args.dataset_id, "dataset_revision": args.dataset_revision,
        "train_rows": 285, "validation_rows": 25, "epochs": args.epochs,
        "seed": 2818, "format": "instruction/input/output headings; output-only loss with EOS",
        "file_sha256": {split: hashlib.sha256(Path(path).read_bytes()).hexdigest() for split, path in files.items()},
        "train_metrics": result.metrics, "validation_metrics": metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
    }
    (Path(args.output_dir) / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for filename in ["adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json"]:
        if not (Path(args.output_dir) / filename).is_file():
            raise RuntimeError(f"Adapter export is missing {filename}")
    api = HfApi(token=token)
    api.create_repo(args.model_id, repo_type="model", private=True, exist_ok=True)
    if not api.model_info(args.model_id).private:
        raise RuntimeError("Refusing to upload the adapter to a public repository")
    api.upload_folder(
        repo_id=args.model_id, repo_type="model", folder_path=args.output_dir,
        ignore_patterns=["checkpoint-*/*"],
        commit_message="Train curated modern-to-original Qwen rewrite adapter",
    )
    print(f"Saved private adapter: https://huggingface.co/{args.model_id}", flush=True)


if __name__ == "__main__":
    main()
