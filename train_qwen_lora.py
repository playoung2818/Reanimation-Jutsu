#!/usr/bin/env python3
"""QLoRA training script for Qwen2.5 style transfer."""
from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


@dataclass
class Row:
    instruction: str
    input: str
    output: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--train_file", required=True, help="jsonl with instruction/input/output")
    p.add_argument("--eval_file", default=None, help="optional jsonl with same schema")
    p.add_argument("--output_dir", default="saved_models/qwen2_5_lora")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--bf16", action="store_true")
    p.add_argument(
        "--train_on_prompt",
        action="store_true",
        help="Also compute loss on the instruction/input prompt. Default trains only on the target output.",
    )
    return p.parse_args()


def build_prompt(instruction: str, inp: str) -> str:
    return (
        "### 指令:\n"
        f"{instruction}\n\n"
        "### 输入:\n"
        f"{inp}\n\n"
        "### 输出:\n"
    )


def build_text(instruction: str, inp: str, out: str, eos_token: str | None = None) -> str:
    text = f"{build_prompt(instruction, inp)}{out}"
    if eos_token:
        text += eos_token
    return text


def main() -> int:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quant_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file
    ds = load_dataset("json", data_files=data_files)

    def preprocess(batch):
        texts = [
            build_text(inst, inp, out, tokenizer.eos_token)
            for inst, inp, out in zip(batch["instruction"], batch["input"], batch["output"])
        ]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

        labels = [ids.copy() for ids in tokenized["input_ids"]]
        if not args.train_on_prompt:
            prompts = [
                build_prompt(inst, inp)
                for inst, inp in zip(batch["instruction"], batch["input"])
            ]
            prompt_tokenized = tokenizer(
                prompts,
                truncation=True,
                max_length=args.max_length,
                padding=False,
            )
            for row_labels, prompt_ids in zip(labels, prompt_tokenized["input_ids"]):
                prompt_len = min(len(prompt_ids), len(row_labels))
                row_labels[:prompt_len] = [-100] * prompt_len
        tokenized["labels"] = labels
        return tokenized

    remove_cols = ds["train"].column_names
    tokenized = ds.map(preprocess, batched=True, remove_columns=remove_cols)

    train_dataset = tokenized["train"]
    eval_dataset = tokenized["validation"] if "validation" in tokenized else None

    training_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=args.bf16,
        fp16=not args.bf16,
        report_to=[],
    )
    strategy_value = "steps" if eval_dataset is not None else "no"
    try:
        training_args = TrainingArguments(
            **training_kwargs,
            eval_strategy=strategy_value,
        )
    except TypeError as exc:
        if "eval_strategy" not in str(exc):
            raise
        training_args = TrainingArguments(
            **training_kwargs,
            evaluation_strategy=strategy_value,
        )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding=True,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
