#!/usr/bin/env python3
"""Inference script for Qwen2.5 + LoRA style rewrite."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--lora_path", required=True)
    p.add_argument(
        "--instruction",
        default=(
            "请将下面这句话或短段文字改写成钱钟书《围城》式的中文。"
            "必须保持原意，不要回答问题，不要解释，不要补充新事实。"
            "语言应有讽刺、机智、比喻和冷峭的观察。"
        ),
    )
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--input_path", default=None)
    p.add_argument("--output_path", default=None)
    p.add_argument("--interactive", action="store_true")
    return p.parse_args()


def build_prompt(instruction: str, text: str) -> str:
    return (
        "### 指令:\n"
        f"{instruction}\n\n"
        "### 输入:\n"
        f"{text.strip()}\n\n"
        "### 输出:\n"
    )


def load_model(base: str, lora: str):
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:  
        device = "cpu" #Tring to train on a Raspberry Pi
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, lora)
    model = model.to(device)
    model.eval()
    return tokenizer, model #tokenizer (text to number) -> model -> tokenizer(numbder to text)


def generate_one(tokenizer, model, instruction: str, text: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    prompt = build_prompt(instruction, text)
    inputs = tokenizer(prompt, return_tensors="pt") #tokenizer (text to number) 
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
    print("Raw output:", out)

    # The output includes your prompt. Select only the newly generated tokens.
    new_tokens = out[0, inputs["input_ids"].shape[1]:]

    print("New token IDs:", new_tokens.tolist())
    print("Rewritten text:", tokenizer.decode(new_tokens, skip_special_tokens=True))
    generated = tokenizer.decode(out[0], skip_special_tokens=True) #tokenizer(numbder to text)
    return generated[len(prompt) :].strip()


def main() -> int:
    args = parse_args()
    tokenizer, model = load_model(args.model_name_or_path, args.lora_path)

    if args.interactive:
        print("Interactive mode. Type empty line to exit.")
        while True:
            text = input("Input> ").strip()
            if not text:
                break
            ans = generate_one(
                tokenizer, model, args.instruction, text, args.max_new_tokens, args.temperature, args.top_p
            )
            print(f"Output> {ans}\n")
        return 0

    if not args.input_path:
        raise ValueError("input_path is required for non-interactive mode")

    lines = [x.strip() for x in Path(args.input_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    outputs = []
    for line in lines:
        out = generate_one(
            tokenizer, model, args.instruction, line, args.max_new_tokens, args.temperature, args.top_p
        )
        outputs.append(f"Instruction: {args.instruction}\nInput: {line}\nOutput: {out}\n")

    if args.output_path:
        Path(args.output_path).write_text("\n".join(outputs), encoding="utf-8")
    else:
        print("\n".join(outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

