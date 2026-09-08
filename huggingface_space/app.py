import os
from pathlib import Path

import gradio as gr
import spaces
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
QIAN_ADAPTER = str(Path(__file__).resolve().parent)
LINCOLN_ADAPTER = "Playoung2818/lincoln-qwen2.5-7b-lora"
HF_TOKEN = os.getenv("HF_TOKEN")
QIAN_INSTRUCTION = "请用钱钟书式的讽刺、机智和比喻回答下面的问题或续写下面的文字。"
LINCOLN_INSTRUCTION = (
    "You are a historical analysis assistant inspired by Abraham Lincoln's "
    "documented writings. Use plain language, moral clarity, balanced clauses, "
    "humility, and reasoned persuasion. Never claim to be Lincoln. Distinguish "
    "historical evidence from speculation. Never invent quotations."
)

tokenizer = None
model = None


def load_model() -> None:
    global model, tokenizer
    if model is not None:
        return

    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        adapter_file = Path(QIAN_ADAPTER) / filename
        if not adapter_file.is_file():
            raise FileNotFoundError(f"Required Qian adapter file is missing: {adapter_file}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        QIAN_ADAPTER,
        adapter_name="qian",
    )
    model.load_adapter(
        LINCOLN_ADAPTER,
        adapter_name="lincoln",
        token=HF_TOKEN,
    )
    model.eval()


def generate(prompt: str, adapter_name: str) -> str:
    model.set_adapter(adapter_name)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


@spaces.GPU(duration=120)
def respond_qian(message: str, history: list[dict[str, str]]) -> str:
    load_model()
    del history
    prompt = (
        f"### 指令:\n{QIAN_INSTRUCTION}\n\n"
        f"### 输入:\n{message.strip()}\n\n"
        "### 输出:\n"
    )
    return generate(prompt, "qian")


@spaces.GPU(duration=120)
def respond_lincoln(message: str, history: list[dict[str, str]]) -> str:
    load_model()

    messages = [{"role": "system", "content": LINCOLN_INSTRUCTION}]
    for item in history[-10:]:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message.strip()})
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return generate(prompt, "lincoln")


with gr.Blocks(title="Reanimation Jutsu") as demo:
    gr.Markdown("# Reanimation Jutsu")
    with gr.Tab("Abraham Lincoln"):
        gr.ChatInterface(
            fn=respond_lincoln,
            cache_examples=False,
            description=(
                "Ask about modern society or historical principles. The model "
                "uses a Lincoln-inspired style without claiming to be Lincoln."
            ),
            examples=[
                "How might Lincoln's principles inform a discussion about border walls?",
                "What can Lincoln's writings teach us about political hatred today?",
                "How should a democracy respond when false claims spread online?",
            ],
        )
    with gr.Tab("钱钟书 Qian Zhongshu"):
        gr.ChatInterface(
            fn=respond_qian,
            cache_examples=False,
            description="输入一个问题或一段文字。模型将使用微调后的讽刺、机智和比喻风格作答。",
            examples=[
                "请谈谈现代人对手机的依赖。",
                "一个人为了显得有学问，总爱引用自己不懂的书。请评论他。",
            ],
        )

demo.queue(default_concurrency_limit=1).launch(ssr_mode=False)
