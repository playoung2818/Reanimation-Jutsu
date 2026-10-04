import os

import gradio as gr
import spaces
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
QIAN_ADAPTER = "Playoung2818/qianzhongshu-qwen2.5-7b-curated-lora-v1"
QIAN_REVISION = "83395c25502b9579f2d01802af19a841cad57eec"
LINCOLN_ADAPTER = "Playoung2818/lincoln-qwen2.5-7b-lora"
LINCOLN_REVISION = "5a82215478a821fe83e28c91ab30f22174d2ebd2"
HF_TOKEN = os.getenv("HF_TOKEN")
QIAN_INSTRUCTION = (
    "请将下面的现代中文改写成带有讽刺、机智、比喻和冷峭观察的文学中文。"
    "保持原意，不回答问题，不解释，不补充新事实。"
)
LINCOLN_INSTRUCTION = (
    "You are a historical analysis assistant inspired by Abraham Lincoln's "
    "documented writings. Use plain language, moral clarity, balanced clauses, "
    "humility, and reasoned persuasion. Never claim to be Lincoln. Distinguish "
    "historical evidence from speculation. Never invent quotations."
)

tokenizer = None
model = None
base_path = None
qian_path = None
lincoln_path = None


def prepare_assets() -> None:
    """Download on CPU before accepting requests, not during a GPU allocation."""
    global base_path, qian_path, lincoln_path
    if base_path is not None:
        return
    if not HF_TOKEN:
        raise RuntimeError("Add HF_TOKEN as a Space secret with read access to both private adapters.")
    base = snapshot_download(
        BASE_MODEL, revision=BASE_REVISION, token=HF_TOKEN,
        allow_patterns=[
            "config.json", "generation_config.json", "tokenizer*", "merges.txt", "vocab.json",
            "chat_template*", "model*.safetensors", "model.safetensors.index.json",
        ],
    )
    adapter_files = ["adapter_config.json", "adapter_model.safetensors"]
    qian = snapshot_download(
        QIAN_ADAPTER, revision=QIAN_REVISION, token=HF_TOKEN, allow_patterns=adapter_files,
    )
    lincoln = snapshot_download(
        LINCOLN_ADAPTER, revision=LINCOLN_REVISION, token=HF_TOKEN, allow_patterns=adapter_files,
    )
    base_path, qian_path, lincoln_path = base, qian, lincoln
    print(f"Cached rewrite adapter: {QIAN_ADAPTER}@{QIAN_REVISION}", flush=True)
    print(f"Cached Lincoln adapter: {LINCOLN_ADAPTER}@{LINCOLN_REVISION}", flush=True)


def load_model() -> None:
    global model, tokenizer
    if model is not None:
        return

    prepare_assets()
    loaded_tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    if loaded_tokenizer.pad_token is None:
        loaded_tokenizer.pad_token = loaded_tokenizer.eos_token
    loaded_tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path, local_files_only=True, dtype=dtype,
        device_map="auto", quantization_config=quantization_config,
    )
    loaded_model = PeftModel.from_pretrained(
        base_model, qian_path, adapter_name="qian", is_trainable=False,
    )
    loaded_model.load_adapter(lincoln_path, adapter_name="lincoln", is_trainable=False)
    loaded_model.config.use_cache = True
    loaded_model.eval()
    tokenizer, model = loaded_tokenizer, loaded_model


def generate(prompt: str, adapter_name: str) -> str:
    model.set_adapter(adapter_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    if inputs["input_ids"].shape[1] > 1024:
        raise gr.Error("This message is too long. Use a shorter sentence or paragraph.")
    inputs = inputs.to(model.device)
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
    with gr.Tab("钱钟书 Qian Zhongshu"):
        gr.ChatInterface(
            fn=respond_qian,
            cache_examples=False,
            description="输入一句话或一小段文字。模型只改写原文，不回答问题，也不补充新事实。",
            examples=[
                "老实人也会有恶意，而且往往让人毫无防备地。",
                "一个人为了显得有学问，总爱引用自己不懂的书。",
                "他拼命回想，却始终留不住那些记忆。"
            ],
        )
    with gr.Tab("Abraham Lincoln", visible=False):
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

if __name__ == "__main__":
    prepare_assets()
    demo.queue(default_concurrency_limit=1).launch(ssr_mode=False)
