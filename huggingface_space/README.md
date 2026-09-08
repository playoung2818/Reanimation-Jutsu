---
title: Reanimation Jutsu
emoji: 🏛️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
---

# Reanimation Jutsu

This Space provides separate chat tabs for the Qian Zhongshu and Lincoln adapters.

Both tabs share the `Qwen/Qwen2.5-7B-Instruct` base model.

The Qian adapter path resolves to the directory that contains `app.py`. The Lincoln adapter loads from `Playoung2818/lincoln-qwen2.5-7b-lora`.

Place `adapter_config.json` and `adapter_model.safetensors` beside `app.py`.

Add `HF_TOKEN` as a Space secret. The token must grant read access to the private adapter repository.
