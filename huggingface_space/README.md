---
title: Reanimation Jutsu
emoji: 🏛️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.22.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# Reanimation Jutsu

This Space currently shows the Qian Zhongshu tab. The Lincoln tab is temporarily hidden. Its adapter and code remain available.

The Qian tab rewrites the user's sentence or short paragraph. It does not answer the input as a question.

Both adapters share the `Qwen/Qwen2.5-7B-Instruct` base model.

The Qian adapter loads from the private repository `Playoung2818/qianzhongshu-qwen2.5-7b-curated-lora-v1`. It uses revision `83395c25502b9579f2d01802af19a841cad57eec`, trained on 285 modern-input/original-output pairs. The rewrite prompt matches the training prompt. `deployment.json` records the pinned model revisions and the training job ID.

The Lincoln adapter loads from `Playoung2818/lincoln-qwen2.5-7b-lora`. Its revision is `5a82215478a821fe83e28c91ab30f22174d2ebd2`.

The application downloads the base model and both adapters before it accepts requests. It loads the model onto a GPU only inside `@spaces.GPU` functions. Keep the Space on ZeroGPU or paid GPU hardware.

Add `HF_TOKEN` as a Space secret. The token must grant read access to both private adapter repositories. Do not upload private adapter weights into this public Space.
