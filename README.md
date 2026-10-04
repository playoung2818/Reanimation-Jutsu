# Reanimation Jutsu Workflow

Open the [Reanimation Jutsu Space](https://huggingface.co/spaces/Playoung2818/Reanimation-Jutsu).


The Space currently shows the Qian Zhongshu rewrite tab. The Lincoln tab is temporarily hidden. Both adapters remain configured.


## Workflow

```text
围城.txt (original source sentences)
      |
      v
310 reviewed modern-input/original-output pairs
(example_data/qianzhongshu_rewrite_curated.json)
      |
      v
scripts/prepare_curated_weicheng_dataset.py
      |
      v
285 training rows + 25 validation rows (JSONL)
(example_data/qianzhongshu_rewrite_train.jsonl)
(example_data/qianzhongshu_rewrite_eval.jsonl)
      |
      v
Private Hugging Face dataset
Playoung2818/qianzhongshu-curated-rewrite-data
      |
      v
L4 Hugging Face Job
scripts/train_qian_hf_job.py
      |
      v
Private Qian LoRA adapter
Playoung2818/qianzhongshu-qwen2.5-7b-curated-lora-v1
      |
      v
Qwen/Qwen2.5-7B-Instruct + Qian adapter + existing Lincoln adapter
      |
      v
Gradio Space on ZeroGPU
```

## Important files

| File | Purpose |
|---|---|
| `source_data/lincoln_speeches_letters.txt` | Stores the public-domain Lincoln source text. |
| `scripts/prepare_lincoln_dataset.py` | Creates historical train and validation examples. |
| `example_data/lincoln_synthetic_modern.jsonl` | Stores reviewed modern-domain examples. |
| `example_data/lincoln_train.jsonl` | Stores the final train split. |
| `example_data/lincoln_eval.jsonl` | Stores the validation split. |
| `scripts/train_lincoln_hf_job.py` | Runs QLoRA on Hugging Face Jobs. |
| `huggingface_space/app.py` | Loads both adapters|
| `example_data/qianzhongshu_rewrite_curated.json` | Stores all 310 modern-input/original-output pairs. |
| `scripts/prepare_curated_weicheng_dataset.py` | Creates the 285-row training split and 25-row validation split. |
| `scripts/train_qian_hf_job.py` | Trains a private rewrite adapter on Hugging Face Jobs. |
| `saved_models/qian_curated_hf_job.json` | Records the job ID, repository IDs, source revisions, and file hashes. |
| `huggingface_space/deployment.json` | Records the adapter revisions deployed to the Space. |

## Data design


The Qian adapter is trained to rewrite modern Chinese in a Qian-inspired literary style.

I first selected 10 sentences from 《围城》 and asked ChatGPT to rewrite them in modern Chinese. After reviewing these examples, I expanded the dataset to 310 pairs. For training, I reversed the direction: the modern paraphrase is the input, and the unchanged original sentence is the output.

example_data/qianzhongshu_rewrite_curated.json stores all 310 pairs: 10 preview examples and 300 additional examples. The originals come from 围城.txt and exclude names of people and places. source_data/weicheng_curated_manifest.json records their source locations.

scripts/prepare_curated_weicheng_dataset.py checks that each output matches the source and rejects near-identical pairs. It splits the dataset into 285 training rows and 25 validation rows.

The remote trainer `scripts/train_qian_hf_job.py` uses `Playoung2818/qianzhongshu-curated-rewrite-data`. It reads only `train.jsonl` and `validation.jsonl` from a fixed dataset revision. It trains for three passes and selects the checkpoint with the lowest validation loss, a measure of prediction error. It saves the adapter to the private repository `Playoung2818/qianzhongshu-qwen2.5-7b-curated-lora-v1`. It does not change the deployed Space.

The Lincoln source comes from Project Gutenberg eBook 14721.

The [dataset playground](Dataset_Playground.ipynb) is the Draft Book


## Project origin

This workspace started from [Suffoquer-fang/LuXun-GPT](https://github.com/Suffoquer-fang/LuXun-GPT).

The current workflow uses Qwen2.5, PEFT, TRL, Hugging Face Jobs, and Gradio.
