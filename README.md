# Reanimation Jutsu Workflow

Open the [Reanimation Jutsu Space](https://huggingface.co/spaces/Playoung2818/Reanimation-Jutsu).


The current Space provides two giants:

- Qian Zhongshu 
- Abraham Lincoln 


## Workflow

```text
[Source documents](source_data/lincoln_speeches_letters.txt)
      |
      v
Dataset preparation script
      |
      +--> Historical topic-response examples
      +--> Historical continuation examples
      +--> Reviewed modern-domain examples (lincoln_synthetic_modern)
      |
      v
Train and validation JSONL files
      |
      v
Private Hugging Face dataset
      |
      v
L4 Hugging Face Job
      |
      v
Lincoln LoRA adapter
      |
      v
Qwen base model + Qian adapter + Lincoln adapter
      |
      v
Two-tab Gradio Space
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
| `huggingface_space/app.py` | Loads both adapters and creates both chat tabs. |
| `example_data/qianzhongshu_rewrite_curated.json` | Stores all 310 modern-input/original-output pairs. |
| `scripts/prepare_curated_weicheng_dataset.py` | Creates the 285-row training split and 25-row validation split. |
| `scripts/train_qian_hf_job.py` | Trains a private rewrite adapter on Hugging Face Jobs. |
| `saved_models/qian_curated_hf_job.json` | Records the job ID, repository IDs, source revisions, and file hashes. |
| `huggingface_space/deployment.json` | Records the adapter revisions deployed to the Space. |

## Data design

The [dataset playground](Dataset_Playground.ipynb) provides dataset edits, experimental copies, and a small LoRA experiment in one notebook. It compares model responses after training on original and edited data.

The Qian module is now trained as a rewrite/style-transfer adapter. The user provides one sentence or a short paragraph. The model rewrites it in a Qian-inspired style while preserving the original meaning.

The rewrite datasets are `example_data/qianzhongshu_rewrite_train.jsonl` and `example_data/qianzhongshu_rewrite_eval.jsonl`.

The file `example_data/qianzhongshu_rewrite_curated.json` contains 310 pairs: 10 preview examples and 300 additional examples. Each `input` is an individually written modern-Chinese paraphrase. Each `output` is an unchanged sentence from the novel in `围城.txt`. The selected sentences contain no names of people or places. `source_data/weicheng_curated_manifest.json` records the source location of each sentence.

The script `scripts/prepare_curated_weicheng_dataset.py` makes sure that the outputs match the source and that the pairs differ beyond spelling and punctuation. It creates 285 training rows and 25 validation rows. Both Qian training scripts use this file and do not append the old seed examples. The training script requires an NVIDIA GPU with CUDA support.

The remote trainer `scripts/train_qian_hf_job.py` uses `Playoung2818/qianzhongshu-curated-rewrite-data`. It reads only `train.jsonl` and `validation.jsonl` from a fixed dataset revision. It trains for three passes and selects the checkpoint with the lowest validation loss, a measure of prediction error. It saves the adapter to the private repository `Playoung2818/qianzhongshu-qwen2.5-7b-curated-lora-v1`. It does not change the deployed Space.

The job uses one L4 GPU with a two-hour limit. `saved_models/qian_curated_hf_job.json` records the job ID and the dataset and base-model revisions. Use `hf jobs inspect JOB_ID` for its status and `hf jobs logs -f JOB_ID` for live logs. The job receives `HF_TOKEN` as a secret, not as a file or a command-line token.

The older script `scripts/prepare_weicheng_dataset.py` remains available for experiments. Do not use it to replace the curated pairs. Its rewrite mode uses automatic clause deletion, not individually written paraphrases.

The Lincoln source comes from Project Gutenberg eBook 14721.

The preparation script extracts documents, splits passages, and creates two historical task types.

The `topic_response` task uses a document title as the prompt. Lincoln's text becomes the target response.

The `continuation` task uses a passage opening as the prompt. The remaining passage becomes the target response.

The modern-domain file contains reviewed responses about current issues. These examples separate historical evidence from reasoned speculation.

The final train split contains 381 historical examples and 16 modern-domain examples. The validation split contains 23 historical examples.

The split reserves 10% of source document titles for validation. All examples with the same title stay in one split.

## Project origin

This workspace started from [Suffoquer-fang/LuXun-GPT](https://github.com/Suffoquer-fang/LuXun-GPT).

The current workflow uses Qwen2.5, PEFT, TRL, Hugging Face Jobs, and Gradio.
