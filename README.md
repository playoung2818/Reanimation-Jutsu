# Reanimation Jutsu Workflow

Open the [Reanimation Jutsu Space](https://huggingface.co/spaces/Playoung2818/Zhongshu_Qian).


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

## Data design

The [dataset playground](Dataset_Playground.ipynb) provides dataset edits, experimental copies, and a small LoRA experiment in one notebook. It compares model responses after training on original and edited data.

The Qian persona uses examples from *围城* (Weicheng). Its canonical datasets are `example_data/qianzhongshu_train.jsonl` and `example_data/qianzhongshu_eval.jsonl`.

These files contain 300 train examples and 26 validation examples. Both Qian training scripts use these files. The duplicate `weicheng_*.jsonl` files were removed.

The script `scripts/prepare_weicheng_dataset.py` creates the Qian datasets from `围城.txt`.

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
