# Lincoln LoRA Deployment Instructions

Follow these steps to prepare the dataset, train the adapter, and deploy the Hugging Face Space.

## 1. Prepare the Lincoln dataset

Install the required Python packages.

```bash
pip install -r requirements_qwen.txt
```

Place the Project Gutenberg text at this path.

```text
source_data/lincoln_speeches_letters.txt
```

Create the train and validation files.

```bash
python scripts/prepare_lincoln_dataset.py
```

Verify the row counts.

```bash
wc -l example_data/lincoln_train.jsonl example_data/lincoln_eval.jsonl
```

Verify that the train file contains 397 rows.

Verify that the validation file contains 23 rows.

## 2. Upload the Lincoln dataset

Authenticate with Hugging Face.

```bash
hf auth login
```

Create the private dataset repository once.

```bash
hf repos create Playoung2818/lincoln-style-chat-data --type dataset --private --exist-ok
```

Upload the train split.

```bash
hf upload Playoung2818/lincoln-style-chat-data \
  example_data/lincoln_train.jsonl train.jsonl \
  --type dataset
```

Upload the validation split.

```bash
hf upload Playoung2818/lincoln-style-chat-data \
  example_data/lincoln_eval.jsonl validation.jsonl \
  --type dataset
```

## 3. Train the Lincoln LoRA adapter

Set `HF_TOKEN` in the local shell.

Use a token with write access.

```bash
export HF_TOKEN="YOUR_WRITE_TOKEN"
```

Do not commit the token. The token grants access to private Hugging Face resources.

Start the L4 job.

```bash
hf jobs uv run scripts/train_lincoln_hf_job.py \
  --flavor l4x1 \
  --secrets HF_TOKEN \
  --timeout 2h
```

List the jobs.

```bash
hf jobs ps --all
```

Read the job log.

```bash
hf jobs logs JOB_ID
```

Verify that the job saves the adapter to `Playoung2818/lincoln-qwen2.5-7b-lora`.

Verify that the adapter repository contains these files:

- `adapter_config.json`
- `adapter_model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`

## 4. Configure the Space

Open the Space settings page.

Add `HF_TOKEN` under **Variables and secrets**.

Use a token that can read the private Lincoln adapter.

Select ZeroGPU or paid GPU hardware.

Use `@spaces.GPU` for ZeroGPU support.

## 5. Deploy both chat tabs

Verify the application syntax.

```bash
python -m py_compile huggingface_space/app.py
```

Upload the Space files.

```bash
hf upload Playoung2818/Zhongshu_Qian huggingface_space \
  --type space \
  --exclude '__pycache__/*' \
  --commit-message "Update Reanimation Jutsu"
```

Verify the Space status.

```bash
hf spaces info Playoung2818/Zhongshu_Qian \
  --expand runtime,sha,lastModified \
  --format json
```

Wait until the runtime stage reports `RUNNING`.
