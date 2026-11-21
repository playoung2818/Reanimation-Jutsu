# Zhang Ailing rewrite model

This workflow follows the reviewed Qian GPT process. It trains a rewrite model, not a factual chatbot about Zhang Ailing.
A LoRA adapter is a small set of learned model changes. This workflow adds one to `Qwen/Qwen2.5-7B-Instruct`.

## Current state

The three texts came from the local Zhang Ailing collection in Downloads. No new book download was needed.
The extractor reads the EPUB contents and keeps only the selected stories and all twelve chapters of 《小团圆》.
It excludes covers, copyright pages, contents pages, headings, and other works.

| Local file | Characters, including separators |
|---|---:|
| `source_data/zhang_ailing/小团圆.txt` | 152,757 |
| `source_data/zhang_ailing/倾城之恋.txt` | 29,007 |
| `source_data/zhang_ailing/金锁记.txt` | 31,252 |

`source_data/zhang_ailing/extraction_manifest.json` records the EPUB sections and file hashes.
A hash identifies the exact file contents. These records identify this local edition, not its accuracy against a published edition.
Use copies that you are authorized to process. Do not publish the texts or derived excerpts without the necessary rights.

The dataset contains 310 draft pairs: 104 from 《小团圆》 and 103 from each other work.
All modern-Chinese inputs are filled in `example_data/zhangailing_rewrite_candidates.json`.
The same initial drafts are saved in `example_data/zhangailing_rewrite_curated.json`. Review and revise that curated file from now on.
The original outputs, source offsets, source hashes, instructions, and review flags remain unchanged.
All review flags are false. Human review is not complete.
Both private splits are uploaded. The experimental L4 job completed all three training passes using the unreviewed drafts.
The private adapter export contains all required files.
The [Space](https://huggingface.co/spaces/Playoung2818/Reanimation-Jutsu) now includes the experimental Zhang Ailing tab.
It pins adapter revision `6375152a5939596052e0e8d6734a48de61c040b6`. Live requests and desktop/mobile rendering passed the deployment checks.
The selected checkpoint is `checkpoint-18`, with validation loss 2.05795. Rewrite quality still needs testing.
Track [job 6acbc77cfee2c90070199d63](https://huggingface.co/jobs/Playoung2818/6acbc77cfee2c90070199d63).
`saved_models/zhangailing_curated_hf_job.json` records the job, fixed revisions, hashes, and unreviewed status.

`source_data/zhang_ailing/modern_paraphrases.json` stores the authored inputs and the hash of their matching output sequence.
`source_data/zhang_ailing/candidates_before_inputs.json` preserves the selection before its inputs were filled.
Some selected excerpts still contain names or depend on context. Rows 165, 235, and 264 have apparent text or boundary errors.
Review these issues before training. The drafts do not yet include newly written modern-day examples about exercise, meals, or work.

## Review the curated pairs

Open `example_data/zhangailing_rewrite_curated.json`. The separate ten-pair preview is optional and is not needed for training.
Do not overwrite the curated file with the candidate file after review begins. That would discard your review changes.
Run commands from the project root with `.venv312/bin/python`, which works in this moved workspace.
The old `.venv312/bin/hf` launcher refers to the previous project path. The new training launcher avoids it.

Complete these steps for each of the 310 records:

1. Read the excerpt and its surrounding source paragraph.
2. Reject excerpts with names, places, incomplete dialogue, or unclear meaning outside their context.
3. If you replace an excerpt, record its work, character offset, length, and source hash.
4. Review or revise the modern-Chinese paraphrase in `input`.
5. Make sure that `output` remains the exact source excerpt.
6. Make sure that the paraphrase keeps the meaning and does not add facts.
7. Set `reviewed` to `true` only after human review.

Use plain contemporary Chinese, not automatically deleted clauses. Keep the facts and mood without copying the literary wording.
Do not append the preview as extra records. The target is 310 total pairs, as in Qian GPT.
The automatic name list is only a guard. It does not detect every person or place.

If you need to repeat extraction on the same collection, run:

```bash
.venv312/bin/python scripts/extract_zhang_epub.py /path/to/collection.epub
```

The extractor stops if its output already exists. It supports this collection's NCX contents and paragraph classes, not arbitrary EPUB files.
If you need to regenerate candidates, run the command below before manual curation:

```bash
.venv312/bin/python scripts/prepare_zhang_dataset.py candidates
```

## Prepare the train and validation files

The preparation command rejects missing inputs, changed originals, changed source files, duplicate pairs, copy-like inputs, and unreviewed records.
It creates a repeatable split that includes all three works in training and validation.
It records the sources and split membership in `source_data/zhang_ailing/curated_manifest.json`.

```bash
.venv312/bin/python scripts/prepare_zhang_dataset.py prepare
wc -l example_data/zhangailing_rewrite_train.jsonl \
  example_data/zhangailing_rewrite_eval.jsonl
```

Make sure that the counts are 285 training rows and 25 validation rows.
Source files and derived Zhang data are excluded from Git. Keep a separate private backup.

## Upload and train

The launcher uses the existing `scripts/train_qian_hf_job.py` with explicit Zhang repository arguments.
It preserves Qian's prompt format, output-only training, 4-bit model loading, and three training passes.
The trainer selects the checkpoint with the lowest validation loss, a measure of prediction error.
It rejects examples above 768 tokens instead of cutting their original outputs.

The destination dataset is `Playoung2818/zhangailing-curated-rewrite-data`.
The destination adapter is `Playoung2818/zhangailing-qwen2.5-7b-curated-lora-v1`.
Both repositories must be private. Existing Qian and Lincoln repositories remain unchanged.

The default launcher requires every pair to be human-reviewed. It stops on the current draft dataset.
If you want to train an experiment before review, add `--allow-unreviewed`.
This flag leaves the review fields unchanged and records the draft status in the job and adapter manifests.
Run the local data check after review. This command performs no network request and starts no job.

```bash
.venv312/bin/python scripts/launch_zhang_hf_job.py
```

If the data passes review and you accept the GPU cost, set `HF_TOKEN` to your write token in the shell.
Do not put the token in a source file. If a write token is already saved by Hugging Face, load it without printing it:

```bash
export HF_TOKEN="$(.venv312/bin/python -c 'from huggingface_hub import get_token; print(get_token() or "")')"
```

The next command uploads both splits in one commit and starts a paid `l4x1` job with a two-hour limit.
The launcher records fixed dataset and base-model revisions, file hashes, and the job ID.
Do not repeat submission while a job is running. Each submission starts another paid job.

```bash
.venv312/bin/python scripts/launch_zhang_hf_job.py --submit
```

The current experiment used `--submit --allow-unreviewed`. Do not repeat that submission while its job is running.
Read `saved_models/zhangailing_curated_hf_job.json` for the job ID. Monitor the job on the Hugging Face Jobs page.
After training, make sure that the private adapter repository contains these files:

- `adapter_config.json`
- `adapter_model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `training_manifest.json`

The training launcher does not change the Space. The Zhang tab was deployed separately after training.
The first live outputs are quite plain. Use the tab to test meaning, mood, literary style, and original-text memorization before treating the model as ready.

## Run tests

The test suite uses synthetic text and optionally checks the local preview. It does not require a GPU or a paid job.
The preview test is skipped when the private local data is absent.

```bash
.venv312/bin/python -m unittest discover -s tests -q
```
