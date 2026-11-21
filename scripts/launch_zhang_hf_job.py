#!/usr/bin/env python3
"""Check the reviewed Zhang dataset; upload and start a paid L4 job only with --submit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

try:
    from .prepare_zhang_dataset import load_sources, split_rows, training_row, validate_rows
    from .train_qian_hf_job import BASE_MODEL, check_splits
except ImportError:
    from prepare_zhang_dataset import load_sources, split_rows, training_row, validate_rows
    from train_qian_hf_job import BASE_MODEL, check_splits

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "Playoung2818/zhangailing-curated-rewrite-data"
MODEL_ID = "Playoung2818/zhangailing-qwen2.5-7b-curated-lora-v1"


def build_training_data(rows: list[dict], sources: dict[str, str], *, require_review: bool = True) -> dict[str, bytes]:
    validate_rows(rows, sources, require_review=require_review)
    train, evaluation = split_rows(rows)
    dataset = {"train": [training_row(row) for row in train], "validation": [training_row(row) for row in evaluation]}
    check_splits(dataset)
    return {
        f"{split}.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records).encode("utf-8")
        for split, records in dataset.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "example_data/zhangailing_rewrite_curated.json")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "source_data/zhang_ailing")
    parser.add_argument("--submit", action="store_true", help="Upload private data and start a billable L4 GPU job")
    parser.add_argument("--allow-unreviewed", action="store_true", help="Explicitly train on draft pairs; keep review flags unchanged and record their status")
    args = parser.parse_args()
    try:
        rows = json.loads(args.dataset.read_text(encoding="utf-8"))
        files = build_training_data(rows, load_sources(args.source_dir), require_review=not args.allow_unreviewed)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    reviewed_rows = sum(row["reviewed"] for row in rows)
    review_status = "reviewed" if reviewed_rows == len(rows) else "unreviewed"
    if review_status == "unreviewed":
        print(f"WARNING: experimental draft-data training; only {reviewed_rows}/{len(rows)} pairs are human-reviewed. Review flags remain unchanged.", flush=True)
    print(f"Checked 285 training rows and 25 validation rows. Dataset: {DATASET_ID}. Adapter: {MODEL_ID}.", flush=True)
    if not args.submit:
        print("Dry run only. No upload or paid job. Add --submit after reviewing the data and GPU cost.")
        return 0
    token = os.environ.get("HF_TOKEN")
    if not token:
        parser.error("Set HF_TOKEN to a write token before using --submit. Do not put the token in a file or command argument.")

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    api.create_repo(DATASET_ID, repo_type="dataset", private=True, exist_ok=True)
    if not api.dataset_info(DATASET_ID).private:
        raise RuntimeError("Refusing to upload Zhang excerpts to a public dataset")
    api.create_repo(MODEL_ID, repo_type="model", private=True, exist_ok=True)
    if not api.model_info(MODEL_ID).private:
        raise RuntimeError("Refusing to train into a public adapter repository")
    # One atomic commit keeps train and validation files together at one revision.
    commit = api.create_commit(
        DATASET_ID, repo_type="dataset",
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=data) for name, data in files.items()],
        commit_message=f"Add {review_status} Zhang Ailing modern-to-original splits",
    )
    base_revision = api.model_info(BASE_MODEL).sha
    if not base_revision:
        raise RuntimeError("Could not resolve an immutable base model revision")
    trainer = ROOT / "scripts/train_qian_hf_job.py"
    script_args = [
        "--dataset-id", DATASET_ID, "--dataset-revision", commit.oid,
        "--base-revision", base_revision, "--model-id", MODEL_ID,
        "--output-dir", "zhangailing-curated-lora", "--epochs", "3",
        "--data-review-status", review_status,
    ]
    manifest = {
        "dataset_id": DATASET_ID, "dataset_revision": commit.oid,
        "base_model": BASE_MODEL, "base_revision": base_revision, "model_id": MODEL_ID,
        "trainer": str(trainer.relative_to(ROOT)), "trainer_sha256": hashlib.sha256(trainer.read_bytes()).hexdigest(),
        "file_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        "curated_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "train_rows": 285, "validation_rows": 25, "flavor": "l4x1", "timeout": "2h", "script_args": script_args,
        "data_review_status": review_status, "reviewed_rows": reviewed_rows,
        "allow_unreviewed": args.allow_unreviewed,
    }
    record_path = ROOT / "saved_models/zhangailing_curated_hf_job.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    job = api.run_uv_job(
        script=str(trainer), script_args=script_args, python="3.12", flavor="l4x1", timeout="2h",
        secrets={"HF_TOKEN": token},
    )
    manifest["job_id"] = job.id
    record_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Submitted paid L4 job: {job.id}\nRecord: {record_path}\nThe existing Space was not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
