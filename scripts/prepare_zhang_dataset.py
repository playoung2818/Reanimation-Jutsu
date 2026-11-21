#!/usr/bin/env python3
"""Select Zhang Ailing excerpts, then export reviewed modern-to-original pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path

try:
    from .prepare_weicheng_dataset import is_near_identical, write_jsonl
except ImportError:
    from prepare_weicheng_dataset import is_near_identical, write_jsonl

WORKS = ("小团圆", "倾城之恋", "金锁记")
INSTRUCTION = (
    "请将下面的现代中文改写成张爱玲式的文学中文。保持原意，不回答问题，不解释，不补充新事实。"
    "语言应有细腻的心理观察、具体的日常意象、含蓄的反讽和苍凉感。"
)
# Selection guard only. Human review must also remove other names and places.
FORBIDDEN_NAMES = (
    "九莉", "邵之雍", "之雍", "蕊秋", "比比", "楚娣", "汝狄", "燕山", "邵家",
    "流苏", "柳原", "宝络", "白公馆", "白四爷", "白老太太", "徐太太", "范柳原",
    "七巧", "长安", "长白", "季泽", "芝寿", "兰仙", "凤箫", "云泽", "姜公馆", "姜家",
    "上海", "香港", "伦敦", "罗马", "斯巴达克斯", "朵云轩", "英吉利", "中国", "英国",
)
FIELDS = {"work", "source_offset", "source_length", "source_sha256", "instruction", "input", "output", "reviewed"}


def load_sources(directory: Path) -> dict[str, str]:
    return {work: (directory / f"{work}.txt").read_text(encoding="utf-8") for work in WORKS}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def excerpt_candidates(text: str, min_chars: int = 22, max_chars: int = 150) -> list[tuple[int, str]]:
    """Keep intact sentences within paragraphs, with character offsets."""
    candidates = []
    seen = set()
    for paragraph in re.finditer(r"[^\n]+", text):
        for match in re.finditer(r"[^。！？；]+[。！？；](?:[”』」])?", paragraph.group()):
            raw = match.group()
            output = raw.strip()
            if not min_chars <= len(output) <= max_chars:
                continue
            if any(name in output for name in FORBIDDEN_NAMES):
                continue
            # Avoid partial dialogue, nested quotations, and context-dependent fragments.
            if any(char in output for char in '“”「」『』：:'):
                continue
            if output in seen:
                continue
            start = paragraph.start() + match.start() + len(raw) - len(raw.lstrip())
            candidates.append((start, output))
            seen.add(output)
    return candidates


def select_candidates(sources: dict[str, str], counts: tuple[int, ...] = (104, 103, 103), seed: int = 2818) -> list[dict]:
    rows = []
    seen_outputs = set()
    rng = random.Random(seed)
    for work, count in zip(WORKS, counts):
        candidates = [(offset, text) for offset, text in excerpt_candidates(sources[work]) if text not in seen_outputs]
        if len(candidates) < count:
            raise ValueError(f"{work}: only {len(candidates)} eligible sentences for {count} requested candidates")
        # Sample each region of the work instead of collecting only the opening pages.
        selected = []
        for i in range(count):
            bucket = candidates[i * len(candidates) // count:(i + 1) * len(candidates) // count]
            # Favor imagery without changing a single character of the source.
            rich = [item for item in bucket if any(marker in item[1] for marker in ("像", "仿佛", "似乎", "颜色", "月", "光", "心"))]
            selected.append(rng.choice(rich or bucket))
        for offset, output in selected:
            seen_outputs.add(output)
            rows.append({
                "work": work, "source_offset": offset, "source_length": len(output),
                "source_sha256": sha256(sources[work]), "instruction": INSTRUCTION,
                "input": "", "output": output, "reviewed": False,
            })
    return rows


def validate_rows(rows: list[dict], sources: dict[str, str], expected_count: int = 310, require_review: bool = True) -> None:
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} curated pairs")
    hashes = {work: sha256(text) for work, text in sources.items()}
    seen_inputs, seen_outputs = set(), set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError(f"Row {index}: invalid curated record fields")
        work = row["work"]
        if not isinstance(work, str) or work not in sources:
            raise ValueError(f"Row {index}: unknown source work")
        if row["source_sha256"] != hashes[work]:
            raise ValueError(f"Row {index}: source hash changed")
        start, length = row["source_offset"], row["source_length"]
        if type(start) is not int or type(length) is not int or start < 0 or length < 1:
            raise ValueError(f"Row {index}: invalid source offset or length")
        if any(not isinstance(row[field], str) or not row[field].strip() for field in ("instruction", "input", "output")):
            raise ValueError(f"Row {index}: instruction/input/output must be nonempty strings")
        if row["instruction"] != INSTRUCTION:
            raise ValueError(f"Row {index}: use the Zhang Ailing rewrite instruction")
        if len(row["output"]) != length or sources[work][start:start + length] != row["output"]:
            raise ValueError(f"Row {index}: output is not the exact source excerpt at its recorded offset")
        if type(row["reviewed"]) is not bool or (require_review and not row["reviewed"]):
            raise ValueError(f"Row {index}: human review required before training")
        if any(name in row[field] for field in ("input", "output") for name in FORBIDDEN_NAMES):
            raise ValueError(f"Row {index}: input/output contains a source name or place")
        if row["input"] in seen_inputs or row["output"] in seen_outputs:
            raise ValueError(f"Row {index}: duplicate input or output")
        if is_near_identical(row["input"], row["output"]):
            raise ValueError(f"Row {index}: input needs a substantial modern-Chinese paraphrase")
        seen_inputs.add(row["input"])
        seen_outputs.add(row["output"])
    if {row["work"] for row in rows} != set(WORKS):
        raise ValueError("Curated pairs must cover all three works")


def split_rows(rows: list[dict], eval_ratio: float = 0.08, seed: int = 2818) -> tuple[list[dict], list[dict]]:
    """Deterministic, work-balanced split with an exact total validation count."""
    if not 0 < eval_ratio < 1:
        raise ValueError("Evaluation ratio must be between zero and one")
    groups = {work: [row for row in rows if row["work"] == work] for work in WORKS}
    if any(len(group) < 2 for group in groups.values()):
        raise ValueError("Each work needs at least two pairs for train and validation")
    total_eval = round(len(rows) * eval_ratio)
    if not len(WORKS) <= total_eval <= len(rows) - len(WORKS):
        raise ValueError("Evaluation ratio leaves too few rows per work")
    quotas = {work: len(group) * total_eval / len(rows) for work, group in groups.items()}
    counts = {work: min(len(groups[work]) - 1, max(1, math.floor(quota))) for work, quota in quotas.items()}
    while sum(counts.values()) != total_eval:
        if sum(counts.values()) < total_eval:
            work = max((w for w in WORKS if counts[w] < len(groups[w]) - 1), key=lambda w: quotas[w] - counts[w])
            counts[work] += 1
        else:
            work = max((w for w in WORKS if counts[w] > 1), key=lambda w: counts[w] - quotas[w])
            counts[work] -= 1
    train, evaluation = [], []
    rng = random.Random(seed)
    for work, group in groups.items():
        rng.shuffle(group)
        evaluation.extend(group[:counts[work]])
        train.extend(group[counts[work]:])
    rng.shuffle(train)
    rng.shuffle(evaluation)
    return train, evaluation


def training_row(row: dict) -> dict[str, str]:
    return {field: row[field] for field in ("instruction", "input", "output")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("source_data/zhang_ailing"))
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("candidates", help="Select 310 originals; inputs stay blank pending curation")
    select.add_argument("--out", type=Path, default=Path("example_data/zhangailing_rewrite_candidates.json"))
    select.add_argument("--force", action="store_true")
    preview = commands.add_parser("preview", help="Check and display ten draft pairs without exporting training data")
    preview.add_argument("--dataset", type=Path, default=Path("example_data/zhangailing_rewrite_preview.json"))
    prepare = commands.add_parser("prepare", help="Export only 310 complete, reviewed pairs")
    prepare.add_argument("--dataset", type=Path, default=Path("example_data/zhangailing_rewrite_curated.json"))
    prepare.add_argument("--train-out", type=Path, default=Path("example_data/zhangailing_rewrite_train.jsonl"))
    prepare.add_argument("--eval-out", type=Path, default=Path("example_data/zhangailing_rewrite_eval.jsonl"))
    prepare.add_argument("--manifest-out", type=Path, default=Path("source_data/zhang_ailing/curated_manifest.json"))
    args = parser.parse_args()
    try:
        sources = load_sources(args.source_dir)
        if args.command == "candidates":
            if args.out.exists() and not args.force:
                parser.error("Candidate file exists; refusing to overwrite manual curation")
            rows = select_candidates(sources)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Selected {len(rows)} candidates: {args.out}; NOT ready for training")
            return 0
        rows = json.loads(args.dataset.read_text(encoding="utf-8"))
        if args.command == "preview":
            validate_rows(rows, sources, expected_count=10, require_review=False)
            for index, row in enumerate(rows, 1):
                print(f"{index}. {row['work']}\n现代中文：{row['input']}\n原文：{row['output']}\n")
            print("Source checks passed. These drafts still need human review.")
            return 0
        validate_rows(rows, sources)
        train, evaluation = split_rows(rows)
        write_jsonl(args.train_out, [training_row(row) for row in train])
        write_jsonl(args.eval_out, [training_row(row) for row in evaluation])
        manifest = {
            "source_sha256": {work: sha256(text) for work, text in sources.items()},
            "curated_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            "total_pairs": len(rows), "train_rows": len(train), "validation_rows": len(evaluation), "seed": 2818,
            "input_method": "Individually written modern-Chinese paraphrases, marked reviewed by a human",
            "output_method": "Unchanged source excerpts, checked at recorded character offsets",
            "train": train, "validation": evaluation,
        }
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"train={len(train)} {args.train_out}\nvalidation={len(evaluation)} {args.eval_out}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
