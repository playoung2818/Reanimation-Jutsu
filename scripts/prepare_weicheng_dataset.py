#!/usr/bin/env python3
"""Build instruction JSONL data from local Weicheng text.

The training script in this repo expects rows with:
  instruction, input, output

With only a source text available, the least synthetic supervised task is
continuation: given one passage, continue in the same prose style.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


INSTRUCTION = "请续写下面这段文字，保持钱钟书《围城》式的讽刺、机智和叙事语气。"


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def drop_front_matter(text: str) -> str:
    marker = "\n第一章"
    idx = text.find(marker)
    if idx >= 0:
        return text[idx + len(marker) :].strip()
    return text


def paragraph_blocks(text: str) -> list[str]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    return [b for b in blocks if len(b) >= 40]


def build_examples(blocks: list[str], min_input_chars: int, max_input_chars: int) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for block in blocks:
        compact = re.sub(r"\s*\n\s*", "\n", block).strip()
        if len(compact) < min_input_chars * 2:
            continue

        split_at = min(max_input_chars, max(min_input_chars, len(compact) // 2))
        # Prefer splitting at a Chinese sentence boundary near the target split.
        window_start = max(min_input_chars, split_at - 80)
        window_end = min(len(compact) - min_input_chars, split_at + 80)
        boundary = -1
        for i in range(window_end, window_start, -1):
            if compact[i - 1] in "。！？；":
                boundary = i
                break
        if boundary > 0:
            split_at = boundary

        inp = compact[:split_at].strip()
        out = compact[split_at:].strip()
        if len(inp) >= min_input_chars and len(out) >= min_input_chars:
            examples.append(
                {
                    "instruction": INSTRUCTION,
                    "input": inp,
                    "output": out,
                }
            )
    return examples


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="围城.txt")
    parser.add_argument("--train-out", default="example_data/qianzhongshu_train.jsonl")
    parser.add_argument("--eval-out", default="example_data/qianzhongshu_eval.jsonl")
    parser.add_argument("--eval-ratio", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=2818)
    parser.add_argument("--min-input-chars", type=int, default=120)
    parser.add_argument("--max-input-chars", type=int, default=520)
    args = parser.parse_args()

    source = Path(args.source)
    text = normalize_text(source.read_text(encoding="utf-8"))
    text = drop_front_matter(text)
    examples = build_examples(paragraph_blocks(text), args.min_input_chars, args.max_input_chars)

    random.Random(args.seed).shuffle(examples)
    eval_count = max(1, int(len(examples) * args.eval_ratio)) if examples else 0
    eval_rows = examples[:eval_count]
    train_rows = examples[eval_count:]

    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)

    print(f"source_chars={len(text)}")
    print(f"examples={len(examples)}")
    print(f"train={len(train_rows)} {args.train_out}")
    print(f"eval={len(eval_rows)} {args.eval_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
