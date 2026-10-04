#!/usr/bin/env python3
"""Validate curated modern-to-original pairs and create training JSONL splits."""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

try:
    from .prepare_weicheng_dataset import is_near_identical, write_jsonl
except ImportError:
    from prepare_weicheng_dataset import is_near_identical, write_jsonl


# A guard against common source names, not a general named-entity detector.
# Sentence selection and paraphrases were also checked individually.
FORBIDDEN_NAMES = (
    "鸿渐", "辛楣", "文纨", "晓芙", "柔嘉", "遯翁", "梅亭", "尔谦",
    "松年", "处厚", "学愈", "刘东方", "元朗", "效成", "鹏图", "凤仪",
    "子潇", "斜川", "慎明", "蘅孙", "阿刘", "阿福", "阿丑", "阿凶",
    "李妈", "张妈", "苏小姐", "鲍小姐", "孙小姐", "唐小姐", "方先生",
    "赵先生", "汪太太", "陆太太", "沈太太", "张太太", "周经理", "孙氏",
    "周家", "苏氏", "方家", "孙家", "老汪", "老李", "侯营长"
)


def novel_body(source_text: str) -> str:
    """Skip the contents page, front matter, and appendix."""
    heading = re.compile(r"^第[一二三四五六七八九十]+章\s*$", re.MULTILINE)
    matches = list(heading.finditer(source_text))
    starts = [match for match in matches if match.group().strip() == "第一章"]
    if not starts:
        raise ValueError("Source has no first chapter heading")
    start = starts[-1].end()
    appendix = re.search(r"^附录[^\n]*", source_text[start:], re.MULTILINE)
    end = start + appendix.start() if appendix else len(source_text)
    return source_text[start:end]


def validate_rows(
    rows: list[dict[str, str]], source_text: str, expected_count: int = 310,
) -> None:
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} curated pairs")
    body = novel_body(source_text)
    outputs: set[str] = set()
    inputs: set[str] = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {"instruction", "input", "output"}:
            raise ValueError(f"Row {index}: expected instruction/input/output")
        if any(not isinstance(value, str) or not value.strip() for value in row.values()):
            raise ValueError(f"Row {index}: fields must be nonempty strings")
        if row["output"] not in body:
            raise ValueError(f"Row {index}: output is not an exact novel excerpt")
        if row["output"] in outputs or row["input"] in inputs:
            raise ValueError(f"Row {index}: duplicate input or output")
        for field in ("instruction", "input", "output"):
            if any(name in row[field] for name in FORBIDDEN_NAMES):
                raise ValueError(f"Row {index}: {field} contains a forbidden name")
        if is_near_identical(row["input"], row["output"]):
            raise ValueError(f"Row {index}: input needs a more substantial paraphrase")
        outputs.add(row["output"])
        inputs.add(row["input"])


def split_rows(
    rows: list[dict[str, str]], eval_ratio: float = 0.08, seed: int = 2818,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if len(rows) < 2 or not 0 < eval_ratio < 1:
        raise ValueError("Need at least two rows and an eval ratio between 0 and 1")
    shuffled = rows.copy()
    random.Random(seed).shuffle(shuffled)
    eval_count = min(len(rows) - 1, max(1, round(len(rows) * eval_ratio)))
    return shuffled[eval_count:], shuffled[:eval_count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="example_data/qianzhongshu_rewrite_curated.json")
    parser.add_argument("--source", default="围城.txt")
    parser.add_argument("--train-out", default="example_data/qianzhongshu_rewrite_train.jsonl")
    parser.add_argument("--eval-out", default="example_data/qianzhongshu_rewrite_eval.jsonl")
    parser.add_argument("--eval-ratio", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=2818)
    args = parser.parse_args()
    rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    source_text = Path(args.source).read_text(encoding="utf-8")
    try:
        validate_rows(rows, source_text)
        train, evaluation = split_rows(rows, args.eval_ratio, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    write_jsonl(Path(args.train_out), train)
    write_jsonl(Path(args.eval_out), evaluation)
    print(f"curated_pairs={len(rows)}")
    print(f"train={len(train)} {args.train_out}")
    print(f"eval={len(evaluation)} {args.eval_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
