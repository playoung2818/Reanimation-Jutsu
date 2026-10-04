#!/usr/bin/env python3
"""Build Qian Zhongshu style-transfer JSONL data from local Weicheng text.

The training script expects rows with:
  instruction, input, output

Default mode is now rewrite/style-transfer, not continuation. It creates short
examples where the target output is Weicheng prose and the input is a more
neutralized version of the same sentence or short passage. This is a bootstrap
method; reviewed plain-to-Qian rewrite pairs will give better results.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


CONTINUATION_INSTRUCTION = "请续写下面这段文字，保持钱钟书《围城》式的讽刺、机智和叙事语气。"
REWRITE_INSTRUCTION = (
    "请将下面这句话或短段文字改写成钱钟书《围城》式的中文。"
    "必须保持原意，不要回答问题，不要解释，不要补充新事实。"
    "语言应有讽刺、机智、比喻和冷峭的观察。"
)
STYLE_MARKERS = (
    "好像",
    "像",
    "仿佛",
    "似乎",
    "似的",
    "譬如",
    "比方",
    "活像",
    "宛如",
    "俨然",
)

DEFAULT_MAX_SIMILARITY = 0.90
PLAIN_REPLACEMENTS = {
    "什麽": "什么",
    "甚麽": "什么",
    "怎麽": "怎么",
    "那麽": "那么",
    "这麽": "这么",
    "为什麽": "为什么",
    "於": "于",
    "後": "后",
    "裡": "里",
    "罢": "吧",
    "乾": "干",
    "麽": "么",
}


def normalize_plain_text(text: str) -> str:
    """Normalize the spelling and dialogue variants used by the generator."""
    for old, new in PLAIN_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = re.sub(r"([\u4e00-\u9fff]{1,4})问道[:：]", r"\1问：", text)
    text = re.sub(r"([\u4e00-\u9fff]{1,4})说道[:：]", r"\1说：", text)
    return re.sub(r"([\u4e00-\u9fff]{1,4})道[:：]", r"\1说：", text)


def normalize_for_similarity(text: str) -> str:
    text = normalize_plain_text(unicodedata.normalize("NFKC", text))
    return "".join(
        char for char in text
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def is_near_identical(source: str, target: str, max_similarity: float = DEFAULT_MAX_SIMILARITY) -> bool:
    """Reject empty or copy-like pairs, including cosmetic-only changes."""
    if not 0 < max_similarity <= 1:
        raise ValueError("max_similarity must be greater than 0 and at most 1")
    source = normalize_for_similarity(source)
    target = normalize_for_similarity(target)
    if not source or not target or source == target:
        return True
    # Disabling autojunk keeps repeated Chinese characters in the comparison.
    # Check both directions because SequenceMatcher can break ties differently.
    similarity = max(
        SequenceMatcher(None, source, target, autojunk=False).ratio(),
        SequenceMatcher(None, target, source, autojunk=False).ratio(),
    )
    return similarity >= max_similarity


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


def compact_block(text: str) -> str:
    return re.sub(r"\s*\n\s*", "", text).strip()


def split_sentences(text: str) -> list[str]:
    sentences = re.findall(r".+?[。！？；](?:[”』」])?", text)
    tail = re.sub(r".*?[。！？；](?:[”』」])?", "", text)
    if tail.strip():
        sentences.append(tail.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def neutralize_input(text: str) -> str:
    """Create a rough, less-stylized input while preserving most content."""
    plain = normalize_plain_text(text)

    pieces = re.split(r"([，；。！？])", plain)
    kept: list[str] = []
    for index in range(0, len(pieces), 2):
        clause = pieces[index].strip()
        punct = pieces[index + 1] if index + 1 < len(pieces) else ""
        if not clause:
            continue
        is_style_clause = any(marker in clause for marker in STYLE_MARKERS)
        # Drop only some strongly marked decorative clauses. Keep enough text so
        # the model still has the original meaning to rewrite.
        if is_style_clause and len(clause) <= 45:
            continue
        kept.append(clause + punct)

    candidate = "".join(kept).strip()
    candidate = re.sub(r"[，；]+([。！？])", r"\1", candidate)
    candidate = re.sub(r"[，；]$", "。", candidate)

    # If the heuristic removed too much, keep the normalized original text.
    # The pair filter below rejects unchanged and near-identical examples.
    if len(candidate) < max(20, int(len(plain) * 0.55)):
        return plain
    return candidate


def build_continuation_examples(blocks: list[str], min_input_chars: int, max_input_chars: int) -> list[dict[str, str]]:
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
                    "instruction": CONTINUATION_INSTRUCTION,
                    "input": inp,
                    "output": out,
                }
            )
    return examples


def build_rewrite_targets(blocks: list[str], min_chars: int, max_chars: int) -> list[str]:
    targets: list[str] = []
    for block in blocks:
        sentences = split_sentences(compact_block(block))
        current = ""
        for sentence in sentences:
            if len(sentence) > max_chars:
                if len(current) >= min_chars:
                    targets.append(current)
                current = ""
                continue

            candidate = f"{current}{sentence}" if current else sentence
            if current and len(candidate) > max_chars:
                if len(current) >= min_chars:
                    targets.append(current)
                current = sentence
            else:
                current = candidate

            if len(current) >= min_chars and current[-1] in "。！？；":
                targets.append(current)
                current = ""

        if len(current) >= min_chars:
            targets.append(current)
    return targets


def build_rewrite_examples(
    blocks: list[str], min_chars: int, max_chars: int,
    max_similarity: float = DEFAULT_MAX_SIMILARITY,
) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for target in build_rewrite_targets(blocks, min_chars, max_chars):
        source = neutralize_input(target)
        if is_near_identical(source, target, max_similarity):
            continue
        key = (source, target)
        if key in seen:
            continue
        seen.add(key)
        examples.append(
            {
                "instruction": REWRITE_INSTRUCTION,
                "input": source,
                "output": target,
            }
        )
    return examples


def read_extra_pairs(
    path: Path | None, max_similarity: float = DEFAULT_MAX_SIMILARITY,
) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if {"input", "output"} <= item.keys():
            if is_near_identical(item["input"], item["output"], max_similarity):
                continue
            rows.append(
                {
                    "instruction": item.get("instruction", REWRITE_INSTRUCTION),
                    "input": item["input"],
                    "output": item["output"],
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="围城.txt")
    parser.add_argument("--train-out", default="example_data/qianzhongshu_rewrite_train.jsonl")
    parser.add_argument("--eval-out", default="example_data/qianzhongshu_rewrite_eval.jsonl")
    parser.add_argument("--task", choices=("rewrite", "continuation"), default="rewrite")
    parser.add_argument("--extra-pairs", default=None, help="Optional reviewed plain-to-Qian JSONL pairs to append to train.")
    parser.add_argument(
        "--max-similarity", type=float, default=DEFAULT_MAX_SIMILARITY,
        help="Drop rewrite pairs at or above this normalized text similarity (default: 0.90).",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=2818)
    parser.add_argument("--min-input-chars", type=int, default=120)
    parser.add_argument("--max-input-chars", type=int, default=520)
    parser.add_argument("--min-rewrite-chars", type=int, default=40)
    parser.add_argument("--max-rewrite-chars", type=int, default=360)
    args = parser.parse_args()
    if not 0 < args.max_similarity <= 1:
        parser.error("--max-similarity must be greater than 0 and at most 1")

    source = Path(args.source)
    text = normalize_text(source.read_text(encoding="utf-8"))
    text = drop_front_matter(text)
    blocks = paragraph_blocks(text)

    if args.task == "continuation":
        examples = build_continuation_examples(blocks, args.min_input_chars, args.max_input_chars)
        extra_pairs: list[dict[str, str]] = []
    else:
        examples = build_rewrite_examples(
            blocks, args.min_rewrite_chars, args.max_rewrite_chars, args.max_similarity,
        )
        extra_pairs = read_extra_pairs(
            Path(args.extra_pairs) if args.extra_pairs else None, args.max_similarity,
        )

    random.Random(args.seed).shuffle(examples)
    eval_count = max(1, int(len(examples) * args.eval_ratio)) if examples else 0
    eval_rows = examples[:eval_count]
    train_rows = examples[eval_count:]
    train_rows.extend(extra_pairs)
    random.Random(args.seed + 1).shuffle(train_rows)

    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)

    print(f"task={args.task}")
    if args.task == "rewrite":
        print(f"max_similarity={args.max_similarity}")
    print(f"source_chars={len(text)}")
    print(f"examples={len(examples)}")
    print(f"extra_train_pairs={len(extra_pairs)}")
    print(f"train={len(train_rows)} {args.train_out}")
    print(f"eval={len(eval_rows)} {args.eval_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
