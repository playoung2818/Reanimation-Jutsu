#!/usr/bin/env python3
"""Build a Lincoln-style chat dataset from a public-domain text collection."""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a historical writing assistant inspired by Abraham Lincoln's "
    "public-domain speeches and letters. Use plain language, moral clarity, "
    "balanced clauses, humility, and reasoned persuasion. Do not claim to be "
    "Lincoln. Do not invent quotations."
)
SOURCE_URL = "https://www.gutenberg.org/ebooks/14721"


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def source_body(text: str) -> str:
    start_marker = "LINCOLN'S SPEECHES AND LETTERS"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker, start)
    return text[start:end].strip()


def documents(text: str) -> list[tuple[str, str]]: ## (title, body)
    heading = re.compile(r"(?ms)^_([^_]+?)_\s*$")
    matches = list(heading.finditer(text))
    docs: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        title = re.sub(r"\s+", " ", match.group(1)).strip()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = normalize(text[match.end() : body_end])
        if len(body) >= 100:
            docs.append((title, body))
    return docs


def chunks(text: str, minimum: int = 500, maximum: int = 1800) -> list[str]:
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    result: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip()
        if current and len(candidate) > maximum:
            if len(current) >= minimum:
                result.append(current)
            current = paragraph
        else:
            current = candidate
    if len(current) >= minimum:
        result.append(current)
    return result


def split_passage(text: str) -> tuple[str, str] | None:
    target = max(250, int(len(text) * 0.42))
    boundaries = [match.end() for match in re.finditer(r"[.!?](?:[\"']|\])?\s+", text)]
    valid = [value for value in boundaries if 220 <= value <= len(text) - 220]
    if not valid:
        return None
    split_at = min(valid, key=lambda value: abs(value - target))
    return text[:split_at].strip(), text[split_at:].strip()


def row(user: str, assistant: str, title: str, task: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "title": title,
        "task": task,
        "source": SOURCE_URL,
    }


def build_rows(docs: list[tuple[str, str]]) -> list[dict]:
    rows: list[dict] = []
    for title, body in docs:
        passages = chunks(body)
        if not passages:
            continue

        topic_answer = passages[0]
        rows.append(
            row(
                f"Write a concise address or letter concerning this subject: {title}",
                topic_answer,
                title,
                "topic_response",
            )
        )
        for passage in passages:
            pair = split_passage(passage)
            if pair is None:
                continue
            opening, completion = pair
            rows.append(
                row(
                    "Continue this passage in a concise, principled nineteenth-century "
                    f"American rhetorical style:\n\n{opening}",
                    completion,
                    title,
                    "continuation",
                )
            )
    return rows


def split_rows(rows: list[dict], eval_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not 0 < eval_ratio < 1:
        raise ValueError("The evaluation ratio must be between 0 and 1.")
    titles = sorted({item["title"] for item in rows})
    if len(titles) < 2:
        raise ValueError("The dataset must contain at least two source documents.")
    random.Random(seed).shuffle(titles)
    eval_count = min(len(titles) - 1, max(1, round(len(titles) * eval_ratio)))
    eval_titles = set(titles[:eval_count])
    train_rows = [item for item in rows if item["title"] not in eval_titles]
    eval_rows = [item for item in rows if item["title"] in eval_titles]
    return train_rows, eval_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="source_data/lincoln_speeches_letters.txt")
    parser.add_argument("--train-out", default="example_data/lincoln_train.jsonl")
    parser.add_argument("--eval-out", default="example_data/lincoln_eval.jsonl")
    parser.add_argument(
        "--synthetic-source",
        default="example_data/lincoln_synthetic_modern.jsonl",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Fraction of source document titles for validation")
    parser.add_argument("--seed", type=int, default=1809)
    args = parser.parse_args()

    text = Path(args.source).read_text(encoding="utf-8-sig")
    historical_rows = build_rows(documents(source_body(text)))
    rng = random.Random(args.seed)
    train_rows, eval_rows = split_rows(historical_rows, args.eval_ratio, args.seed)
    synthetic_rows = read_jsonl(Path(args.synthetic_source))
    train_rows.extend(synthetic_rows)
    rng.shuffle(train_rows)

    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)
    print(
        f"historical={len(historical_rows)} synthetic={len(synthetic_rows)} "
        f"train={len(train_rows)} eval={len(eval_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
