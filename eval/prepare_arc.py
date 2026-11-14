"""Build the fixed quality-eval set: 1,000 ARC-Easy questions (grade-school science, multiple choice).

Run once; the output is committed so every backend (CPU or GPU, today or later) sees the
exact same questions.
Usage: python eval/prepare_arc.py [--n 1000] [--seed 0]
Writes: eval/arc_easy_1000.jsonl
"""
import argparse
import json
import random

from datasets import load_dataset

LETTERS = "ABCDE"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval/arc_easy_1000.jsonl")
    args = ap.parse_args()

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    rows = []
    for ex in ds:
        labels, texts = ex["choices"]["label"], ex["choices"]["text"]
        if ex["answerKey"] not in labels or len(texts) > len(LETTERS):
            continue
        # Some questions label options 1-4 instead of A-D: normalize to letters by position.
        rows.append({
            "id": ex["id"],
            "question": ex["question"],
            "choices": [{"label": LETTERS[i], "text": t} for i, t in enumerate(texts)],
            "answer": LETTERS[labels.index(ex["answerKey"])],
        })
    random.Random(args.seed).shuffle(rows)
    rows = sorted(rows[: args.n], key=lambda r: r["id"])
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} questions (of {len(ds)} in the test split) -> {args.out}")


if __name__ == "__main__":
    main()
