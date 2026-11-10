"""Token statistics for the fixed prompt set (after applying the chat template).

Usage: python scripts/prompt_stats.py [loadtest/prompts.jsonl]
"""
import json
import os
import sys
from collections import defaultdict
from statistics import mean

from transformers import AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "loadtest/prompts.jsonl"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    rows = [json.loads(line) for line in open(path) if line.strip()]

    by_cat = defaultdict(list)
    for r in rows:
        text = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                       tokenize=False, add_generation_prompt=True)
        by_cat[r["category"]].append((len(tok(text)["input_ids"]), r["max_tokens"]))

    print(f"{len(rows)} prompts from {path}\n")
    print(f"{'category':<12} {'n':>3} {'in_min':>7} {'in_mean':>8} {'in_max':>7} {'max_out':>8}")
    for cat, vals in by_cat.items():
        ins = [v[0] for v in vals]
        print(f"{cat:<12} {len(vals):>3} {min(ins):>7} {mean(ins):>8.0f} {max(ins):>7} {vals[0][1]:>8}")
    all_in = [v[0] for vals in by_cat.values() for v in vals]
    all_out = [v[1] for vals in by_cat.values() for v in vals]
    print(f"\nmean input tokens: {mean(all_in):.0f} | mean max output tokens: {mean(all_out):.0f}")


if __name__ == "__main__":
    main()