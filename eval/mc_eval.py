"""Multiple-choice quality eval against ANY OpenAI-compatible server (Version A, llama.cpp, vLLM).

Greedy decoding, answer = the first option letter in the reply. Same questions, same prompt,
same parsing for every backend, so differences come from the model weights (quantization).

Usage:
  python eval/mc_eval.py --label llamacpp_q4_k_m --url http://localhost:8080 --concurrency 8
Writes:
  results/quality/<label>_preds.csv   (one row per question)
  results/quality_summary.csv         (one row per run, appended)
"""
import argparse
import asyncio
import csv
import json
import math
import os
import re
import time

import httpx

PROMPT = ("Answer the following multiple-choice science question with only the letter "
          "of the correct option.\n\nQuestion: {question}\n{options}\nAnswer:")
# Parsing, in order of trust (never a bare capital letter in prose: "A thermometer..." is not option A):
LEADING_RE = re.compile(r"^\W*([A-E])(?=\s*$|[.):*])")                     # "B", "B. metals", "**C**" (not "A thermometer")
ANSWER_IS_RE = re.compile(r"(?i:answer(?:\s+is)?)\s*[:\-]?\s*\**\(?([A-E])\b")  # "The correct answer is: B"
LETTER_PUNCT_RE = re.compile(r"(?:^|\s)\(?([A-E])[.):]")                     # "...so (C) is right"


def build_prompt(q: dict) -> str:
    options = "\n".join(f"{c['label']}. {c['text']}" for c in q["choices"])
    return PROMPT.format(question=q["question"], options=options)


def parse_letter(text: str, choices: list[dict]) -> str:
    """Which option did the model pick? Tries, in order: reply starts with a letter,
    'answer is X', the reply quotes exactly one option's text, a letter followed by . ) or :"""
    valid = {c["label"] for c in choices}
    t = text.strip()
    for rx in (LEADING_RE, ANSWER_IS_RE):
        m = rx.search(t)
        if m and m.group(1) in valid:
            return m.group(1)
    quoted = [c["label"] for c in choices if c["text"].strip().lower() in t.lower()]
    if len(quoted) == 1:
        return quoted[0]
    for m in LETTER_PUNCT_RE.finditer(t):
        if m.group(1) in valid:
            return m.group(1)
    return ""


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for an accuracy of k/n (Wilson score interval)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


async def ask(client, url, model, q, sem, max_tokens) -> dict:
    # Room for "The correct answer is: B" (a 5-token limit cut that off mid-sentence).
    body = {"model": model, "messages": [{"role": "user", "content": build_prompt(q)}],
            "max_tokens": max_tokens, "temperature": 0}
    async with sem:
        for attempt in range(3):
            t0 = time.perf_counter()
            try:
                r = await client.post(f"{url}/v1/chat/completions", json=body)
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"] or ""
                pred = parse_letter(text, q["choices"])
                return {"id": q["id"], "answer": q["answer"], "pred": pred,
                        "correct": int(pred == q["answer"]), "raw": text[:120].replace("\n", " "),
                        "latency_s": round(time.perf_counter() - t0, 3)}
            except (httpx.HTTPError, KeyError, ValueError):
                if attempt == 2:
                    return {"id": q["id"], "answer": q["answer"], "pred": "", "correct": 0,
                            "raw": "<request failed>", "latency_s": round(time.perf_counter() - t0, 3)}
                await asyncio.sleep(1 + attempt)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="e.g. llamacpp_q8_0, vllm_t4_awq")
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"))
    ap.add_argument("--data", default="eval/arc_easy_1000.jsonl")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke test)")
    ap.add_argument("--max-tokens", type=int, default=48, help="reply length limit (keep the same for all models)")
    args = ap.parse_args()

    qs = [json.loads(line) for line in open(args.data) if line.strip()]
    if args.limit:
        qs = qs[: args.limit]
    sem = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency * 2)
    print(f"{args.label}: {len(qs)} questions, concurrency {args.concurrency} -> {args.url}")
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=120, limits=limits) as client:
        rows = await asyncio.gather(*[ask(client, args.url, args.model, q, sem, args.max_tokens) for q in qs])
    wall = time.perf_counter() - t0

    os.makedirs("results/quality", exist_ok=True)
    with open(f"results/quality/{args.label}_preds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n, k = len(rows), sum(r["correct"] for r in rows)
    lo, hi = wilson(k, n)
    failed = sum(r["raw"] == "<request failed>" for r in rows)
    unparsed = sum(r["pred"] == "" for r in rows) - failed
    summary = {"label": args.label, "max_tokens": args.max_tokens, "n": n, "correct": k, "accuracy": round(k / n, 4),
               "ci95_low": round(lo, 4), "ci95_high": round(hi, 4), "unparsed": unparsed,
               "failed": failed, "wall_s": round(wall, 1), "questions_per_s": round(n / wall, 2),
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    # Append; if an older summary file has different columns, merge them instead of corrupting it.
    path = "results/quality_summary.csv"
    rows_out, fields = [], list(summary.keys())
    if os.path.exists(path):
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            rows_out = list(r)
            fields = list(r.fieldnames or []) + [k for k in summary if k not in (r.fieldnames or [])]
    rows_out.append(summary)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows_out)

    print(f"accuracy {k}/{n} = {k / n:.1%}  (95% CI {lo:.1%}-{hi:.1%}) | unparsed {unparsed} | "
          f"failed {failed} | {wall:.0f}s ({n / wall:.1f} q/s)")


if __name__ == "__main__":
    asyncio.run(main())
