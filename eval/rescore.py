"""Re-score saved replies with the CURRENT parser (no server needed).

Use after changing parse rules, so every model is scored by the same rule.
Usage: python eval/rescore.py llamacpp_fp16 llamacpp_q8_0 llamacpp_q4_k_m
Updates results/quality/<label>_preds.csv in place and appends a summary row
(label kept the same, so reports pick up the newest numbers).
"""
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from mc_eval import parse_letter, wilson  # noqa: E402


def main() -> None:
    labels = sys.argv[1:]
    if not labels:
        sys.exit(__doc__)
    questions = {q["id"]: q for q in map(json.loads, open("eval/arc_easy_1000.jsonl"))}
    for label in labels:
        path = f"results/quality/{label}_preds.csv"
        rows = list(csv.DictReader(open(path)))
        changed = 0
        for r in rows:
            new = "" if r["raw"] == "<request failed>" else parse_letter(r["raw"], questions[r["id"]]["choices"])
            changed += new != r["pred"]
            r["pred"], r["correct"] = new, str(int(new == r["answer"]))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        n, k = len(rows), sum(int(r["correct"]) for r in rows)
        unparsed = sum(r["pred"] == "" for r in rows)
        lo, hi = wilson(k, n)
        summary = {"label": label, "max_tokens": "", "n": n, "correct": k, "accuracy": round(k / n, 4),
                   "ci95_low": round(lo, 4), "ci95_high": round(hi, 4), "unparsed": unparsed, "failed": "",
                   "wall_s": "", "questions_per_s": "", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S") + " rescored"}
        spath = "results/quality_summary.csv"
        existing = list(csv.DictReader(open(spath))) if os.path.exists(spath) else []
        fields = list(existing[0].keys()) if existing else list(summary)
        fields += [k2 for k2 in summary if k2 not in fields]
        with open(spath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(existing + [summary])
        print(f"{label:<20} accuracy {k / n:.1%} (95% CI {lo:.1%}-{hi:.1%}) | unparsed {unparsed} "
              f"| {changed} answers changed by the new parser")


if __name__ == "__main__":
    main()
