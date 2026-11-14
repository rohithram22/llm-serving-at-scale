"""Paired comparison of quantized models against a baseline on the SAME questions.

Accuracy alone is noisy (+-3% at n=1000). Comparing answer-by-answer is far more sensitive:
  agreement  = % of questions where both models gave the same letter
  b / c      = questions only the baseline got right / only the other model got right
  McNemar p  = chance of seeing a b-vs-c split this lopsided if the models were equally good
               (p < 0.05 -> a real quality difference, not luck)

Usage: python eval/compare_quality.py llamacpp_fp16 llamacpp_q8_0 llamacpp_q4_k_m
       (first label = baseline)
"""
import csv
import math
import sys


def load(label: str) -> dict:
    with open(f"results/quality/{label}_preds.csv") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test = binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    base_label, others = sys.argv[1], sys.argv[2:]
    base = load(base_label)
    out = []
    print(f"baseline: {base_label}\n")
    print(f"{'model':<22} {'accuracy':>9} {'vs base':>8} {'agree':>7} {'b':>4} {'c':>4} {'McNemar p':>10}  verdict")
    for label in [base_label] + others:
        other = load(label)
        ids = sorted(set(base) & set(other))
        acc = sum(int(other[i]["correct"]) for i in ids) / len(ids)
        base_acc = sum(int(base[i]["correct"]) for i in ids) / len(ids)
        agree = sum(base[i]["pred"] == other[i]["pred"] for i in ids) / len(ids)
        b = sum(base[i]["correct"] == "1" and other[i]["correct"] == "0" for i in ids)
        c = sum(base[i]["correct"] == "0" and other[i]["correct"] == "1" for i in ids)
        p = mcnemar_exact(b, c)
        verdict = "(baseline)" if label == base_label else (
            "no significant difference" if p >= 0.05 else ("significantly WORSE" if b > c else "significantly better"))
        print(f"{label:<22} {acc:>9.1%} {acc - base_acc:>+8.1%} {agree:>7.1%} {b:>4} {c:>4} {p:>10.3f}  {verdict}")
        out.append({"baseline": base_label, "label": label, "n": len(ids), "accuracy": round(acc, 4),
                    "delta_vs_baseline": round(acc - base_acc, 4), "agreement": round(agree, 4),
                    "only_baseline_correct": b, "only_model_correct": c, "mcnemar_p": round(p, 4),
                    "verdict": verdict})
    path = f"results/quality_comparison_{base_label}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
