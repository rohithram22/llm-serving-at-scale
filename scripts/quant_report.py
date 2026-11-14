"""One table for Step 7a: size, memory, speed, quality and capacity per quantization level.

Reads what run_quant_cpu.sh produced; writes results/quant_cpu_summary.csv.
Usage: python scripts/quant_report.py [fp16 q8_0 q4_k_m]
"""
import os
import re
import sys

import pandas as pd

QUANTS = sys.argv[1:] or ["fp16", "q8_0", "q4_k_m"]
THROUGHPUT_RE = re.compile(r"throughput ([\d.]+) req/s \| ([\d.]+) tokens/s")


def probe_tokens_per_s(quant: str) -> dict:
    """tokens/s per probe section, e.g. {'slots=1 (no batching), n=1': 27.8, ...}"""
    path = f"results/llamacpp_{quant}_probe.txt"
    out, section = {}, None
    if not os.path.exists(path):
        return out
    for line in open(path):
        if line.startswith("### "):
            section = line[4:].strip()
        m = THROUGHPUT_RE.search(line)
        if m and section:
            out[section] = float(m.group(2))
    return out


def to_mb(s: str) -> float | None:
    m = re.match(r"([\d.]+)\s*([KMG]i?B)", str(s))
    if not m:
        return None
    return float(m.group(1)) * {"KiB": 1 / 1024, "KB": 1 / 1000, "MiB": 1, "MB": 1,
                                "GiB": 1024, "GB": 1000}[m.group(2)]


def latest(df: pd.DataFrame, **where) -> pd.Series | None:
    for k, v in where.items():
        df = df[df[k] == v]
    return df.iloc[-1] if len(df) else None


mem = pd.read_csv("results/quant_cpu_memory.csv") if os.path.exists("results/quant_cpu_memory.csv") else pd.DataFrame()
qual = pd.read_csv("results/quality_summary.csv") if os.path.exists("results/quality_summary.csv") else pd.DataFrame()
comp_path = "results/quality_comparison_llamacpp_fp16.csv"
comp = pd.read_csv(comp_path) if os.path.exists(comp_path) else pd.DataFrame()
load = pd.read_csv("results/loadtest_summary.csv") if os.path.exists("results/loadtest_summary.csv") else pd.DataFrame()

rows = []
for q in QUANTS:
    tps = probe_tokens_per_s(q)
    m = latest(mem, quant=q) if len(mem) else None
    a = latest(qual, label=f"llamacpp_{q}") if len(qual) else None
    c = latest(comp, label=f"llamacpp_{q}") if len(comp) else None
    l = latest(load, label=f"llamacpp_{q}_np8", users=10) if len(load) else None
    one = tps.get("slots=1 (no batching), n=1")
    eight = tps.get("slots=8 (continuous batching), n=8")
    rows.append({
        "quant": q,
        "file_mb": m["file_mb"] if m is not None else None,
        "ram_mb": round(to_mb(m["container_mem"]), 0) if m is not None and to_mb(m["container_mem"]) else None,
        "tok_s_batch1": one,
        "tok_s_batch8": eight,
        "accuracy": a["accuracy"] if a is not None else None,
        "acc_ci95": f"{a['ci95_low']:.3f}-{a['ci95_high']:.3f}" if a is not None else None,
        "agree_vs_fp16": c["agreement"] if c is not None else None,
        "mcnemar_p": c["mcnemar_p"] if c is not None else None,
        "goodput_10u": l["goodput_rps"] if l is not None else None,
        "p95_10u_s": l["p95_s"] if l is not None else None,
    })

df = pd.DataFrame(rows)
base = df[df["quant"] == "fp16"]
if len(base) and pd.notna(base.iloc[0]["tok_s_batch1"]):
    for col in ("tok_s_batch1", "tok_s_batch8", "goodput_10u"):
        if pd.notna(base.iloc[0][col]):
            df[f"{col}_x_fp16"] = (df[col] / base.iloc[0][col]).round(2)
df.to_csv("results/quant_cpu_summary.csv", index=False)
pd.set_option("display.width", 250)
print(df.to_string(index=False))
print("\nsaved results/quant_cpu_summary.csv")
