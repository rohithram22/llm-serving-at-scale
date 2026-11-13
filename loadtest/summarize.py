"""Summarize one load test from its raw per-request CSV and append to the summary table.

Usage:
  python loadtest/summarize.py --raw results/raw/version_a_u10.csv --label version_a --users 10 \
      [--warmup 30] [--slo 15] [--server-stats FILE] [--server-metrics FILE] \
      [--locust-cpu PCT] [--out results/loadtest_summary.csv]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

FIELDS = ["label", "users", "window_s", "n_requests", "n_ok", "error_rate", "errors",
          "throughput_rps", "goodput_rps", "slo_s", "p50_s", "p95_s", "p99_s", "mean_s",
          "p99_reliable", "tokens_per_s", "mean_completion_tokens", "mean_queue_s",
          "mean_compute_s", "client_ok_total", "server_completed", "server_rejected",
          "server_cancelled", "server_waiting_at_end", "avg_batch", "locust_cpu_pct", "timestamp"]


def read_prometheus(path: str | None) -> dict:
    """Parse Prometheus text format into {metric_name: value}, dropping prefixes like
    'llamacpp:' and labels. Returns {} if the file is missing or empty."""
    vals: dict = {}
    if not path or not os.path.exists(path):
        return vals
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        name = name.split("{")[0].split(":")[-1]
        try:
            vals[name] = float(value)
        except ValueError:
            pass
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--users", type=int, required=True)
    ap.add_argument("--warmup", type=float, default=30)
    ap.add_argument("--slo", type=float, default=15)
    ap.add_argument("--server-stats")
    ap.add_argument("--server-metrics")
    ap.add_argument("--locust-cpu", type=float)
    ap.add_argument("--out", default="results/loadtest_summary.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.raw)
    if df.empty:
        sys.exit(f"no requests recorded in {args.raw}")
    for col in ("latency_s", "completion_tokens", "queue_ms", "compute_ms"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    t0 = df["start_ts"].min()
    m = df[df["start_ts"] >= t0 + args.warmup]          # steady state only
    if m.empty:
        sys.exit("no requests after warm-up: run longer or lower --warmup")
    window = max(m["end_ts"].max() - (t0 + args.warmup), 1e-9)

    ok = m[m["status"] == 200]
    lat = ok["latency_s"].dropna()

    def pct(q):
        return round(float(np.percentile(lat, q)), 2) if len(lat) else None

    errors = m.loc[m["status"] != 200, "error"].fillna("unknown").value_counts().to_dict()

    server = {}
    if args.server_stats and os.path.exists(args.server_stats):
        try:
            server = json.load(open(args.server_stats))
        except json.JSONDecodeError:
            server = {}
    metrics = read_prometheus(args.server_metrics)
    rejected = (server.get("rejected_queue_full", 0) or 0) + (server.get("rejected_queue_timeout", 0) or 0)
    cancelled = (server.get("cancelled_while_queued", 0) or 0) + (server.get("cancelled_mid_generation", 0) or 0)

    row = {
        "label": args.label, "users": args.users, "window_s": round(window, 1),
        "n_requests": len(m), "n_ok": len(ok),
        "error_rate": round(1 - len(ok) / len(m), 4),
        "errors": json.dumps(errors),
        "throughput_rps": round(len(ok) / window, 4),
        "goodput_rps": round(int((lat <= args.slo).sum()) / window, 4),
        "slo_s": args.slo,
        "p50_s": pct(50), "p95_s": pct(95), "p99_s": pct(99),
        "mean_s": round(float(lat.mean()), 2) if len(lat) else None,
        "p99_reliable": len(lat) >= 100,
        "tokens_per_s": round(float(ok["completion_tokens"].sum()) / window, 2),
        "mean_completion_tokens": round(float(ok["completion_tokens"].mean()), 1) if len(ok) else None,
        "mean_queue_s": round(float(ok["queue_ms"].mean()) / 1000, 2) if ok["queue_ms"].notna().any() else None,
        "mean_compute_s": round(float(ok["compute_ms"].mean()) / 1000, 2) if ok["compute_ms"].notna().any() else None,
        "client_ok_total": int((df["status"] == 200).sum()),
        "server_completed": server.get("completed"),
        "server_rejected": rejected if server else None,
        "server_cancelled": cancelled if server else None,
        "server_waiting_at_end": server.get("waiting", metrics.get("requests_deferred")),
        "avg_batch": round(metrics["n_busy_slots_per_decode"], 2) if "n_busy_slots_per_decode" in metrics else None,
        "locust_cpu_pct": args.locust_cpu,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Append. If an older summary file has fewer columns, merge instead of corrupting it.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new = pd.DataFrame([row], columns=FIELDS)
    if os.path.exists(args.out):
        new = pd.concat([pd.read_csv(args.out), new], ignore_index=True)
    new.to_csv(args.out, index=False)

    # ---- human-readable report
    print(f"\n=== {args.label} @ {args.users} users "
          f"(steady-state window {window:.0f}s, first {args.warmup:.0f}s excluded)")
    print(f"requests {len(m)} | ok {len(ok)} | error rate {row['error_rate']:.1%} {errors or ''}")
    if len(lat):
        flag = "" if row["p99_reliable"] else f"  <- p99 unreliable (only {len(lat)} samples)"
        print(f"latency (ok): p50 {row['p50_s']}s  p95 {row['p95_s']}s  p99 {row['p99_s']}s{flag}")
    print(f"throughput {row['throughput_rps']} req/s | goodput (<= {args.slo:.0f}s) "
          f"{row['goodput_rps']} req/s | {row['tokens_per_s']} tokens/s")
    if row["mean_queue_s"] is not None:
        print(f"server timing (ok): mean queue {row['mean_queue_s']}s, mean compute {row['mean_compute_s']}s")
    if server:
        print(f"server: completed {server.get('completed')} (client received {row['client_ok_total']}) "
              f"| rejected {rejected} | cancelled {cancelled} | still queued at end {server.get('waiting')}")
    if metrics:
        print(f"server metrics: avg batch size {row['avg_batch']} | "
              f"queued (deferred) at end {metrics.get('requests_deferred')} | "
              f"tokens generated {metrics.get('tokens_predicted_total')}")
    if args.locust_cpu is not None:
        print(f"locust CPU: {args.locust_cpu:.0f}% of one core")
    print(f"appended -> {args.out}")


if __name__ == "__main__":
    main()