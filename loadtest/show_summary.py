"""Print the load-test summary table (optionally only some labels).

Usage: python loadtest/show_summary.py [label_prefix ...]
"""
import sys

import pandas as pd

COLS = ["label", "users", "n_ok", "error_rate", "p50_s", "p95_s", "throughput_rps",
        "goodput_rps", "tokens_per_s", "avg_batch", "server_completed", "client_ok_total",
        "server_rejected", "server_waiting_at_end"]

df = pd.read_csv("results/loadtest_summary.csv")
if len(sys.argv) > 1:
    df = df[df["label"].str.startswith(tuple(sys.argv[1:]))]
pd.set_option("display.width", 250)
print(df[[c for c in COLS if c in df.columns]].to_string(index=False))