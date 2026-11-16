"""What did users see, second by second, while we broke something?

Reads a Locust raw CSV (one row per request) plus the drill's event log and prints a timeline in
5-second windows: answers from the cache (fast), answers from the model, and each kind of error.

Usage: python scripts/chaos_timeline.py results/raw/drill_redis_u10.csv results/raw/drill_redis_events.txt
Event log lines: "<unix_ts> <text>", e.g. "1763221000.5 redis STOPPED"
"""
import sys

import pandas as pd

FAST_S = 0.2          # an answer this fast can only have come from the cache
WINDOW_S = 5


def main() -> None:
    raw, events_file = sys.argv[1], sys.argv[2]
    df = pd.read_csv(raw, keep_default_na=False)
    t0 = df["start_ts"].min()
    df["t"] = ((df["end_ts"] - t0) // WINDOW_S * WINDOW_S).astype(int)     # bucket by completion time
    ok = df["status"].astype(str) == "200"
    df["kind"] = "error:" + df["error"].astype(str).replace("", "unknown")
    df.loc[ok & (df["latency_s"] < FAST_S), "kind"] = "ok_cache"
    df.loc[ok & (df["latency_s"] >= FAST_S), "kind"] = "ok_model"

    events = []
    for line in open(events_file):
        ts, _, text = line.strip().partition(" ")
        if ts:
            events.append((int((float(ts) - t0) // WINDOW_S * WINDOW_S), text))

    table = df.pivot_table(index="t", columns="kind", values="latency_s", aggfunc="count", fill_value=0)
    cols = [c for c in ("ok_cache", "ok_model") if c in table] + sorted(c for c in table if c.startswith("error:"))
    table = table.reindex(columns=cols, fill_value=0)

    print(f"{'t (s)':>6}  " + "  ".join(f"{c:>14}" for c in cols) + "   event")
    for t, row in table.iterrows():
        marks = "; ".join(text for et, text in events if et == t)
        print(f"{t:>6}  " + "  ".join(f"{int(row[c]):>14}" for c in cols) + (f"   <-- {marks}" if marks else ""))

    errs = df[~ok]
    print(f"\ntotal {len(df)} requests | ok {int(ok.sum())} ({ok.mean():.1%}) | "
          f"errors {len(errs)}: {errs['error'].value_counts().to_dict() or 'none'}")
    if len(errs):
        print(f"errors between t={int(errs['end_ts'].min() - t0)}s and t={int(errs['end_ts'].max() - t0)}s")


if __name__ == "__main__":
    main()
