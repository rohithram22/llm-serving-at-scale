#!/usr/bin/env bash
# Load-test Version A at several user counts, with a FRESH server for each run
# (otherwise the previous run's backlog of queued requests pollutes the next one).
#
# Usage:
#   bash scripts/run_version_a_load.sh                                  # baseline (no admission control)
#   LABEL=version_a_guarded MAX_QUEUE=2 QUEUE_TIMEOUT_S=6 CANCEL_ON_DISCONNECT=1 \
#     bash scripts/run_version_a_load.sh                                # Step 4b
#   DURATION=60 USERS_LIST="5" WARMUP_S=10 bash scripts/run_version_a_load.sh   # smoke test
#
# Any env var the server understands (MAX_QUEUE, QUEUE_TIMEOUT_S, ...) is passed through.
set -uo pipefail
export TORCH_THREADS="${TORCH_THREADS:-2}"
export MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
LABEL="${LABEL:-version_a}"
DURATION="${DURATION:-180}"
USERS_LIST="${USERS_LIST:-10 100 1000}"
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)"

stop_server() { pkill -f "uvicorn app:app" 2>/dev/null; sleep 3; }

for U in $USERS_LIST; do
  stop_server
  HF_HUB_OFFLINE=1 setsid nohup uvicorn app:app --app-dir serving/naive_hf \
    --host 0.0.0.0 --port 8000 > "results/raw/${LABEL}_u${U}_server.log" 2>&1 < /dev/null &
  if ! bash scripts/wait_ready.sh; then
    tail -20 "results/raw/${LABEL}_u${U}_server.log"; stop_server; exit 1
  fi
  grep -m1 "READY" "results/raw/${LABEL}_u${U}_server.log"
  if ! bash loadtest/run_load.sh "$LABEL" "$U" "$DURATION"; then
    echo "load test failed at ${U} users; stopping."; stop_server; exit 1
  fi
done
stop_server

echo
echo "=== summary (results/loadtest_summary.csv)"
python - <<'PY'
import pandas as pd
df = pd.read_csv("results/loadtest_summary.csv")
cols = [c for c in ["label", "users", "n_ok", "error_rate", "p50_s", "p95_s", "throughput_rps",
                    "goodput_rps", "server_completed", "client_ok_total", "server_rejected",
                    "server_cancelled", "server_waiting_at_end"] if c in df.columns]
print(df[cols].to_string(index=False))
PY