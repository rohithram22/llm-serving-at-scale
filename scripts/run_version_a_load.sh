#!/usr/bin/env bash
# Load-test Version A at several user counts, with a FRESH server for each run
# (otherwise the previous run's backlog of queued requests pollutes the next one).
#
# Usage: bash scripts/run_version_a_load.sh
#        DURATION=60 USERS_LIST="5" WARMUP_S=10 bash scripts/run_version_a_load.sh   # smoke test
set -uo pipefail
export TORCH_THREADS="${TORCH_THREADS:-2}"
DURATION="${DURATION:-180}"
USERS_LIST="${USERS_LIST:-10 100 1000}"
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)"

stop_server() { pkill -f "uvicorn app:app" 2>/dev/null; sleep 3; }

for U in $USERS_LIST; do
  stop_server
  MAX_CONCURRENT=1 HF_HUB_OFFLINE=1 setsid nohup uvicorn app:app --app-dir serving/naive_hf \
    --host 0.0.0.0 --port 8000 > "results/raw/version_a_u${U}_server.log" 2>&1 < /dev/null &
  if ! bash scripts/wait_ready.sh; then
    tail -20 "results/raw/version_a_u${U}_server.log"; stop_server; exit 1
  fi
  if ! bash loadtest/run_load.sh version_a "$U" "$DURATION"; then
    echo "load test failed at ${U} users; stopping."; stop_server; exit 1
  fi
done
stop_server

echo
echo "=== summary (results/loadtest_summary.csv)"
python - <<'PY'
import pandas as pd
df = pd.read_csv("results/loadtest_summary.csv")
cols = ["label", "users", "n_ok", "error_rate", "p50_s", "p95_s", "p99_s",
        "throughput_rps", "goodput_rps", "server_completed", "client_ok_total", "server_waiting_at_end"]
print(df[cols].to_string(index=False))
PY