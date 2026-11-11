#!/usr/bin/env bash
# Run ONE headless Locust test against an OpenAI-compatible server, then summarize it.
#
# Usage: bash loadtest/run_load.sh <label> <users> [duration_s] [base_url]
#   e.g. bash loadtest/run_load.sh version_a 10 180 http://localhost:8000
#
# Tunables (env): TIMEOUT_S=60 THINK_MIN=1 THINK_MAX=3 REPEAT_RATIO=0
#                 SLO_S=15 WARMUP_S=30 SPAWN_RATE=min(users,50)
set -uo pipefail

LABEL="$1"; USERS="$2"; DURATION="${3:-180}"; URL="${4:-http://localhost:8000}"
export TIMEOUT_S="${TIMEOUT_S:-60}"
SLO_S="${SLO_S:-15}"; WARMUP_S="${WARMUP_S:-30}"
SPAWN_RATE="${SPAWN_RATE:-$(( USERS < 50 ? USERS : 50 ))}"
LOCUSTFILE="${LOCUSTFILE:-loadtest/locustfile.py}"
TAG="${LABEL}_u${USERS}"
mkdir -p results/raw
export RAW_CSV="results/raw/${TAG}.csv"
STDOUT_FILE="results/raw/${TAG}_locust_stdout.txt"
TIME_FILE="results/raw/${TAG}_time.txt"
rm -f "$RAW_CSV"

# Every open connection is a file descriptor; the default limit (often 1024) is too low for 1000 users.
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)"

if [[ ! -f "$LOCUSTFILE" ]]; then
  echo "ERROR: $LOCUSTFILE not found. Run this from the repo root."; exit 1
fi

echo ">>> ${TAG}: ${USERS} users for ${DURATION}s -> ${URL}" \
     "(spawn ${SPAWN_RATE}/s, timeout ${TIMEOUT_S}s, fd limit $(ulimit -n))"

TIMEFORMAT="%R %U %S"
{ time locust -f "$LOCUSTFILE" --headless \
      -u "$USERS" -r "$SPAWN_RATE" -t "${DURATION}s" --host "$URL" \
      --stop-timeout "$TIMEOUT_S" --only-summary \
      --csv "results/raw/${TAG}_locust" --logfile "results/raw/${TAG}_locust.log" \
      > "$STDOUT_FILE" 2>&1 ; } 2> "$TIME_FILE"
LOCUST_EXIT=$?
# Note: Locust exits with code 1 whenever ANY request failed. Under overload that's
# expected, so we judge success by whether requests were recorded, not by the exit code.

if [[ ! -s "$RAW_CSV" ]] || [[ $(wc -l < "$RAW_CSV") -le 1 ]]; then
  echo "ERROR: Locust recorded no requests (exit code $LOCUST_EXIT). Last lines of its output:"
  echo "------------------------------------------------------------"
  tail -40 "$STDOUT_FILE"
  echo "------------------------------------------------------------"
  exit 1
fi

# Locust's average CPU over the run, as % of ONE core (CPU time / wall time).
read -r T_REAL T_USER T_SYS < "$TIME_FILE"
LOCUST_CPU=$(python -c "print(round(100*($T_USER+$T_SYS)/max($T_REAL,1e-9)))")

curl -sf "${URL}/stats" > "results/raw/${TAG}_server_stats.json" \
  || echo '{}' > "results/raw/${TAG}_server_stats.json"

python loadtest/summarize.py --raw "$RAW_CSV" --label "$LABEL" --users "$USERS" \
  --warmup "$WARMUP_S" --slo "$SLO_S" --locust-cpu "$LOCUST_CPU" \
  --server-stats "results/raw/${TAG}_server_stats.json"