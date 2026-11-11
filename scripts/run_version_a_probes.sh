#!/usr/bin/env bash
# Run the Version A concurrency experiments end-to-end and save results.
# Usage: bash scripts/run_version_a_probes.sh
#        TORCH_THREADS=4 bash scripts/run_version_a_probes.sh   # override
set -uo pipefail
export TORCH_THREADS="${TORCH_THREADS:-2}"   # = physical cores here; measured fastest
OUT=results/version_a_probe.txt
mkdir -p results/raw
: > "$OUT"

start_server() {  # $1 = MAX_CONCURRENT
  pkill -f "uvicorn app:app" 2>/dev/null; sleep 3
  MAX_CONCURRENT="$1" HF_HUB_OFFLINE=1 setsid nohup uvicorn app:app --app-dir serving/naive_hf \
    --host 0.0.0.0 --port 8000 > "results/raw/version_a_mc$1.log" 2>&1 < /dev/null &
  if ! bash scripts/wait_ready.sh; then
    echo "server failed to start (MAX_CONCURRENT=$1)"; tail -20 "results/raw/version_a_mc$1.log"; exit 1
  fi
}

run() {  # $1 = label, $2 = number of concurrent requests
  echo "### $1" | tee -a "$OUT"
  python scripts/concurrency_probe.py -n "$2" 2>&1 | tee -a "$OUT"
  echo | tee -a "$OUT"
}

{
  echo "machine: $(nproc) vCPUs, $(free -h | awk '/Mem:/{print $2}') RAM, torch_threads=$TORCH_THREADS"
  echo "date: $(date -u +%FT%TZ)"
  echo
} | tee -a "$OUT"

start_server 1
run "serial (MAX_CONCURRENT=1), n=1" 1
run "serial (MAX_CONCURRENT=1), n=4" 4

start_server 4
run "parallel threads (MAX_CONCURRENT=4), n=4" 4
curl -sS localhost:8000/stats | python -m json.tool | tee -a "$OUT"

pkill -f "uvicorn app:app"
echo "saved $OUT"