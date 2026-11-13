#!/usr/bin/env bash
# Version B (CPU): how continuous batching changes the concurrency picture.
# Same prompt and 32 output tokens as the Version A probes, so the numbers compare directly.
#
# Usage: bash scripts/run_llamacpp_probes.sh          (QUANT=fp16 THREADS=2 by default)
set -uo pipefail
export QUANT="${QUANT:-fp16}" THREADS="${THREADS:-2}"
URL="http://localhost:8080"
OUT="results/llamacpp_${QUANT}_probe.txt"
mkdir -p results
: > "$OUT"

start() {  # $1 = number of slots
  NP="$1" bash serving/llamacpp/run_server.sh || exit 1
  if ! READY_PATH=/health bash scripts/wait_ready.sh "$URL" 300; then
    docker logs --tail 30 llamacpp; exit 1
  fi
  python scripts/concurrency_probe.py --url "$URL" -n 1 > /dev/null   # warm-up request
}

run() {  # $1 = label, $2 = concurrent requests
  echo "### $1" | tee -a "$OUT"
  python scripts/concurrency_probe.py --url "$URL" -n "$2" 2>&1 | tee -a "$OUT"
  echo | tee -a "$OUT"
}

{
  echo "machine: $(nproc) vCPUs, $(free -h | awk '/Mem:/{print $2}') RAM | llama.cpp ${QUANT}, threads=${THREADS}"
  echo "date: $(date -u +%FT%TZ)"
  echo
} | tee -a "$OUT"

start 1
run "slots=1 (no batching), n=1" 1
run "slots=1 (no batching), n=4" 4
start 4
run "slots=4 (continuous batching), n=4" 4
start 8
run "slots=8 (continuous batching), n=8" 8

bash serving/llamacpp/run_server.sh stop
echo "saved $OUT"