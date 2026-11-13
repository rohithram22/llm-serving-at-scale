#!/usr/bin/env bash
# Load-test Version B (llama.cpp, CPU) with the same harness as Version A.
# Fresh server per run so one run's backlog can't pollute the next.
#
# Usage: NP=4 bash scripts/run_llamacpp_load.sh
#        NP=4 DURATION=60 USERS_LIST="5" WARMUP_S=10 bash scripts/run_llamacpp_load.sh   # smoke
set -uo pipefail
export QUANT="${QUANT:-fp16}" THREADS="${THREADS:-2}" NP="${NP:-4}"
LABEL="${LABEL:-llamacpp_${QUANT}_np${NP}}"
DURATION="${DURATION:-180}"
USERS_LIST="${USERS_LIST:-10 100 1000}"
URL="http://localhost:8080"
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)"

for U in $USERS_LIST; do
  bash serving/llamacpp/run_server.sh || exit 1
  if ! READY_PATH=/health bash scripts/wait_ready.sh "$URL" 300; then
    docker logs --tail 30 llamacpp; exit 1
  fi
  python scripts/concurrency_probe.py --url "$URL" -n 1 > /dev/null   # warm-up
  if ! bash loadtest/run_load.sh "$LABEL" "$U" "$DURATION" "$URL"; then
    echo "load test failed at ${U} users"; docker logs --tail 30 llamacpp
    bash serving/llamacpp/run_server.sh stop; exit 1
  fi
  docker logs llamacpp > "results/raw/${LABEL}_u${U}_server.log" 2>&1
done
bash serving/llamacpp/run_server.sh stop
python loadtest/show_summary.py