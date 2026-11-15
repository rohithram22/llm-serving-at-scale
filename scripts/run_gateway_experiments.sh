#!/usr/bin/env bash
# Step 8: what the gateway buys us, measured with the same Locust harness as every other step.
#
#   E1  ADMISSION CONTROL in front of llama.cpp (8 slots) at 10 / 100 / 1000 users
#       -> compare with llamacpp_fp16_np8, which collapsed to goodput 0 at 100+ users (Step 5)
#   E2  REDIS CACHE in front of llama.cpp at 0% / 20% / 50% repeated prompts (100 users)
#   E3  MICRO-BATCHING in front of Version A (static batching, batch 4 and 8) at 10 users
#       -> compare with version_a_guarded (Step 4b)
#
# Usage: bash scripts/run_gateway_experiments.sh              # all three, ~45 min
#        EXPERIMENTS="E2" bash scripts/run_gateway_experiments.sh
#        DURATION=60 EXPERIMENTS="E1" USERS_E1="10" bash scripts/run_gateway_experiments.sh   # smoke
set -uo pipefail
EXPERIMENTS="${EXPERIMENTS:-E1 E2 E3}"
DURATION="${DURATION:-180}"
USERS_E1="${USERS_E1:-10 100 1000}"
REPEATS_E2="${REPEATS_E2:-0 0.2 0.5}"
BATCHES_E3="${BATCHES_E3:-4 8}"
GW="http://localhost:9000"
LLAMA="http://localhost:8080"
VERSION_A="http://localhost:8000"
export THREADS="${THREADS:-2}" TORCH_THREADS="${TORCH_THREADS:-2}"
# Admission settings sized like Step 4b: llama.cpp does ~1.2 req/s with 8 slots; the SLO is 15 s and a
# request takes ~5-7 s, so it may wait ~6 s; in 6 s ~8 requests finish -> a queue of 8.
ADMIT="MAX_INFLIGHT=8 MAX_QUEUE=8 QUEUE_TIMEOUT_S=6"
mkdir -p results/raw
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)"

stop_gateway() { pkill -f "[u]vicorn app:app --app-dir serving/gateway" 2>/dev/null; sleep 2; }

start_gateway() {  # $1 = tag for the log file, rest = VAR=value settings
  local tag="$1"; shift
  stop_gateway
  setsid nohup env "$@" uvicorn app:app --app-dir serving/gateway --host 0.0.0.0 --port 9000 \
    > "results/raw/gateway_${tag}.log" 2>&1 < /dev/null &
  if ! READY_PATH=/ready bash scripts/wait_ready.sh "$GW" 120; then
    tail -20 "results/raw/gateway_${tag}.log"; exit 1
  fi
  grep -m1 "gateway ->" "results/raw/gateway_${tag}.log"
}

start_llamacpp() {
  NP=8 QUANT=fp16 bash serving/llamacpp/run_server.sh > /dev/null || exit 1
  READY_PATH=/health bash scripts/wait_ready.sh "$LLAMA" 300 > /dev/null || { docker logs --tail 20 llamacpp; exit 1; }
  python scripts/concurrency_probe.py --url "$LLAMA" -n 1 > /dev/null        # warm-up
}

start_redis() {
  docker rm -f redis > /dev/null 2>&1
  docker run -d --rm --name redis -p 6379:6379 redis:7-alpine > /dev/null || exit 1
  until docker exec redis redis-cli ping 2>/dev/null | grep -q PONG; do sleep 1; done
  echo "redis ready"
}

cleanup() {
  stop_gateway
  bash serving/llamacpp/run_server.sh stop > /dev/null 2>&1
  docker rm -f redis > /dev/null 2>&1
  pkill -f "[u]vicorn app:app --app-dir serving/naive_hf" 2>/dev/null
}
trap cleanup EXIT

for EXP in $EXPERIMENTS; do
  case "$EXP" in
  E1)
    echo; echo "#################### E1: admission control in front of llama.cpp ####################"
    for U in $USERS_E1; do
      start_llamacpp
      start_gateway "e1_u${U}" BACKEND_URL="$LLAMA" $ADMIT
      bash loadtest/run_load.sh gw_llamacpp_admit "$U" "$DURATION" "$GW" || exit 1
    done
    ;;
  E2)
    echo; echo "#################### E2: Redis response cache (100 users) ####################"
    start_redis
    for R in $REPEATS_E2; do
      start_llamacpp
      docker exec redis redis-cli FLUSHALL > /dev/null            # every run starts with a cold cache
      start_gateway "e2_r${R}" BACKEND_URL="$LLAMA" REDIS_URL="redis://localhost:6379/0" $ADMIT
      REPEAT_RATIO="$R" bash loadtest/run_load.sh "gw_llamacpp_cache_r${R}" 100 "$DURATION" "$GW" || exit 1
    done
    docker rm -f redis > /dev/null 2>&1
    ;;
  E3)
    echo; echo "#################### E3: micro-batching in front of Version A ####################"
    bash serving/llamacpp/run_server.sh stop > /dev/null 2>&1
    for B in $BATCHES_E3; do
      pkill -f "[u]vicorn app:app --app-dir serving/naive_hf" 2>/dev/null; sleep 3
      MAX_CONCURRENT=1 HF_HUB_OFFLINE=1 setsid nohup uvicorn app:app --app-dir serving/naive_hf \
        --host 0.0.0.0 --port 8000 > "results/raw/version_a_for_batch${B}.log" 2>&1 < /dev/null &
      bash scripts/wait_ready.sh "$VERSION_A" 300 > /dev/null || { tail -20 "results/raw/version_a_for_batch${B}.log"; exit 1; }
      # in flight = the batch being computed + the next one forming; then the usual bounded queue
      start_gateway "e3_b${B}" BACKEND_URL="$VERSION_A" BACKEND_READY_PATH=/ready BATCH_MODE=version_a \
        BATCH_MAX="$B" BATCH_WINDOW_MS=50 MAX_INFLIGHT=$(( B * 2 )) MAX_QUEUE="$B" QUEUE_TIMEOUT_S=6
      bash loadtest/run_load.sh "gw_version_a_batch${B}" 10 "$DURATION" "$GW" || exit 1
    done
    pkill -f "[u]vicorn app:app --app-dir serving/naive_hf" 2>/dev/null
    ;;
  *) echo "unknown experiment $EXP"; exit 1 ;;
  esac
done

echo; echo "#################### SUMMARY ####################"
python loadtest/show_summary.py version_a_guarded gw_version_a llamacpp_fp16_np8 gw_llamacpp
