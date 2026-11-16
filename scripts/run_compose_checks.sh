#!/usr/bin/env bash
# Step 9: build + start the stack with Docker Compose, then check it the same way as Steps 4-8.
#
#   1. start-up:   build, start, time until every container is healthy
#   2. smoke:      one request twice through the gateway -> MISS, then HIT
#   3. load test:  100 users, 20% repeats (same as Step 8's E2) + memory/CPU per container
#   4. drill A:    stop Redis for 30 s under load   -> expect NO errors (cache fails open)
#   5. drill B:    restart llama.cpp under load     -> how long are users affected?
#
# Usage: bash scripts/run_compose_checks.sh            (~12 min)
#        DURATION=60 SKIP_DRILLS=1 bash scripts/run_compose_checks.sh
set -uo pipefail
DURATION="${DURATION:-180}"
DRILL_S="${DRILL_S:-120}"
GW="http://localhost:9000"
mkdir -p results/raw
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)"

note() { echo "$(date +%s.%N) $*" >> "$EVENTS"; echo "    [t+$(( $(date +%s) - DRILL_T0 ))s] $*"; }

wait_gateway() {   # gateway /ready = gateway up AND llama.cpp loaded
  local t0; t0=$(date +%s)
  until curl -sf "$GW/ready" > /dev/null; do
    (( $(date +%s) - t0 > 300 )) && { { echo "stack not ready after 300s"; docker compose ps; docker compose logs --tail 20; } >&2; exit 1; }
    sleep 1
  done
  echo $(( $(date +%s) - t0 ))
}

echo "########## 0. clean up anything from earlier steps (same ports) ##########"
docker rm -f llamacpp redis > /dev/null 2>&1
pkill -f "[u]vicorn app:app" 2>/dev/null; sleep 1
[[ -f "models/qwen2.5-0.5b-instruct-${QUANT:-fp16}.gguf" ]] || { echo "model file missing: python scripts/download_gguf.py ${QUANT:-fp16}"; exit 1; }

echo; echo "########## 1. build + start ##########"
docker compose build gateway 2>&1 | tail -3
docker images llm-gateway:latest --format "gateway image: {{.Repository}}:{{.Tag}}  {{.Size}}"
T0=$(date +%s)
docker compose up -d || { docker compose logs --tail 30; exit 1; }
READY_S=$(wait_gateway) || exit 1
echo "stack ready ${READY_S}s after 'up' (total $(( $(date +%s) - T0 ))s)"
docker compose ps --format "table {{.Service}}\t{{.Status}}\t{{.Ports}}"

echo; echo "########## 2. smoke test: same request twice ##########"
REQ='{"model":"qwen","messages":[{"role":"user","content":"What is Docker in one sentence?"}],"max_tokens":40,"temperature":0}'
for i in 1 2; do
  curl -s -D /tmp/h.txt -o /tmp/b.json -X POST "$GW/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ"
  echo "request $i: $(grep -i '^x-cache' /tmp/h.txt | tr -d '\r') | $(grep -i '^x-gateway-ms' /tmp/h.txt | tr -d '\r')"
done
python -c "import json; print('answer:', json.load(open('/tmp/b.json'))['choices'][0]['message']['content'][:120])"

echo; echo "########## 3. load test through compose: 100 users, 20% repeats ##########"
docker compose exec -T redis redis-cli FLUSHALL > /dev/null
( sleep $(( DURATION / 2 )); docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" \
    > results/compose_docker_stats.txt ) &
REPEAT_RATIO=0.2 bash loadtest/run_load.sh compose_llamacpp_cache_r0.2 100 "$DURATION" "$GW" || exit 1
wait
echo "--- resources per container, mid-test (for the capacity plan) ---"
cat results/compose_docker_stats.txt

if [[ "${SKIP_DRILLS:-0}" == "1" ]]; then exit 0; fi

echo; echo "########## 4. drill A: Redis down for 30 s (10 users, 50% repeats) ##########"
EVENTS=results/raw/drill_redis_events.txt; : > "$EVENTS"
docker compose exec -T redis redis-cli FLUSHALL > /dev/null
DRILL_T0=$(date +%s)
( sleep 45; note "redis STOPPED"; docker compose stop redis > /dev/null 2>&1
  sleep 30; docker compose start redis > /dev/null 2>&1; note "redis STARTED (empty: no persistence)" ) &
REPEAT_RATIO=0.5 bash loadtest/run_load.sh drill_redis 10 "$DRILL_S" "$GW" > results/raw/drill_redis_summary.txt
wait
python scripts/chaos_timeline.py results/raw/drill_redis_u10.csv "$EVENTS"
curl -s "$GW/stats" | python -c "import json,sys; s=json.load(sys.stdin); print('gateway cache_errors:', s['cache_errors'], '| hits:', s['cache_hits'])"

echo; echo "########## 5. drill B: restart llama.cpp under load (10 users, 50% repeats) ##########"
EVENTS=results/raw/drill_backend_events.txt; : > "$EVENTS"
docker compose exec -T redis redis-cli FLUSHALL > /dev/null
DRILL_T0=$(date +%s)
( sleep 45; note "llama.cpp RESTART begins"; docker compose restart llamacpp > /dev/null 2>&1
  note "llama.cpp container restarted (loading model)"
  until curl -sf "$GW/ready" > /dev/null; do sleep 0.5; done; note "gateway /ready OK again" ) &
REPEAT_RATIO=0.5 bash loadtest/run_load.sh drill_backend 10 "$DRILL_S" "$GW" > results/raw/drill_backend_summary.txt
wait
python scripts/chaos_timeline.py results/raw/drill_backend_u10.csv "$EVENTS"

echo; echo "########## done. Compare with Step 8: ##########"
python loadtest/show_summary.py gw_llamacpp_cache_r0.2 compose_llamacpp_cache_r0.2
echo "stack is still running: 'docker compose down' to stop it"
