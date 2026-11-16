#!/usr/bin/env bash
# Step 10, part 2: what Kubernetes buys us.  ~20 min, don't touch the terminal.
#
#   1. load balancing:  20 requests -> which pod answered each?
#   2. rolling restart under load (2 replicas)  -> compare with Step 9 drill B (35 errors)
#   3. autoscaling: HPA on, 30 users for 7 min, then 4 min idle -> replicas over time
#
# Usage: bash scripts/run_k8s_checks.sh        (after scripts/setup_kind.sh)
#        SKIP_HPA=1 / SKIP_DRILL=1 to run parts
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
URL="http://localhost:30900"
NS="kubectl -n llm"
LOAD_S="${LOAD_S:-420}"; IDLE_S="${IDLE_S:-240}"; HPA_USERS="${HPA_USERS:-30}"
mkdir -p results/raw
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)"

ready_replicas() { $NS get deploy llm -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0; }
note() { echo "$(date +%s.%N) $*" >> "$EVENTS"; echo "    [t+$(( $(date +%s) - T0 ))s] $*"; }

curl -sf "$URL/ready" > /dev/null || { echo "cluster not reachable at $URL: run scripts/setup_kind.sh"; exit 1; }

echo "########## 1. load balancing: 20 different requests ##########"
for i in $(seq 1 20); do
  curl -s -D - -o /dev/null -X POST "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"q\",\"messages\":[{\"role\":\"user\",\"content\":\"Say the number $i\"}],\"max_tokens\":4,\"temperature\":0}" \
    | grep -i '^x-served-by' | tr -d '\r' | awk '{print $2}'
done | sort | uniq -c | awk '{print "   " $2 ": " $1 " requests"}'

if [[ "${SKIP_DRILL:-0}" != "1" ]]; then
  echo; echo "########## 2. rolling restart under load (2 replicas, 5 users) ##########"
  EVENTS=results/raw/k8s_rollout_events.txt; : > "$EVENTS"; T0=$(date +%s)
  ( sleep 45; note "kubectl rollout restart"; $NS rollout restart deployment/llm > /dev/null
    $NS rollout status deployment/llm --timeout=300s > /dev/null; note "rollout complete" ) &
  bash loadtest/run_load.sh k8s_rolling_restart 5 210 "$URL" > results/raw/k8s_rolling_restart_summary.txt
  wait
  python scripts/chaos_timeline.py results/raw/k8s_rolling_restart_u5.csv "$EVENTS" 10
fi

if [[ "${SKIP_HPA:-0}" != "1" ]]; then
  echo; echo "########## 3. autoscaling ##########"
  $NS apply -f deploy/k8s/hpa.yaml > /dev/null
  echo "HPA applied. Waiting for CPU metrics + scale-down to 1 idle replica (~2-3 min)..."
  for _ in $(seq 1 60); do
    [[ "$($NS get deploy llm -o jsonpath='{.spec.replicas}')" == "1" && "$(ready_replicas)" == "1" ]] \
      && $NS top pods > /dev/null 2>&1 && break
    sleep 5
  done
  $NS get hpa llm

  EVENTS=results/raw/k8s_hpa_events.txt; : > "$EVENTS"; WATCH=results/k8s_hpa_watch.txt; : > "$WATCH"
  T0=$(date +%s)
  (   # watcher: log replicas every 5 s; mark every change on the timeline
    last=""
    while true; do
      desired=$($NS get deploy llm -o jsonpath='{.spec.replicas}')
      ready=$(ready_replicas); ready=${ready:-0}
      pending=$($NS get pods -l app=llm --field-selector=status.phase=Pending --no-headers 2>/dev/null | wc -l)
      cpu=$($NS get hpa llm -o jsonpath='{.status.currentMetrics[0].containerResource.current.averageUtilization}' 2>/dev/null)
      now="desired=$desired ready=$ready pending=$pending"
      echo "$(( $(date +%s) - T0 ))s $now cpu=${cpu:-?}%" >> "$WATCH"
      [[ "$now" != "$last" ]] && note "$now (llama.cpp cpu ${cpu:-?}% of request)"
      last="$now"; sleep 5
    done
  ) &
  WATCHER=$!

  bash loadtest/run_load.sh k8s_hpa "$HPA_USERS" "$LOAD_S" "$URL" > results/raw/k8s_hpa_summary.txt
  note "load STOPPED"
  echo "--- at peak ---"
  $NS get pods -l app=llm -o wide
  pend=$($NS get pods -l app=llm --field-selector=status.phase=Pending -o name | head -1)
  [[ -n "$pend" ]] && { echo "why $pend is Pending:"; $NS describe "$pend" | grep -A3 "^Events" | tail -2; }
  $NS top pods --containers 2>/dev/null | grep -E "NAME|llamacpp" | tee results/k8s_top_pods.txt
  echo "watching scale-down for ${IDLE_S}s..."
  sleep "$IDLE_S"
  kill "$WATCHER" 2>/dev/null; wait "$WATCHER" 2>/dev/null

  echo; echo "--- goodput vs replicas over time (30 s windows) ---"
  python scripts/chaos_timeline.py results/raw/k8s_hpa_u${HPA_USERS}.csv "$EVENTS" 30
  echo; tail -n 25 results/raw/k8s_hpa_summary.txt | grep -E "===|requests|latency|goodput"
  $NS describe hpa llm | sed -n '/^Events/,$p' | tail -12
fi

echo; echo "done. Cluster still running ('kind delete cluster --name llm' to remove)."
