#!/usr/bin/env bash
# Step 7a: quantization on CPU with llama.cpp. For each model file (fp16, q8_0, q4_k_m):
#   1. SPEED    - the same batching probes as Step 5 (1 / 4 / 8 slots, 32 tokens)
#   2. MEMORY   - RAM used by the server container with the model loaded
#   3. QUALITY  - 1,000 ARC-Easy questions (identical for every model)
#   4. CAPACITY - the 10-user load test (set LOAD=0 to skip)
# Every model is measured in the same session on the same machine, so they compare fairly.
#
# Usage: bash scripts/run_quant_cpu.sh                      # all three, ~45 min
#        QUANTS="q4_k_m" LOAD=0 bash scripts/run_quant_cpu.sh
#        SPEED=0 LOAD=0 bash scripts/run_quant_cpu.sh          # quality only (re-score), ~20 min
set -uo pipefail
QUANTS="${QUANTS:-fp16 q8_0 q4_k_m}"
LOAD="${LOAD:-1}"
SPEED="${SPEED:-1}"
URL="http://localhost:8080"
export THREADS="${THREADS:-2}"
mkdir -p results/raw
[[ -f eval/arc_easy_1000.jsonl ]] || { echo "Missing eval set. Run: python eval/prepare_arc.py"; exit 1; }
MEM_CSV="results/quant_cpu_memory.csv"
[[ -f "$MEM_CSV" ]] || echo "quant,file_mb,container_mem" > "$MEM_CSV"

for Q in $QUANTS; do
  MODEL_FILE="models/qwen2.5-0.5b-instruct-${Q}.gguf"
  [[ -f "$MODEL_FILE" ]] || python scripts/download_gguf.py "$Q" || exit 1
  echo; echo "#################### ${Q} ####################"

  # 1. SPEED (this restarts the server with 1, 4 and 8 slots)
  if [[ "$SPEED" == "1" ]]; then
    QUANT="$Q" bash scripts/run_llamacpp_probes.sh || exit 1
  fi

  # 2. MEMORY + 3. QUALITY, with 8 slots
  QUANT="$Q" NP=8 bash serving/llamacpp/run_server.sh || exit 1
  READY_PATH=/health bash scripts/wait_ready.sh "$URL" 300 || { docker logs --tail 30 llamacpp; exit 1; }
  python scripts/concurrency_probe.py --url "$URL" -n 1 > /dev/null          # warm-up
  MEM=$(docker stats --no-stream --format '{{.MemUsage}}' llamacpp | awk '{print $1}')
  FILE_MB=$(( $(stat -c %s "$MODEL_FILE") / 1048576 ))
  [[ "$SPEED" == "1" ]] && echo "${Q},${FILE_MB},${MEM}" >> "$MEM_CSV"
  echo "memory: file ${FILE_MB} MB, server container ${MEM}"
  python eval/mc_eval.py --label "llamacpp_${Q}" --url "$URL" --concurrency 8 || exit 1
  bash serving/llamacpp/run_server.sh stop

  # 4. CAPACITY
  if [[ "$LOAD" == "1" ]]; then
    QUANT="$Q" NP=8 USERS_LIST=10 LABEL="llamacpp_${Q}_np8" bash scripts/run_llamacpp_load.sh > /dev/null \
      || { echo "load test failed for ${Q}"; exit 1; }
    python loadtest/show_summary.py "llamacpp_${Q}_np8" | tail -1
  fi
done

echo; echo "#################### QUALITY (paired vs fp16) ####################"
OTHERS=$(for Q in $QUANTS; do [[ "$Q" != "fp16" ]] && echo "llamacpp_${Q}"; done)
python eval/compare_quality.py llamacpp_fp16 $OTHERS
echo; echo "#################### REPORT ####################"
python scripts/quant_report.py
