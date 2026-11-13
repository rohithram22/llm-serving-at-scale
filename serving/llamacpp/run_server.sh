#!/usr/bin/env bash
# Start (or restart) llama.cpp's OpenAI-compatible server in Docker, CPU only.
#
# Usage:
#   bash serving/llamacpp/run_server.sh                 # fp16, 4 slots, 2 threads
#   QUANT=q4_k_m NP=8 bash serving/llamacpp/run_server.sh
#   bash serving/llamacpp/run_server.sh stop
#
# NP = parallel slots = max sequences decoded together (continuous batching batch size).
set -euo pipefail
IMAGE="${LLAMACPP_IMAGE:-ghcr.io/ggml-org/llama.cpp:server}"
NAME="llamacpp"

if [[ "${1:-}" == "stop" ]]; then
  docker rm -f "$NAME" > /dev/null 2>&1 || true
  echo "llama.cpp stopped"; exit 0
fi

QUANT="${QUANT:-fp16}"
NP="${NP:-4}"
THREADS="${THREADS:-2}"              # = physical cores (Step 3 measurement)
CTX_PER_SLOT="${CTX_PER_SLOT:-2048}" # each slot's KV cache holds this many tokens
PORT="${PORT:-8080}"
MODEL_FILE="qwen2.5-0.5b-instruct-${QUANT}.gguf"

if [[ ! -f "models/${MODEL_FILE}" ]]; then
  echo "models/${MODEL_FILE} not found. Run: python scripts/download_gguf.py ${QUANT}"; exit 1
fi

docker rm -f "$NAME" > /dev/null 2>&1 || true
docker run -d --name "$NAME" -p "${PORT}:8080" -v "$PWD/models:/models:ro" "$IMAGE" \
  -m "/models/${MODEL_FILE}" --host 0.0.0.0 --port 8080 \
  --parallel "$NP" --ctx-size $(( NP * CTX_PER_SLOT )) --cont-batching \
  --threads "$THREADS" --threads-batch "$THREADS" \
  --jinja --metrics > /dev/null

echo "llama.cpp starting: ${MODEL_FILE} | slots=${NP} | ctx/slot=${CTX_PER_SLOT} | threads=${THREADS} | :${PORT}"