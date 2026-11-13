#!/usr/bin/env bash
# Poll a readiness endpoint until the server can take traffic.
# Usage: bash scripts/wait_ready.sh [base_url] [timeout_seconds]
#        READY_PATH=/health bash scripts/wait_ready.sh http://localhost:8080   # llama.cpp
URL="${1:-http://localhost:8000}"
TIMEOUT="${2:-300}"
READY_PATH="${READY_PATH:-/ready}"
start=$(date +%s)
echo "waiting for $URL$READY_PATH ..."
until curl -sf "$URL$READY_PATH" > /dev/null; do
  if (( $(date +%s) - start > TIMEOUT )); then
    echo "NOT ready after ${TIMEOUT}s; check the server log"
    exit 1
  fi
  sleep 2
done
echo "READY after $(( $(date +%s) - start ))s"