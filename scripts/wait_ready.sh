#!/usr/bin/env bash
# Poll /ready until the server can take traffic.
# Usage: bash scripts/wait_ready.sh [base_url] [timeout_seconds]
URL="${1:-http://localhost:8000}"
TIMEOUT="${2:-300}"
start=$(date +%s)
echo "waiting for $URL/ready ..."
until curl -sf "$URL/ready" > /dev/null; do
  if (( $(date +%s) - start > TIMEOUT )); then
    echo "NOT ready after ${TIMEOUT}s; check the server log"
    exit 1
  fi
  sleep 2
done
echo "READY after $(( $(date +%s) - start ))s"