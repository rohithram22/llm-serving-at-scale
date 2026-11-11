"""Locust load test for any OpenAI-compatible /v1/chat/completions server.

Each simulated user: pick a prompt -> send it -> wait for the answer -> think -> repeat.
Every request is written as one row to RAW_CSV so percentiles are computed from raw data.

Config (env vars):
  PROMPTS_FILE          default loadtest/prompts.jsonl
  RAW_CSV               default results/raw/requests.csv
  TIMEOUT_S             client timeout per request, default 60
  THINK_MIN / THINK_MAX think time between requests (s), default 1 / 3
  REPEAT_RATIO          fraction of requests that reuse an exact prompt (default 0 = all unique)
  MODEL_NAME            "model" field to send (default Qwen/Qwen2.5-0.5B-Instruct)
  MAX_TOKENS_OVERRIDE   if set, overrides each prompt's max_tokens
"""
import csv
import json
import os
import random
import time
import uuid

from locust import FastHttpUser, between, events, task

PROMPTS_FILE = os.environ.get("PROMPTS_FILE", "loadtest/prompts.jsonl")
RAW_CSV = os.environ.get("RAW_CSV", "results/raw/requests.csv")
TIMEOUT_S = float(os.environ.get("TIMEOUT_S", "60"))
THINK_MIN = float(os.environ.get("THINK_MIN", "1"))
THINK_MAX = float(os.environ.get("THINK_MAX", "3"))
REPEAT_RATIO = float(os.environ.get("REPEAT_RATIO", "0"))
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_TOKENS_OVERRIDE = int(os.environ.get("MAX_TOKENS_OVERRIDE", "0")) or None

PROMPTS = [json.loads(line) for line in open(PROMPTS_FILE) if line.strip()]

FIELDS = ["start_ts", "end_ts", "latency_s", "status", "error", "category", "prompt_id",
          "repeated", "max_tokens", "completion_tokens", "queue_ms", "compute_ms", "users"]
_fh = None
_writer = None


@events.test_start.add_listener
def _open_csv(environment, **_kw):
    global _fh, _writer
    os.makedirs(os.path.dirname(RAW_CSV) or ".", exist_ok=True)
    _fh = open(RAW_CSV, "w", newline="")
    _writer = csv.writer(_fh)
    _writer.writerow(FIELDS)


@events.test_stop.add_listener
def _close_csv(environment, **_kw):
    if _fh:
        _fh.flush()
        _fh.close()


def pick_prompt():
    """Return (prompt_row, text, repeated). Unique prompts get a random tag at the END,
    so the shared beginning (chat template + prompt) stays identical, like real traffic."""
    p = random.choice(PROMPTS)
    repeated = random.random() < REPEAT_RATIO
    text = p["prompt"] if repeated else f"{p['prompt']}\n\n(ref {uuid.uuid4().hex[:8]})"
    return p, text, repeated


def classify_error(resp) -> str:
    if resp.status_code and resp.status_code != 200:
        return f"http_{resp.status_code}"
    err = str(getattr(resp, "error", "") or "").lower()
    if "timed out" in err or "timeout" in err:
        return "timeout"
    if "refused" in err or "reset" in err or "connect" in err:
        return "connection"
    return "other"


class ChatUser(FastHttpUser):
    wait_time = between(THINK_MIN, THINK_MAX)
    network_timeout = TIMEOUT_S
    connection_timeout = 10.0

    @task
    def chat(self):
        p, text, repeated = pick_prompt()
        max_tokens = MAX_TOKENS_OVERRIDE or p["max_tokens"]
        body = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        start = time.time()
        status, error, tokens, queue_ms, compute_ms = 0, "", "", "", ""
        with self.client.post("/v1/chat/completions", json=body,
                              name=f"chat:{p['category']}", catch_response=True) as resp:
            status = resp.status_code or 0
            if status == 200:
                try:
                    data = resp.json()
                    tokens = data["usage"]["completion_tokens"]
                    timing = data.get("timing") or {}
                    queue_ms = timing.get("queue_ms", "")
                    compute_ms = timing.get("compute_ms", "")
                    resp.success()
                except Exception:
                    status, error = -1, "bad_response"
                    resp.failure(error)
            else:
                error = classify_error(resp)
                resp.failure(error)
        end = time.time()

        if _writer:
            _writer.writerow([round(start, 3), round(end, 3), round(end - start, 3), status, error,
                              p["category"], p["id"], int(repeated), max_tokens, tokens,
                              queue_ms, compute_ms, self.environment.runner.user_count])
            _fh.flush()