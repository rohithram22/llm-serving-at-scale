"""Gateway: one small service in front of ANY OpenAI-compatible backend (Version A, llama.cpp, vLLM).

It adds the three things a model server shouldn't have to do itself:
  1. ADMISSION CONTROL - at most MAX_INFLIGHT requests reach the backend; a bounded queue with a
     deadline in front of it; everything else gets a fast 503 + Retry-After (load shedding).
  2. RESPONSE CACHE (Redis) - identical deterministic requests (temperature 0) are answered from
     Redis, plus COALESCING: identical requests that arrive while the first is still running wait
     for that one result instead of all hitting the backend.
  3. MICRO-BATCHING (BATCH_MODE=version_a) - collects requests for a few ms and sends them to
     Version A's batch endpoint as one generate() call. Pointless for llama.cpp / vLLM, which
     already batch continuously inside the engine.

Order matters: cache -> coalesce -> admission -> (batch) -> backend. A cache hit never queues.

Run:  uvicorn app:app --app-dir serving/gateway --host 0.0.0.0 --port 9000
Env:
  BACKEND_URL         default http://localhost:8080      BACKEND_READY_PATH  default /health
  MAX_INFLIGHT        0 = no admission control           MAX_QUEUE / QUEUE_TIMEOUT_S  (0 = off)
  REDIS_URL           e.g. redis://localhost:6379/0 (empty = no cache)   CACHE_TTL_S  default 3600
  REDIS_COOLDOWN_S    after a Redis error, skip the cache this long (circuit breaker), default 5
  COALESCE            1 (default) / 0
  BATCH_MODE          off (default) | version_a          BATCH_MAX 8 · BATCH_WINDOW_MS 50
"""
import asyncio
import hashlib
import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8080").rstrip("/")
BACKEND_READY_PATH = os.environ.get("BACKEND_READY_PATH", "/health")
BACKEND_TIMEOUT_S = float(os.environ.get("BACKEND_TIMEOUT_S", "120"))
MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", "0"))
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "0"))
QUEUE_TIMEOUT_S = float(os.environ.get("QUEUE_TIMEOUT_S", "0"))
REDIS_URL = os.environ.get("REDIS_URL", "")
REDIS_COOLDOWN_S = float(os.environ.get("REDIS_COOLDOWN_S", "5"))   # circuit breaker: skip Redis this long after an error
CACHE_TTL_S = int(os.environ.get("CACHE_TTL_S", "3600"))
COALESCE = os.environ.get("COALESCE", "1") == "1"
BATCH_MODE = os.environ.get("BATCH_MODE", "off")
BATCH_MAX = int(os.environ.get("BATCH_MAX", "8"))
BATCH_WINDOW_MS = float(os.environ.get("BATCH_WINDOW_MS", "50"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gateway")

# Fields that decide the answer. Two requests with the same values get the same cached response.
CACHE_FIELDS = ("model", "messages", "max_tokens", "temperature", "top_p", "stop", "seed")


class Stats:
    def __init__(self) -> None:
        self.started = time.time()
        self.requests = 0
        self.completed = 0                # 200s returned (any path)
        self.errors = 0                   # backend failures (502)
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_errors = 0             # Redis unreachable -> treated as a miss (fail open)
        self.coalesced = 0
        self.rejected_queue_full = 0
        self.rejected_queue_timeout = 0
        self.waiting = 0
        self.in_flight = 0
        self.admitted = 0                 # inside the gate: queued + holding a slot (counted on arrival)
        self.backend_calls = 0
        self.batches = 0
        self.batched_requests = 0


STATS = Stats()
STATE: dict = {}
INFLIGHT: dict[str, asyncio.Future] = {}      # cache key -> result future of the running request


# ---------------------------------------------------------------- helpers
def cache_key(body: dict) -> str | None:
    """Only deterministic requests can share an answer: greedy (temperature 0), not streamed."""
    if body.get("stream") or body.get("temperature", 1.0) not in (0, 0.0):
        return None
    canon = json.dumps({k: body.get(k) for k in CACHE_FIELDS}, sort_keys=True, separators=(",", ":"))
    return "chat:v1:" + hashlib.sha256(canon.encode()).hexdigest()


def overloaded(reason: str) -> tuple[int, dict, dict]:
    retry_after = max(1, math.ceil(QUEUE_TIMEOUT_S or 5))
    return 503, {"error": "overloaded", "reason": reason}, {"Retry-After": str(retry_after)}


async def acquire_slot(sem: asyncio.Semaphore) -> str | None:
    """Bounded queue + deadline. Returns None once a slot is held, or the reason for rejecting.
    The caller has ALREADY counted this request in STATS.admitted (see process())."""
    deadline = time.perf_counter() + QUEUE_TIMEOUT_S if QUEUE_TIMEOUT_S > 0 else None
    STATS.waiting += 1
    acq = asyncio.ensure_future(sem.acquire())
    reason = None
    try:
        while not acq.done():
            if deadline is not None and time.perf_counter() >= deadline:
                reason = "queue_timeout"
                break
            await asyncio.wait({acq}, timeout=0.05)
    finally:
        STATS.waiting -= 1
        if not acq.done():
            acq.cancel()
            try:
                await acq
            except asyncio.CancelledError:
                pass
    if acq.cancelled():
        return reason or "queue_timeout"
    if reason is not None:            # slot arrived just as we gave up: hand it back
        sem.release()
        return reason
    return None


def redis_usable():
    """The Redis client, or None while the circuit breaker is open (recent error) or no cache is configured.
    After the cooldown the next request simply tries again, so the cache comes back on its own."""
    r = STATE.get("redis")
    if r is None or time.monotonic() < STATE.get("redis_down_until", 0.0):
        return None
    return r


def redis_failed() -> None:
    STATS.cache_errors += 1
    if time.monotonic() >= STATE.get("redis_down_until", 0.0):
        log.warning("Redis error -> cache bypassed for %.0fs", REDIS_COOLDOWN_S)
    STATE["redis_down_until"] = time.monotonic() + REDIS_COOLDOWN_S


async def cache_get(key: str) -> str | None:
    r = redis_usable()
    if r is None:
        return None
    try:
        return await r.get(key)
    except Exception:                  # cache down must never take the service down (fail open)
        redis_failed()
        return None


async def cache_set(key: str, value: dict) -> None:
    r = redis_usable()
    if r is None:
        return
    try:
        await r.set(key, json.dumps(value), ex=CACHE_TTL_S)
    except Exception:
        redis_failed()


# ------------------------------------------------------------ backend I/O
async def call_backend(body: dict) -> tuple[int, dict, dict]:
    STATS.backend_calls += 1
    try:
        r = await STATE["client"].post(f"{BACKEND_URL}/v1/chat/completions", json=body)
        try:
            content = r.json()
        except ValueError:
            content = {"error": r.text[:300]}
        return r.status_code, content, {}
    except httpx.HTTPError as e:
        return 502, {"error": f"backend unreachable: {type(e).__name__}"}, {}


class MicroBatcher:
    """Collects greedy requests for up to BATCH_WINDOW_MS (or BATCH_MAX of them) and sends them to
    Version A's /v1/batch/chat/completions as ONE call. While a batch runs, new requests pile up
    in the queue, so the next batch forms instantly ("batch while busy")."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()
        self.task: asyncio.Task | None = None

    def start(self) -> None:
        self.task = asyncio.create_task(self._loop())

    async def submit(self, body: dict) -> tuple[int, dict, dict]:
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((body, fut))
        return await fut

    async def _loop(self) -> None:
        while True:
            items = [await self.queue.get()]
            deadline = time.perf_counter() + BATCH_WINDOW_MS / 1000
            # Poll instead of wait_for(queue.get()): cancelling a get() at the wrong moment can lose an item.
            while len(items) < BATCH_MAX:
                try:
                    items.append(self.queue.get_nowait())
                    continue
                except asyncio.QueueEmpty:
                    pass
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(0.005, remaining))
            await self._run(items)

    async def _run(self, items: list) -> None:
        STATS.backend_calls += 1
        STATS.batches += 1
        STATS.batched_requests += len(items)
        try:
            r = await STATE["client"].post(f"{BACKEND_URL}/v1/batch/chat/completions",
                                           json={"requests": [b for b, _ in items]})
            r.raise_for_status()
            responses = r.json()["responses"]
            if len(responses) != len(items):
                raise ValueError(f"backend returned {len(responses)} responses for {len(items)} requests")
            results = [(200, resp, {"X-Batch-Size": str(len(items))}) for resp in responses]
        except Exception as e:
            log.warning("batch of %d failed: %s", len(items), e)
            results = [(502, {"error": f"batch failed: {type(e).__name__}"}, {})] * len(items)
        for (_, fut), res in zip(items, results):
            if not fut.done():
                fut.set_result(res)


async def process(body: dict, key: str | None) -> tuple[int, dict, dict]:
    """Admission -> backend (direct or batched) -> cache store. Never raises."""
    sem = STATE.get("sem")
    if sem is not None:
        # Count requests AT THE DOOR, synchronously, before any await. Checking "are the slots taken?"
        # instead lets a burst that arrives in the same instant all slip in (nobody holds a slot yet).
        if MAX_QUEUE and STATS.admitted >= MAX_INFLIGHT + MAX_QUEUE:
            STATS.rejected_queue_full += 1
            return overloaded("queue_full")
        STATS.admitted += 1
        reason = await acquire_slot(sem)
        if reason is not None:
            STATS.admitted -= 1
            STATS.rejected_queue_timeout += 1
            return overloaded(reason)
    STATS.in_flight += 1
    try:
        batcher = STATE.get("batcher")
        if batcher is not None and body.get("temperature", 1.0) in (0, 0.0) and not body.get("stream"):
            status, content, headers = await batcher.submit(body)
        else:
            status, content, headers = await call_backend(body)
    except Exception as e:                       # belt and braces: process() must never raise
        log.exception("unexpected error")
        status, content, headers = 502, {"error": type(e).__name__}, {}
    finally:
        STATS.in_flight -= 1
        if sem is not None:
            sem.release()
            STATS.admitted -= 1
    if status == 200 and key:
        await cache_set(key, content)
    if status >= 500 and status != 503:
        STATS.errors += 1
    return status, content, headers


# ----------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["client"] = httpx.AsyncClient(timeout=BACKEND_TIMEOUT_S,
                                        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=200))
    STATE["sem"] = asyncio.Semaphore(MAX_INFLIGHT) if MAX_INFLIGHT > 0 else None
    if REDIS_URL:
        # Keep the client even if Redis is down right now: containers start in any order and restart.
        # redis-py reconnects by itself; the circuit breaker avoids hammering a dead Redis meanwhile.
        STATE["redis"] = aioredis.from_url(REDIS_URL, decode_responses=True, socket_timeout=0.5,
                                           socket_connect_timeout=0.5)
        try:
            await STATE["redis"].ping()
            log.info("cache: Redis at %s (TTL %ss)", REDIS_URL, CACHE_TTL_S)
        except Exception as e:
            redis_failed()
            log.warning("cache: Redis at %s not reachable yet (%s) - serving without cache until it is", REDIS_URL, e)
    if BATCH_MODE == "version_a":
        STATE["batcher"] = MicroBatcher()
        STATE["batcher"].start()
    log.info("gateway -> %s | max_inflight=%s max_queue=%s queue_timeout=%ss | cache=%s coalesce=%s | batch=%s",
             BACKEND_URL, MAX_INFLIGHT or "unlimited", MAX_QUEUE or "unlimited", QUEUE_TIMEOUT_S or "none",
             "configured" if STATE.get("redis") else "off", COALESCE,
             f"{BATCH_MODE} (max {BATCH_MAX}, window {BATCH_WINDOW_MS:.0f}ms)" if STATE.get("batcher") else "off")
    yield
    if STATE.get("batcher") and STATE["batcher"].task:
        STATE["batcher"].task.cancel()
    await STATE["client"].aclose()


app = FastAPI(title="LLM gateway", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Ready = the backend is ready. (Redis being down only disables the cache.)"""
    try:
        r = await STATE["client"].get(f"{BACKEND_URL}{BACKEND_READY_PATH}", timeout=3)
        ok = r.status_code == 200
    except httpx.HTTPError:
        ok = False
    return JSONResponse({"ready": ok, "cache": await cache_reachable()}, status_code=200 if ok else 503)


async def cache_reachable() -> bool:
    r = STATE.get("redis")
    if r is None:
        return False
    try:
        return bool(await r.ping())
    except Exception:
        return False


@app.get("/stats")
async def stats():
    lookups = STATS.cache_hits + STATS.cache_misses
    return {
        "backend": BACKEND_URL, "max_inflight": MAX_INFLIGHT, "max_queue": MAX_QUEUE,
        "queue_timeout_s": QUEUE_TIMEOUT_S, "cache": bool(STATE.get("redis")),
        "cache_bypassed_now": STATE.get("redis") is not None and redis_usable() is None, "coalesce": COALESCE,
        "batch_mode": BATCH_MODE if STATE.get("batcher") else "off",
        "requests": STATS.requests, "completed": STATS.completed, "errors": STATS.errors,
        "waiting": STATS.waiting, "in_flight": STATS.in_flight, "admitted": STATS.admitted,
        "rejected_queue_full": STATS.rejected_queue_full, "rejected_queue_timeout": STATS.rejected_queue_timeout,
        "cache_hits": STATS.cache_hits, "cache_misses": STATS.cache_misses, "cache_errors": STATS.cache_errors,
        "cache_hit_rate": round(STATS.cache_hits / lookups, 4) if lookups else None,
        "coalesced": STATS.coalesced, "backend_calls": STATS.backend_calls,
        "batches": STATS.batches, "avg_batch_size": round(STATS.batched_requests / STATS.batches, 2) if STATS.batches else None,
        "uptime_s": round(time.time() - STATS.started, 1),
    }


@app.get("/metrics")
async def metrics():
    """Pass the backend's Prometheus metrics through, so load tests via the gateway still see them."""
    try:
        r = await STATE["client"].get(f"{BACKEND_URL}/metrics", timeout=3)
        return PlainTextResponse(r.text, status_code=r.status_code)
    except httpx.HTTPError:
        return PlainTextResponse("", status_code=502)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    t0 = time.perf_counter()
    STATS.requests += 1
    try:
        body = await request.json()
        assert isinstance(body, dict)
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    key = cache_key(body)
    source = "MISS"
    if key and STATE.get("redis"):                                   # 1. cache
        cached = await cache_get(key)
        if cached is not None:
            STATS.cache_hits += 1
            STATS.completed += 1
            return JSONResponse(json.loads(cached), headers={
                "X-Cache": "HIT", "X-Gateway-Ms": f"{(time.perf_counter() - t0) * 1000:.1f}"})
        STATS.cache_misses += 1

    if key and COALESCE and key in INFLIGHT:                          # 2. coalesce
        STATS.coalesced += 1
        status, content, headers = await asyncio.shield(INFLIGHT[key])
        source = "COALESCED"
    elif key and COALESCE:                                            # leader for this key
        fut = asyncio.get_running_loop().create_future()
        INFLIGHT[key] = fut
        try:
            status, content, headers = await process(body, key)      # 3+4. admission, backend
            fut.set_result((status, content, headers))
        finally:
            INFLIGHT.pop(key, None)
            if not fut.done():
                fut.set_result((502, {"error": "gateway error"}, {}))
    else:
        status, content, headers = await process(body, key)

    if status == 200:
        STATS.completed += 1
    return JSONResponse(content, status_code=status, headers={
        **headers, "X-Cache": source, "X-Gateway-Ms": f"{(time.perf_counter() - t0) * 1000:.1f}"})
