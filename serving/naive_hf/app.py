"""Version A: naive LLM server (FastAPI + Hugging Face transformers, CPU).

One model copy. At most MAX_CONCURRENT generations run at once (default 1).
No batching. This is the baseline every other version is compared against.

Run:
    uvicorn app:app --app-dir serving/naive_hf --host 0.0.0.0 --port 8000

Env:
    MODEL_ID              HF model repo (default Qwen/Qwen2.5-0.5B-Instruct)
    MAX_CONCURRENT        simultaneous generations (default 1)
    TORCH_THREADS         PyTorch CPU threads (default: all cores)

Admission control (Step 4b). All OFF by default, so the baseline stays reproducible:
    MAX_QUEUE             max requests waiting for a slot; more -> 503 immediately (0 = unlimited)
    QUEUE_TIMEOUT_S       max seconds a request may wait for a slot -> 503 (0 = wait forever)
    CANCEL_ON_DISCONNECT  1 = stop work for clients that hung up (while queued or mid-generation)
"""
import os

# Must be set BEFORE torch is imported: OpenMP reads it once, for every thread.
TORCH_THREADS = int(os.environ.get("TORCH_THREADS", str(os.cpu_count() or 1)))
os.environ.setdefault("OMP_NUM_THREADS", str(TORCH_THREADS))

import asyncio  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

import torch  # noqa: E402
from fastapi import FastAPI, HTTPException, Request, Response  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer,  # noqa: E402
                          StoppingCriteria, StoppingCriteriaList)

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "1"))
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "0"))
QUEUE_TIMEOUT_S = float(os.environ.get("QUEUE_TIMEOUT_S", "0"))
CANCEL_ON_DISCONNECT = os.environ.get("CANCEL_ON_DISCONNECT", "0") == "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("version_a")
torch.set_num_threads(TORCH_THREADS)


# ----------------------------------------------------------------- schemas
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None  # accepted for OpenAI compatibility, ignored
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


# ------------------------------------------------------------------- state
class Stats:
    def __init__(self) -> None:
        self.started = time.time()
        self.waiting = 0                   # requests queued for a slot
        self.in_flight = 0                 # requests currently generating
        self.completed = 0
        self.errors = 0
        self.rejected_queue_full = 0       # 503: queue already at MAX_QUEUE
        self.rejected_queue_timeout = 0    # 503: waited longer than QUEUE_TIMEOUT_S
        self.cancelled_while_queued = 0    # client hung up before we started
        self.cancelled_mid_generation = 0  # client hung up while we were generating
        self.total_compute_s = 0.0
        self.total_completion_tokens = 0


STATE: dict = {"ready": False, "failed": False}
STATS = Stats()


# ------------------------------------------------------------ model logic
class StopWhenSet(StoppingCriteria):
    """Ends generate() early once the event is set (client disconnected)."""

    def __init__(self, event: threading.Event) -> None:
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.event.is_set(), dtype=torch.bool)


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()
    # The model ships sampling defaults (temperature/top_p/top_k). Clear them so
    # greedy requests don't emit warnings; we set them per request when sampling.
    gc = model.generation_config
    gc.do_sample = False
    gc.temperature = None
    gc.top_p = None
    gc.top_k = None
    return tok, model


def generate_blocking(messages: list[dict], max_tokens: int, temperature: float,
                      cancel: threading.Event | None = None):
    """CPU-heavy and blocking: must run in a worker thread, never on the event loop.
    Returns (text, prompt_tokens, completion_tokens, was_cancelled)."""
    seen = torch.get_num_threads()
    if "worker_threads_before_fix" not in STATE:
        STATE["worker_threads_before_fix"] = seen
    if seen != TORCH_THREADS:
        torch.set_num_threads(TORCH_THREADS)

    tok, model = STATE["tok"], STATE["model"]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    kwargs = {"max_new_tokens": max_tokens, "pad_token_id": tok.pad_token_id or tok.eos_token_id}
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    if cancel is not None:
        kwargs["stopping_criteria"] = StoppingCriteriaList([StopWhenSet(cancel)])
    with torch.inference_mode():
        out = model.generate(**inputs, **kwargs)
    prompt_len = int(inputs["input_ids"].shape[-1])
    new_ids = out[0, prompt_len:]
    text = tok.decode(new_ids, skip_special_tokens=True)
    return text, prompt_len, int(new_ids.shape[-1]), bool(cancel is not None and cancel.is_set())


async def startup() -> None:
    """Load + warm up in the background so /health answers immediately."""
    try:
        t0 = time.perf_counter()
        STATE["tok"], STATE["model"] = await asyncio.to_thread(load_model)
        STATE["load_s"] = round(time.perf_counter() - t0, 2)
        log.info("model %s loaded in %.1fs (torch threads=%d)", MODEL_ID, STATE["load_s"], TORCH_THREADS)

        t1 = time.perf_counter()
        await asyncio.to_thread(generate_blocking, [{"role": "user", "content": "Hello"}], 8, 0.0)
        STATE["warmup_s"] = round(time.perf_counter() - t1, 2)
        log.info("warm-up done in %.1fs -> READY (max_concurrent=%d, max_queue=%s, "
                 "queue_timeout=%s, cancel_on_disconnect=%s)", STATE["warmup_s"], MAX_CONCURRENT,
                 MAX_QUEUE or "unlimited", f"{QUEUE_TIMEOUT_S}s" if QUEUE_TIMEOUT_S else "none",
                 CANCEL_ON_DISCONNECT)
        STATE["ready"] = True
    except Exception:
        STATE["failed"] = True
        log.exception("startup failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["sem"] = asyncio.Semaphore(MAX_CONCURRENT)
    task = asyncio.create_task(startup())
    yield
    task.cancel()


app = FastAPI(title="Version A: naive HF server", lifespan=lifespan)


# ------------------------------------------------------ admission control
def overloaded(reason: str) -> JSONResponse:
    """Fast rejection. Retry-After = rough guess of when a slot frees up."""
    avg = STATS.total_compute_s / STATS.completed if STATS.completed else 5.0
    retry_after = max(1, math.ceil(avg * (STATS.waiting + 1) / MAX_CONCURRENT))
    return JSONResponse(status_code=503, content={"error": "overloaded", "reason": reason},
                        headers={"Retry-After": str(retry_after)})


async def watch_disconnect(request: Request, cancel: threading.Event) -> None:
    """Poll the connection; set `cancel` as soon as the client hangs up."""
    try:
        while not cancel.is_set():
            if await request.is_disconnected():
                cancel.set()
                return
            await asyncio.sleep(0.25)
    except asyncio.CancelledError:
        pass


async def acquire_slot(sem: asyncio.Semaphore, cancel: threading.Event) -> str | None:
    """Wait for a generation slot. Returns None once acquired, or the reason we gave up:
    'timeout' (waited > QUEUE_TIMEOUT_S) or 'disconnected' (client left while queued)."""
    deadline = time.perf_counter() + QUEUE_TIMEOUT_S if QUEUE_TIMEOUT_S > 0 else None
    acq = asyncio.ensure_future(sem.acquire())
    reason = None
    try:
        while not acq.done():
            if deadline is not None and time.perf_counter() >= deadline:
                reason = "timeout"
                break
            if cancel.is_set():
                reason = "disconnected"
                break
            await asyncio.wait({acq}, timeout=0.25)
    finally:
        if not acq.done():
            acq.cancel()
            try:
                await acq
            except asyncio.CancelledError:
                pass
    if acq.cancelled():
        return reason or "cancelled"
    if reason is not None:      # the slot arrived at the same moment we gave up: hand it back
        sem.release()
        return reason
    return None


# --------------------------------------------------------------- endpoints
@app.get("/health")
async def health():
    """Liveness: the process is up (even if the model is still loading)."""
    return {"status": "failed" if STATE["failed"] else "ok"}


@app.get("/ready")
async def ready(response: Response):
    """Readiness: the model is loaded and warmed up; safe to send traffic."""
    if not STATE["ready"]:
        response.status_code = 503
    return {"ready": STATE["ready"]}


@app.get("/stats")
async def stats():
    avg_compute = STATS.total_compute_s / STATS.completed if STATS.completed else 0.0
    tps = STATS.total_completion_tokens / STATS.total_compute_s if STATS.total_compute_s else 0.0
    return {
        "ready": STATE["ready"], "model": MODEL_ID,
        "max_concurrent": MAX_CONCURRENT, "torch_threads": TORCH_THREADS,
        "max_queue": MAX_QUEUE, "queue_timeout_s": QUEUE_TIMEOUT_S,
        "cancel_on_disconnect": CANCEL_ON_DISCONNECT,
        "waiting": STATS.waiting, "in_flight": STATS.in_flight,
        "completed": STATS.completed, "errors": STATS.errors,
        "rejected_queue_full": STATS.rejected_queue_full,
        "rejected_queue_timeout": STATS.rejected_queue_timeout,
        "cancelled_while_queued": STATS.cancelled_while_queued,
        "cancelled_mid_generation": STATS.cancelled_mid_generation,
        "uptime_s": round(time.time() - STATS.started, 1),
        "load_s": STATE.get("load_s"), "warmup_s": STATE.get("warmup_s"),
        "avg_compute_s": round(avg_compute, 3),
        "tokens_per_s_while_computing": round(tps, 2),
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request, response: Response):
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail="model not ready")

    sem: asyncio.Semaphore = STATE["sem"]

    # ---- admission 1: bounded queue. If we'd have to wait AND the line is full, reject now.
    if MAX_QUEUE and sem.locked() and STATS.waiting >= MAX_QUEUE:
        STATS.rejected_queue_full += 1
        return overloaded("queue_full")

    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    t_arrive = time.perf_counter()
    cancel = threading.Event()
    watcher = asyncio.create_task(watch_disconnect(request, cancel)) if CANCEL_ON_DISCONNECT else None
    try:
        # ---- queue: wait for a slot (admission 2: deadline; admission 3: client still there?)
        STATS.waiting += 1
        try:
            reason = await acquire_slot(sem, cancel)
        finally:
            STATS.waiting -= 1
        if reason == "timeout":
            STATS.rejected_queue_timeout += 1
            return overloaded("queue_timeout")
        if reason is None and CANCEL_ON_DISCONNECT and await request.is_disconnected():
            # Got a slot, but the client left in the gap since the watcher last looked: hand it back.
            sem.release()
            reason = "disconnected"
        if reason is not None:
            STATS.cancelled_while_queued += 1
            log.info("%s client left while queued (%.1fs)", req_id, time.perf_counter() - t_arrive)
            return JSONResponse(status_code=499, content={"error": "client closed request"})
        t_start = time.perf_counter()

        # ---- compute: generate() in a worker thread; it stops early if `cancel` gets set
        STATS.in_flight += 1
        try:
            messages = [m.model_dump() for m in req.messages]
            text, p_tok, c_tok, was_cancelled = await asyncio.to_thread(
                generate_blocking, messages, req.max_tokens, req.temperature,
                cancel if CANCEL_ON_DISCONNECT else None,
            )
        except Exception as e:
            STATS.errors += 1
            log.exception("%s failed", req_id)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            STATS.in_flight -= 1
            sem.release()
    finally:
        if watcher is not None:
            watcher.cancel()

    t_end = time.perf_counter()
    queue_ms = (t_start - t_arrive) * 1000
    compute_ms = (t_end - t_start) * 1000
    if was_cancelled:
        STATS.cancelled_mid_generation += 1
        log.info("%s client left mid-generation; stopped after %d tokens (%.0fms)",
                 req_id, c_tok, compute_ms)
        return JSONResponse(status_code=499, content={"error": "client closed request"})

    STATS.completed += 1
    STATS.total_compute_s += compute_ms / 1000
    STATS.total_completion_tokens += c_tok
    log.info("%s queue=%.0fms compute=%.0fms prompt=%d completion=%d",
             req_id, queue_ms, compute_ms, p_tok, c_tok)
    response.headers["X-Queue-Ms"] = f"{queue_ms:.0f}"
    response.headers["X-Compute-Ms"] = f"{compute_ms:.0f}"

    return {
        "id": req_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "length" if c_tok >= req.max_tokens else "stop",
        }],
        "usage": {"prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": p_tok + c_tok},
        "timing": {
            "queue_ms": round(queue_ms, 1),
            "compute_ms": round(compute_ms, 1),
            "tokens_per_s": round(c_tok / (compute_ms / 1000), 2) if compute_ms else None,
        },
    }