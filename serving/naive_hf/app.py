"""Version A: naive LLM server (FastAPI + Hugging Face transformers, CPU).

One model copy. At most MAX_CONCURRENT generations run at once (default 1).
No batching. This is the baseline every other version is compared against.

Run:
    uvicorn app:app --app-dir serving/naive_hf --host 0.0.0.0 --port 8000

Env:
    MODEL_ID        HF model repo   (default Qwen/Qwen2.5-0.5B-Instruct)
    MAX_CONCURRENT  simultaneous generations (default 1)
    TORCH_THREADS   PyTorch CPU threads (default: all cores)
"""
import os

# Must be set BEFORE torch is imported: OpenMP reads it once, and it applies to
# every thread (including the worker threads generate() runs in).
TORCH_THREADS = int(os.environ.get("TORCH_THREADS", str(os.cpu_count() or 1)))
os.environ.setdefault("OMP_NUM_THREADS", str(TORCH_THREADS))

import asyncio  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

import torch  # noqa: E402
from fastapi import FastAPI, HTTPException, Response  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "1"))

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
        self.waiting = 0          # requests queued for the semaphore
        self.in_flight = 0        # requests currently generating
        self.completed = 0
        self.errors = 0
        self.total_compute_s = 0.0
        self.total_completion_tokens = 0


STATE: dict = {"ready": False, "failed": False}
STATS = Stats()


# ------------------------------------------------------------ model logic
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


def generate_blocking(messages: list[dict], max_tokens: int, temperature: float):
    """CPU-heavy and blocking: must run in a worker thread, never on the event loop."""
    # Thread settings are per-thread: a fresh worker thread does NOT inherit the
    # main thread's torch.set_num_threads(). Record what it saw, then fix it.
    seen = torch.get_num_threads()
    if "worker_threads_before_fix" not in STATE:
        STATE["worker_threads_before_fix"] = seen
        log.info("worker thread saw torch threads=%d (target %d)", seen, TORCH_THREADS)
    if seen != TORCH_THREADS:
        torch.set_num_threads(TORCH_THREADS)

    tok, model = STATE["tok"], STATE["model"]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    kwargs = {"max_new_tokens": max_tokens, "pad_token_id": tok.pad_token_id or tok.eos_token_id}
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    with torch.inference_mode():
        out = model.generate(**inputs, **kwargs)
    prompt_len = int(inputs["input_ids"].shape[-1])
    new_ids = out[0, prompt_len:]
    text = tok.decode(new_ids, skip_special_tokens=True)
    return text, prompt_len, int(new_ids.shape[-1])


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
        log.info("warm-up done in %.1fs -> READY (max_concurrent=%d)", STATE["warmup_s"], MAX_CONCURRENT)
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
        "worker_threads_before_fix": STATE.get("worker_threads_before_fix"),
        "waiting": STATS.waiting, "in_flight": STATS.in_flight,
        "completed": STATS.completed, "errors": STATS.errors,
        "uptime_s": round(time.time() - STATS.started, 1),
        "load_s": STATE.get("load_s"), "warmup_s": STATE.get("warmup_s"),
        "avg_compute_s": round(avg_compute, 3),
        "tokens_per_s_while_computing": round(tps, 2),
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, response: Response):
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail="model not ready")

    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    sem: asyncio.Semaphore = STATE["sem"]
    t_arrive = time.perf_counter()

    # ---- queue: wait for a free generation slot
    STATS.waiting += 1
    try:
        await sem.acquire()
    finally:
        STATS.waiting -= 1
    t_start = time.perf_counter()

    # ---- compute: run generate() in a worker thread
    STATS.in_flight += 1
    try:
        messages = [m.model_dump() for m in req.messages]
        text, p_tok, c_tok = await asyncio.to_thread(
            generate_blocking, messages, req.max_tokens, req.temperature
        )
    except Exception as e:
        STATS.errors += 1
        log.exception("%s failed", req_id)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        STATS.in_flight -= 1
        sem.release()

    t_end = time.perf_counter()
    queue_ms = (t_start - t_arrive) * 1000
    compute_ms = (t_end - t_start) * 1000
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