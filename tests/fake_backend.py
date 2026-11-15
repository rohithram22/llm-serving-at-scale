import asyncio
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
app = FastAPI(); sem = asyncio.Semaphore(2); calls = {"single": 0, "batch": 0, "batch_sizes": []}
def reply(b): return {"choices": [{"message": {"content": "echo:" + b["messages"][-1]["content"]}}],
                      "usage": {"prompt_tokens": 5, "completion_tokens": 3}}
@app.get("/health")
async def health(): return {"status": "ok"}
@app.get("/metrics")
async def metrics(): return PlainTextResponse("llamacpp:n_busy_slots_per_decode 1.5\n")
@app.get("/calls")
async def get_calls(): return calls
@app.post("/v1/chat/completions")
async def chat(b: dict):
    calls["single"] += 1
    async with sem:
        await asyncio.sleep(0.5)
    return reply(b)
@app.post("/v1/batch/chat/completions")
async def batch(b: dict):
    calls["batch"] += 1; calls["batch_sizes"].append(len(b["requests"]))
    await asyncio.sleep(0.5)
    return {"responses": [reply(r) for r in b["requests"]]}
