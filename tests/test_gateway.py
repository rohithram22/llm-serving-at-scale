import asyncio, os, subprocess, sys, time
import httpx
BK, GW = "http://localhost:8081", "http://localhost:9001"

def start(cmd, url, env=None):
    p = subprocess.Popen(cmd, env={**os.environ, **(env or {})}, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for _ in range(100):
        try:
            if httpx.get(url + "/health", timeout=1).status_code == 200: return p
        except httpx.HTTPError: pass
        time.sleep(0.1)
    raise SystemExit(p.stderr.read().decode()[-2000:])

def body(text, temp=0, mt=8):
    return {"model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": mt, "temperature": temp}

async def fire(bodies):
    async with httpx.AsyncClient(timeout=30) as c:
        async def one(b):
            t = time.perf_counter(); r = await c.post(GW + "/v1/chat/completions", json=b)
            return r.status_code, r.headers.get("x-cache"), round(time.perf_counter() - t, 2), r.json()
        return await asyncio.gather(*[one(b) for b in bodies])

def scenario(name, env, launcher="fakeredis"):
    backend = start([sys.executable, "-m", "uvicorn", "fake_backend:app", "--app-dir", "tests", "--port", "8081", "--log-level", "warning"], BK)
    cmd = ([sys.executable, "tests/run_gateway_with_fakeredis.py", "9001"] if launcher == "fakeredis" else
           [sys.executable, "-m", "uvicorn", "app:app", "--app-dir", "serving/gateway", "--port", "9001", "--log-level", "warning"])
    gw = start(cmd, GW, {"BACKEND_URL": BK, **env})
    return backend, gw

def stop(*ps):
    for p in ps: p.terminate(); p.wait()

results = []
def check(name, cond, detail=""):
    results.append(cond); print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")

# 1. cache: second identical request is a HIT; backend called once; temperature>0 never cached
b, g = scenario("cache", {"REDIS_URL": "redis://fake"})
r1 = asyncio.run(fire([body("q1")])); r2 = asyncio.run(fire([body("q1")]))
r3 = asyncio.run(fire([body("q1", temp=0.7)])); r4 = asyncio.run(fire([body("q1", temp=0.7)]))
calls = httpx.get(BK + "/calls").json()
check("cache MISS then HIT", r1[0][1] == "MISS" and r2[0][1] == "HIT", f"{r1[0][1]}->{r2[0][1]}, hit took {r2[0][2]}s")
check("cached answer is the right one", r2[0][3]["choices"][0]["message"]["content"] == "echo:q1")
check("temperature>0 is never cached", r3[0][1] == "MISS" and r4[0][1] == "MISS" and calls["single"] == 3, f"backend calls={calls['single']}")
st = httpx.get(GW + "/stats").json()
check("stats: hit rate", st["cache_hits"] == 1 and st["cache_hit_rate"] == 0.5, f"hits={st['cache_hits']} rate={st['cache_hit_rate']}")
stop(b, g)

# 2. coalescing: 10 identical concurrent requests -> ONE backend call
b, g = scenario("coalesce", {"REDIS_URL": "redis://fake"})
rs = asyncio.run(fire([body("same")] * 10))
calls = httpx.get(BK + "/calls").json()
check("10 identical concurrent -> 1 backend call", calls["single"] == 1 and all(r[0] == 200 for r in rs),
      f"backend calls={calls['single']}, sources={sorted(set(r[1] for r in rs))}")
check("all 10 got the right answer", all(r[3]["choices"][0]["message"]["content"] == "echo:same" for r in rs))
stop(b, g)

# 3. admission: 2 in flight + 2 queued; the other 6 rejected FAST with Retry-After
b, g = scenario("admission", {"MAX_INFLIGHT": "2", "MAX_QUEUE": "2", "QUEUE_TIMEOUT_S": "5"})
rs = asyncio.run(fire([body(f"d{i}") for i in range(10)]))
ok, rej = [r for r in rs if r[0] == 200], [r for r in rs if r[0] == 503]
check("admission: 4 served, 6 fast 503", len(ok) == 4 and len(rej) == 6 and max(r[2] for r in rej) < 0.3,
      f"served={len(ok)} rejected={len(rej)} slowest reject={max(r[2] for r in rej) if rej else None}s")
st = httpx.get(GW + "/stats").json()
check("counters back to zero after a burst", st["admitted"] == 0 and st["in_flight"] == 0, f"admitted={st['admitted']}")
stop(b, g)

# 4. queue deadline: 1 slot, deadline 0.3s -> waiting requests give up quickly
b, g = scenario("deadline", {"MAX_INFLIGHT": "1", "QUEUE_TIMEOUT_S": "0.3"})
rs = asyncio.run(fire([body(f"t{i}") for i in range(3)]))
to = [r for r in rs if r[0] == 503 and r[3].get("reason") == "queue_timeout"]
check("queue deadline: 1 served, 2 timed out ~0.3s", len(to) == 2 and all(0.25 <= r[2] <= 0.8 for r in to),
      f"timeouts at {[r[2] for r in to]}s")
st = httpx.get(GW + "/stats").json()
check("no leaked slots after timeouts", st["in_flight"] == 0 and st["waiting"] == 0 and st["admitted"] == 0)
rs = asyncio.run(fire([body("after")])); check("still serves after rejections", rs[0][0] == 200)
stop(b, g)

# 5. a cache HIT bypasses a full queue
b, g = scenario("hit-bypasses", {"REDIS_URL": "redis://fake", "MAX_INFLIGHT": "1", "MAX_QUEUE": "1", "QUEUE_TIMEOUT_S": "5"})
asyncio.run(fire([body("popular")]))                      # warm the cache
async def busy_then_hit():
    async with httpx.AsyncClient(timeout=30) as c:
        blockers = [asyncio.create_task(c.post(GW + "/v1/chat/completions", json=body(f"x{i}"))) for i in range(3)]
        await asyncio.sleep(0.1)
        t = time.perf_counter(); r = await c.post(GW + "/v1/chat/completions", json=body("popular"))
        took = time.perf_counter() - t; await asyncio.gather(*blockers); return r.status_code, r.headers["x-cache"], took
s, src, took = asyncio.run(busy_then_hit())
check("cache HIT served instantly while backend is saturated", s == 200 and src == "HIT" and took < 0.2, f"{src} in {took:.3f}s")
stop(b, g)

# 6. micro-batching: 8 distinct concurrent requests -> few batch calls, every answer matches ITS request
b, g = scenario("batch", {"BATCH_MODE": "version_a", "BATCH_MAX": "8", "BATCH_WINDOW_MS": "50"})
rs = asyncio.run(fire([body(f"b{i}") for i in range(8)] + [body("not-greedy", temp=0.5)]))
calls = httpx.get(BK + "/calls").json()
greedy = rs[:8]
check("8 greedy requests -> batched", calls["batch"] <= 2 and sum(calls["batch_sizes"]) == 8, f"batch sizes={calls['batch_sizes']}")
check("each batched response matches its own request", all(r[3]["choices"][0]["message"]["content"] == f"echo:b{i}" for i, r in enumerate(greedy)))
check("temperature>0 bypasses the batcher", calls["single"] == 1)
stop(b, g)

# 7. Redis down -> cache disabled, gateway still works (fail open)
b, g = scenario("redis-down", {"REDIS_URL": "redis://localhost:6399/0"}, launcher="plain")
rs = asyncio.run(fire([body("r1"), body("r1")]))
rd = httpx.get(GW + "/ready").json()
check("Redis down: requests still served, cache off", all(r[0] == 200 for r in rs) and rd == {"ready": True, "cache": False}, f"ready={rd}")
stop(b, g)

# 8. backend down -> 502, not a crash; /ready says not ready
g = start([sys.executable, "-m", "uvicorn", "app:app", "--app-dir", "serving/gateway", "--port", "9001", "--log-level", "warning"], GW,
          {"BACKEND_URL": "http://localhost:8099", "MAX_INFLIGHT": "2"})
rs = asyncio.run(fire([body("x")])); rd = httpx.get(GW + "/ready")
st = httpx.get(GW + "/stats").json()
check("backend down: 502 + not ready + slot released", rs[0][0] == 502 and rd.status_code == 503 and st["in_flight"] == 0)
stop(g)

print(f"\n{sum(results)}/{len(results)} gateway checks passed")
