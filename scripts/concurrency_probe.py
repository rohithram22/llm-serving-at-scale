"""Fire N identical requests at the same moment and see how they queue.

Works with Version A (reads its "timing" field) and llama.cpp (reads its "timings" field).
Usage:
    python scripts/concurrency_probe.py -n 4
    python scripts/concurrency_probe.py -n 4 --url http://localhost:8080 --max-tokens 32
"""
import argparse
import asyncio
import sys
import time

import httpx

PROMPT = "Explain what an API is to a beginner."


def server_times(body: dict, total_s: float) -> tuple[float | None, float | None]:
    """Return (queue_ms, compute_ms) from whichever timing format the server uses."""
    t = body.get("timing") or {}
    if t.get("compute_ms") is not None:                       # Version A
        return t.get("queue_ms"), t.get("compute_ms")
    tm = body.get("timings") or {}
    if tm.get("predicted_ms") is not None:                    # llama.cpp
        compute = (tm.get("prompt_ms") or 0) + tm["predicted_ms"]
        return max(total_s * 1000 - compute, 0.0), compute    # queue = time not spent computing
    return None, None


async def one(client: httpx.AsyncClient, url: str, max_tokens: int, i: int) -> dict:
    t0 = time.perf_counter()
    r = await client.post(f"{url}/v1/chat/completions", json={
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens, "temperature": 0,
    })
    total_s = time.perf_counter() - t0
    if r.status_code != 200:
        return {"i": i, "error": f"HTTP {r.status_code}: {r.text[:200]}", "total_s": total_s}
    body = r.json()
    queue_ms, compute_ms = server_times(body, total_s)
    return {"i": i, "total_s": total_s, "queue_ms": queue_ms, "compute_ms": compute_ms,
            "tokens": body["usage"]["completion_tokens"]}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=4, help="concurrent requests")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    async with httpx.AsyncClient(timeout=600) as client:
        t0 = time.perf_counter()
        try:
            results = await asyncio.gather(*[one(client, args.url, args.max_tokens, i)
                                             for i in range(args.n)])
        except httpx.ConnectError:
            sys.exit(f"Could not connect to {args.url}: is the server running?")
        wall = time.perf_counter() - t0
        try:
            stats = (await client.get(f"{args.url}/stats")).json()
        except Exception:          # llama.cpp has no /stats
            stats = {}

    errors = [r for r in results if "error" in r]
    for r in errors:
        print(f"req {r['i']} FAILED after {r['total_s']:.2f}s -> {r['error']}")
    ok = [r for r in results if "error" not in r]
    if not ok:
        sys.exit(1)

    who = f"server max_concurrent={stats['max_concurrent']}" if "max_concurrent" in stats else args.url
    print(f"\n{args.n} concurrent requests | {who}\n")
    print(f"{'req':>4} {'total_s':>8} {'queue_s':>8} {'compute_s':>10} {'tokens':>7}")
    for r in sorted(ok, key=lambda r: r["total_s"]):
        q = (r["queue_ms"] or 0) / 1000
        c = (r["compute_ms"] or 0) / 1000
        print(f"{r['i']:>4} {r['total_s']:>8.2f} {q:>8.2f} {c:>10.2f} {r['tokens']:>7}")
    tokens = sum(r["tokens"] for r in ok)
    print(f"\nwall time {wall:.2f}s | throughput {len(ok) / wall:.3f} req/s | "
          f"{tokens / wall:.1f} tokens/s | errors {len(errors)}")


if __name__ == "__main__":
    asyncio.run(main())