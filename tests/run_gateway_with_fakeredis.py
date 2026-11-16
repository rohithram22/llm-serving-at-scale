"""Start the gateway with an in-memory Redis (fakeredis) instead of a real one."""
import os, sys, fakeredis, uvicorn
import redis.asyncio as aioredis
server = fakeredis.FakeServer()
if os.environ.get("FAKE_REDIS_DOWN_FILE"):
    # test hook: Redis is "down" while this file exists (fakeredis's server.connected flag)
    import threading, time
    def watch():
        while True:
            server.connected = not os.path.exists(os.environ["FAKE_REDIS_DOWN_FILE"]); time.sleep(0.05)
    server.connected = not os.path.exists(os.environ["FAKE_REDIS_DOWN_FILE"])
    threading.Thread(target=watch, daemon=True).start()
aioredis.from_url = lambda *a, **k: fakeredis.FakeAsyncRedis(server=server, decode_responses=True)
sys.path.insert(0, "serving/gateway")
import app as gateway
uvicorn.run(gateway.app, port=int(sys.argv[1]), log_level="warning")
