"""Start the gateway with an in-memory Redis (fakeredis) instead of a real one."""
import sys, fakeredis, uvicorn
import redis.asyncio as aioredis
server = fakeredis.FakeServer()
aioredis.from_url = lambda *a, **k: fakeredis.FakeAsyncRedis(server=server, decode_responses=True)
sys.path.insert(0, "serving/gateway")
import app as gateway
uvicorn.run(gateway.app, port=int(sys.argv[1]), log_level="warning")
