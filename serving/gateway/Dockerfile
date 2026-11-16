# Gateway image: small, non-root, one process.
#   docker build -t llm-gateway serving/gateway
FROM python:3.11-slim

# No .pyc files, logs straight to stdout (docker logs / kubectl logs read stdout)
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app
# Dependencies first, code second: editing app.py then only rebuilds the last (tiny) layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

# Never run as root inside a container
RUN useradd --create-home --uid 10001 gateway
USER gateway

EXPOSE 9000
# Liveness: is the process alive? (python, because the slim image has no curl)
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9000/health', timeout=2)"

# Exec form (a JSON list): uvicorn is PID 1 and receives SIGTERM directly -> graceful shutdown.
# ONE worker on purpose: the admission counters live in this process. Scale with more containers
# (replicas), and remember the real backend limit is replicas x MAX_INFLIGHT.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "1"]
