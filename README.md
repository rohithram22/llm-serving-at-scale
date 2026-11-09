# LLM Serving at Scale: from 100 to 1M users

Serve a small LLM four ways, load-test each with the same harness, and turn the measurements into a capacity plan for 100, 10K and 1M users.

> **Status:** 🚧 in progress

## TL;DR results
_Filled in at the end: one table comparing p50/p95/p99 latency, throughput and error rate across versions._

## Architecture
_Diagram goes here (Step 12)._

## Versions
| Version | Stack | Hardware | Key idea |
|---|---|---|---|
| A | FastAPI + HF transformers | CPU (Codespace) | Naive baseline: one request at a time |
| B-cpu | llama.cpp server | CPU (Codespace) | Continuous batching on CPU |
| B-gpu | vLLM | T4 GPU (Kaggle) | Continuous batching + PagedAttention KV cache |
| C | GGUF Q4/Q8, AWQ | CPU + GPU | Quantization: quality vs. speed |
| + | Redis cache + micro-batching gateway | CPU | When caching and batching help |

## Environments
| Where | What ran there | Why |
|---|---|---|
| GitHub Codespaces (2/4-core) | CPU serving, Redis, Locust, Docker, kind + HPA | Reproducible; laptop has no disk to spare |
| Kaggle (T4) | vLLM, AWQ benchmarks | Free GPU |

## Model
`Qwen/Qwen2.5-0.5B-Instruct` (official AWQ and GGUF variants exist, so the comparison stays fair). Set by the `MODEL_ID` env var.

## Reproduce
1. Open this repo in a GitHub Codespace (devcontainer sets everything up).
2. `python scripts/check_env.py`
3. _Per-step commands added as the project grows._

## Load test results
_TBD_

## Capacity plan: 100 / 10K / 1M users
_TBD_

## What couldn't be measured, and how it was estimated
_TBD_

## Project log
- [x] Step 1: Repo + environment
- [ ] Step 2: Model + smoke test
- [ ] Step 3: Version A (naive FastAPI)
- [ ] Step 4: Load-test harness
- [ ] Step 5: Version B (llama.cpp, CPU)
- [ ] Step 6: Version B (vLLM, GPU)
- [ ] Step 7: Version C (quantization)
- [ ] Step 8: Cache + batching gateway
- [ ] Step 9: Docker
- [ ] Step 10: Kubernetes + HPA
- [ ] Step 11: Capacity plan
- [ ] Step 12: Charts + docs