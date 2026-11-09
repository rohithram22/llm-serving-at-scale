#!/usr/bin/env bash
# Runs once when the Codespace container is created.
set -euo pipefail

echo ">>> Upgrading pip"
python -m pip install --upgrade pip

# CPU-only torch FIRST, so transformers doesn't pull the multi-GB CUDA build.
echo ">>> Installing CPU-only PyTorch"
pip install --index-url https://download.pytorch.org/whl/cpu torch

echo ">>> Installing project requirements"
pip install -r requirements.txt

mkdir -p "${HF_HOME:-.cache/huggingface}"

echo ">>> Done. Run: python scripts/check_env.py"