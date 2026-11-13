"""Download official Qwen GGUF files into models/ (gitignored).

Usage:
    python scripts/download_gguf.py fp16              # Step 5
    python scripts/download_gguf.py q8_0 q4_k_m       # Step 7
"""
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = os.environ.get("GGUF_REPO", "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
STEM = os.environ.get("GGUF_STEM", "qwen2.5-0.5b-instruct")


def main() -> None:
    quants = sys.argv[1:] or ["fp16"]
    Path("models").mkdir(exist_ok=True)
    for q in quants:
        fname = f"{STEM}-{q}.gguf"
        print(f"downloading {REPO}/{fname} ...")
        path = Path(hf_hub_download(repo_id=REPO, filename=fname, local_dir="models"))
        print(f"  -> {path} ({path.stat().st_size / 1024**2:.0f} MB)")


if __name__ == "__main__":
    main()