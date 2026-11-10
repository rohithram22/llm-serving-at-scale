"""Download model weights + tokenizer into HF_HOME (no PyTorch needed).

Usage:
    python scripts/download_model.py                 # uses $MODEL_ID
    python scripts/download_model.py some/other-model
"""
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

# Only what we need for inference: skip README images, ONNX exports, etc.
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"]


def dir_size_bytes(path: Path) -> int:
    # Snapshot files are symlinks to blobs; stat() follows them to real sizes.
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    print(f"Downloading {model_id} -> HF_HOME={os.environ.get('HF_HOME')}")
    local_path = Path(snapshot_download(repo_id=model_id, allow_patterns=ALLOW_PATTERNS))

    print(f"\nSaved to: {local_path}")
    for f in sorted(local_path.iterdir()):
        print(f"  {f.name:<32} {f.stat().st_size / 1024**2:8.1f} MB")
    print(f"Total: {dir_size_bytes(local_path) / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()