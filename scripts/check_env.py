"""Environment report: confirms this machine can run the project.

Usage: python scripts/check_env.py
Exit code 0 = all good, 1 = something is missing.
"""
import os
import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

REQUIRED_PACKAGES = [
    "torch", "transformers", "huggingface_hub", "fastapi", "uvicorn",
    "httpx", "redis", "locust", "pandas", "matplotlib", "datasets", "psutil",
]


def gb(n_bytes: float) -> str:
    return f"{n_bytes / 1024**3:.1f} GB"


def pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def docker_status() -> tuple[bool, str]:
    try:
        out = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError:
        return False, "docker CLI not installed"
    except subprocess.TimeoutExpired:
        return False, "docker daemon timed out"
    if out.returncode != 0:
        return False, f"daemon not reachable: {out.stderr.strip()[:150]}"
    return True, f"server {out.stdout.strip()}"


def main() -> int:
    ok = True
    print("=" * 60)
    print("ENVIRONMENT REPORT")
    print("=" * 60)

    # Python
    py_ok = sys.version_info >= (3, 10)
    ok &= py_ok
    print(f"[{'OK' if py_ok else 'FAIL'}] Python {platform.python_version()} ({platform.machine()})")

    # CPU / RAM
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"[OK] CPUs: {os.cpu_count()}  |  RAM total {gb(ram.total)}, available {gb(ram.available)}")
    except ImportError:
        print(f"[WARN] CPUs: {os.cpu_count()}  |  RAM unknown (psutil missing)")

    # Disk
    free = shutil.disk_usage(".").free
    disk_ok = free > 5 * 1024**3
    print(f"[{'OK' if disk_ok else 'WARN'}] Free disk here: {gb(free)} (want > 5 GB)")
    print(f"     HF_HOME = {os.environ.get('HF_HOME', '(not set)')}")
    print(f"     MODEL_ID = {os.environ.get('MODEL_ID', '(not set)')}")

    # Packages
    print("-" * 60)
    for pkg in REQUIRED_PACKAGES:
        v = pkg_version(pkg)
        ok &= v is not None
        print(f"[{'OK' if v else 'MISSING'}] {pkg:<16} {v or ''}")

    # Torch details
    try:
        import torch
        cuda_note = "CUDA available" if torch.cuda.is_available() else "CPU-only (expected here)"
        print(f"     torch build: {torch.__version__} -> {cuda_note}, threads={torch.get_num_threads()}")
    except ImportError:
        pass

    # Docker
    print("-" * 60)
    d_ok, d_msg = docker_status()
    ok &= d_ok
    print(f"[{'OK' if d_ok else 'FAIL'}] Docker: {d_msg}")

    print("=" * 60)
    print("ALL GOOD" if ok else "SOMETHING IS MISSING - see FAIL/MISSING lines above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())