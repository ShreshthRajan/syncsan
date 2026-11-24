"""Build syncsan and run its 37 known-answer cases on a rented GPU through Modal (https://modal.com).

    modal run reproduce/modal_cases.py              # H100
    SYNCSAN_GPU=B200 modal run reproduce/modal_cases.py
"""
import os
import pathlib
import subprocess

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
NVBIT = "https://github.com/NVlabs/NVBit/releases/download/v1.8.2/nvbit-Linux-x86_64-1.8.2.tar.bz2"
GPU = os.environ.get("SYNCSAN_GPU", "H100")
ARCH = "100" if GPU.startswith("B") else "90"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "curl", "bzip2", "build-essential")
    .pip_install("cmake>=3.29")
    .run_commands(f"mkdir -p /opt/nvbit && curl -sL {NVBIT} | tar xj -C /opt/nvbit")
    .add_local_dir(ROOT, "/work/syncsan", ignore=["build", ".git"])
)
app = modal.App("syncsan-cases", image=image)


@app.function(gpu=GPU, cpu=8, memory=32768, timeout=30 * 60)
def run(arch: str) -> str:
    p = subprocess.run(f"cd /work/syncsan && NVBIT_ROOT=/opt/nvbit/nvbit_release_x86_64 bash tests/run_tests.sh sm_{arch}",
                       shell=True, capture_output=True, text=True)
    return p.stdout + p.stderr


@app.local_entrypoint()
def main():
    out = run.remote(ARCH)
    print(out)
    print(f"{out.count('PASS ')} passed, {out.count('FAIL ')} failed")
