"""Lucebox's Qwen3.5-0.8B decode megakernel (github.com/Luce-Org/lucebox, optimizations/megakernel, BF16 sm_90 build)
under syncsan on a rented H100 via Modal: natively (twice, plus a Hugging Face greedy reference), under syncsan, and in
adversarial mode (+ control). The program is reproduce/luce/drive.py (decode path only, greedy).

    modal run reproduce/modal/luce.py [--ntok 8] [--kernel decode_kernel] [--fixed acquire|reset|both]
"""
import pathlib
import subprocess
import time

import modal

ROOT = pathlib.Path(__file__).resolve().parents[2]
NVBIT = "https://github.com/NVlabs/NVBit/releases/download/v1.8.2/nvbit-Linux-x86_64-1.8.2.tar.bz2"
LUCE_COMMIT = "a70ab3c"
MK = "/opt/luce/optimizations/megakernel"


def sh(cmd, timeout):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, executable="/bin/bash")
        rc, out = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out = 124, (e.stdout or b"").decode(errors="replace") + (e.stderr or b"").decode(errors="replace")
    return rc, f"$ {cmd}\n[exit {rc} after {time.time() - t0:.0f} s]\n{out}"


BUILD = (f"cd {MK} && CC=gcc CXX=g++ MEGAKERNEL_CUDA_ARCH=sm_90 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=8 "
         "pip install -v -e . --no-build-isolation --no-deps > /tmp/build.log 2>&1; grep -iE ' error|error:|fatal' /tmp/build.log | head -30; python -c 'import torch, qwen35_megakernel_bf16_C'")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "bzip2", "build-essential")
    .pip_install("cmake>=3.29", "setuptools>=68", "wheel", "ninja")
    .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==5.9.0", "accelerate", "safetensors", "numpy")
    .run_commands(f"mkdir -p /opt/nvbit && curl -sL {NVBIT} | tar xj -C /opt/nvbit",
                  f"git clone -q https://github.com/Luce-Org/lucebox.git /opt/luce && cd /opt/luce && git checkout -q {LUCE_COMMIT}",
                  # line info for syncsan's source chains (no change to the generated code)
                  f"cd {MK} && sed -i 's/\"--use_fast_math\",/\"--use_fast_math\", \"-lineinfo\",/' setup.py && grep -n lineinfo setup.py",
                  "python -c \"from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.5-0.8B')\"",
                  BUILD)
    .add_local_dir(ROOT, "/work/syncsan", ignore=["build", "results", ".git"])
)
app = modal.App("syncsan-luce", image=image)


@app.function(gpu="H100", cpu=8, memory=32768, timeout=45 * 60)
def run(mode: str, ntok: int, kernel: str, fixed: str, extra_env: str, ref: bytes) -> dict:
    """One mode per call (each under the 45-minute limit); the native call returns its outputs as the reference."""
    logs, summary = {}, []
    def step(name, cmd, timeout):
        rc, out = sh(cmd, timeout); logs[name] = out; summary.append(f"{name}: exit {rc}")
        print(f"=== {name}: exit {rc}\n" + out[-6000:], flush=True); return rc, out   # streamed: survives a timeout
    if mode == "native":
        step("gpu", "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader", 60)
    if mode not in ("native", "stress") or fixed:
        step("build_syncsan", "cd /work/syncsan && cmake -S src -B build -DNVBIT_ROOT=/opt/nvbit/nvbit_release_x86_64 "
             "-DNVBIT_TOOL_CUDA_ARCHITECTURES=90 && cmake --build build -j8 2>&1 | tail -3", 900)
    if fixed:
        step("patch", f"python /work/syncsan/reproduce/luce/patch_sync.py {MK}/kernel.cu {fixed} && {BUILD}", 900)
    if ref:
        pathlib.Path("/tmp/ref.pt").write_bytes(ref)
    D = f"python /work/syncsan/reproduce/luce/drive.py {ntok}"
    T = "LD_PRELOAD=/work/syncsan/build/syncsan.so SYNCSAN_VERBOSE=1 SYNCSAN_SHARED=0" + (f" SYNCSAN_KERNEL={kernel}" if kernel else "") \
        + (f" {extra_env}" if extra_env else "")
    keep = lambda out: " | ".join(l for l in out.splitlines() if l.startswith(("DRIVE", "STRESS", "syncsan: WARNING", "syncsan: ADVERSARIAL"))
                                  or ("distinct racing" in l) or "Error" in l)[:900]
    plan = {"native": [("native", f"cd /tmp && {D} /tmp/ref.pt --hf"), ("native2", f"cd /tmp && {D} /tmp/n2.pt /tmp/ref.pt")],
            "syncsan": [("syncsan", f"cd /tmp && env {T} {D} /tmp/s.pt /tmp/ref.pt")],
            "adversarial": [("adversarial", f"cd /tmp && env {T} SYNCSAN_ADVERSARIAL=1 {D} /tmp/a.pt /tmp/ref.pt")],
            "control": [("control", f"cd /tmp && env {T} SYNCSAN_ADVERSARIAL=control {D} /tmp/c.pt /tmp/ref.pt")],
            "stress": [("stress", f"cd /tmp && python /work/syncsan/reproduce/luce/stress.py {ntok} 2000")]}
    for name, cmd in plan[mode]:
        rc, out = step(name, cmd, 600 if name.startswith("native") else 2400)
        summary[-1] += " | " + keep(out)
    r = pathlib.Path("/tmp/ref.pt")
    return {"summary": summary, "logs": logs, "ref": r.read_bytes() if mode == "native" and r.exists() else b""}


@app.local_entrypoint()
def main(ntok: int = 8, kernel: str = "decode_kernel", fixed: str = "", modes: str = "syncsan,adversarial,control", extra_env: str = "SYNCSAN_SKIP_OPS==LDG.E.NA.128",
         tag: str = ""):
    t0 = time.time()
    out = ROOT / "results" / "luce" / time.strftime(f"luce{'_' + fixed if fixed else ''}{'_' + tag if tag else ''}_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    def save(res):
        for name, text in res["logs"].items():
            (out / f"{name}.log").write_text(text)
        summary.extend(res["summary"])
        (out / "SUMMARY.txt").write_text("\n".join(summary) + f"\nwall time {time.time() - t0:.0f} s\n")
    nat = run.remote("native", ntok, kernel, fixed, extra_env, b"")   # native reference (patched build when --fixed)
    save(nat)
    calls = [run.spawn(m, ntok, kernel, fixed, extra_env, nat["ref"]) for m in modes.split(",") if m]   # in parallel
    for c in calls:
        try:
            save(c.get())
        except Exception as e:   # a timed-out mode keeps the others; its streamed output is in the modal log
            summary.append(f"mode failed: {type(e).__name__}: {e}"[:300])
    (out / "SUMMARY.txt").write_text("\n".join(summary) + f"\nwall time {time.time() - t0:.0f} s\n")
    print("\n".join(summary))
