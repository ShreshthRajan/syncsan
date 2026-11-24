"""CUTLASS issue #2027 (Stream-K nondeterministic-reduction fixup) under syncsan on a rented H100 via Modal.

Buggy: CUTLASS v3.6.0 (the final split waits only for the first split). Fixed: v3.7.0 (the final split waits for the
preceding split; fix landed in the 3.7 release commit b78588d, 2025-01-18). Test program: reproduce/cutlass/sk_repro.cu, the
issue's reproducer (fp32 SM90 cooperative GEMM, Stream-K forced, reference check), built with -lineinfo.

    modal run reproduce/modal_cutlass.py [--native "..."] [--san "..."] [--kernel device_kernel]
"""
import pathlib
import subprocess
import time

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
NVBIT = "https://github.com/NVlabs/NVBit/releases/download/v1.8.2/nvbit-Linux-x86_64-1.8.2.tar.bz2"
BUILDS = {"bug": "v3.6.0", "fix": "v3.7.0"}
NVCC = "nvcc -std=c++17 -arch=sm_90a -O3 -lineinfo --expt-relaxed-constexpr -DNDEBUG"


def sh(cmd, timeout):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, executable="/bin/bash")
        rc, out = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out = 124, (e.stdout or b"").decode(errors="replace") + (e.stderr or b"").decode(errors="replace")
    return rc, f"$ {cmd}\n[exit {rc} after {time.time() - t0:.0f} s]\n{out}"


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "bzip2", "build-essential")
    .pip_install("cmake>=3.29")
    .run_commands(f"mkdir -p /opt/nvbit && curl -sL {NVBIT} | tar xj -C /opt/nvbit",
                  *[f"git clone -q --depth 1 -b {tag} https://github.com/NVIDIA/cutlass.git /opt/cutlass_{b}" for b, tag in BUILDS.items()])
    .add_local_file(ROOT / "reproduce" / "cutlass" / "sk_repro.cu", "/opt/sk_repro.cu", copy=True)
    .run_commands(*[f"{NVCC} -I/opt/cutlass_{b}/include -I/opt/cutlass_{b}/tools/util/include /opt/sk_repro.cu -o /opt/sk_{b}"
                    for b in BUILDS])
    .add_local_dir(ROOT, "/work/syncsan", ignore=["build", "results", ".git"])
)
app = modal.App("syncsan-cutlass", image=image)


@app.function(gpu="H100", cpu=8, memory=32768, timeout=45 * 60)
def run(native: list, san: str, kernel: str, builds: list, modes: list) -> dict:
    logs, summary = {}, []
    def step(name, cmd, timeout):
        rc, out = sh(cmd, timeout); logs[name] = out; summary.append(f"{name}: exit {rc}"); return rc, out
    step("gpu", "nvidia-smi --query-gpu=name,driver_version,count --format=csv,noheader; nvcc --version | tail -2; "
         + "; ".join(f"grep -n 'wait_lt\\|wait_eq' /opt/cutlass_{b}/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp" for b in BUILDS), 300)
    step("build_syncsan", "cd /work/syncsan && cmake -S . -B build -DNVBIT_ROOT=/opt/nvbit/nvbit_release_x86_64 "
         "-DNVBIT_TOOL_CUDA_ARCHITECTURES=90 && cmake --build build -j8", 900)
    T = "LD_PRELOAD=/work/syncsan/build/syncsan.so SYNCSAN_VERBOSE=1 SYNCSAN_SHARED=0" + (f" SYNCSAN_KERNEL={kernel}" if kernel else "")
    keep = lambda out: " | ".join(l.strip() for l in out.splitlines() if any(k in l for k in (
        "iterations wrong", "can_implement", "CUDA error", "syncsan: ADVERSARIAL", "distinct racing")))[:900]
    for b in builds:
        for i, args in enumerate(native):
            rc, out = step(f"{b}_native{i}", f"/opt/sk_{b} {args}", 600); summary[-1] += " | " + keep(out)
        for name, env in (("syncsan", T), ("adversarial", T + " SYNCSAN_ADVERSARIAL=1"), ("control", T + " SYNCSAN_ADVERSARIAL=control")):
            if name in modes:
                rc, out = step(f"{b}_{name}", f"env {env} /opt/sk_{b} {san}", 900); summary[-1] += " | " + keep(out)
    return {"summary": summary, "logs": logs}


@app.local_entrypoint()
def main(native: str = "--mode=nd --k=4096 --iters=50;--mode=nd --k=4096 --sms=114 --iters=50;--mode=nd --k=16384 --iters=20;"
                       "--mode=det --k=4096 --iters=20",
         san: str = "--mode=nd --k=4096 --sms=114 --iters=1", kernel: str = "device_kernel", builds: str = "bug,fix", modes: str = "syncsan,adversarial,control", tag: str = ""):
    t0 = time.time()
    res = run.remote([a for a in native.split(";") if a], san, kernel, builds.split(","), [m for m in modes.split(",") if m])
    out = ROOT / "results" / time.strftime(f"cutlass{'_' + tag if tag else ''}_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    for name, text in res["logs"].items():
        (out / f"{name}.log").write_text(text)
    (out / "SUMMARY.txt").write_text(f"native: {native}\nsyncsan: {san}\nkernel: {kernel}\n" + "\n".join(res["summary"])
                                     + f"\nwall time {time.time() - t0:.0f} s\n")
    print("\n".join(res["summary"]))
    print(f"results in {out}")
