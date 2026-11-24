"""megakernel-zoo (github.com/dangxingyu/megakernel-zoo) H100 engines under syncsan on a rented H100 via Modal. Default:
the Qwen3-8B offline-a engine (track A, the checkpoint's BF16 weights; Qwen/Qwen3-8B is ungated). The program is
reproduce/zoo/drive.py: greedy decode through Engine.step(), logits compared bitwise with the native run, plus a Hugging
Face BF16 greedy reference. Modes: native (twice), syncsan, adversarial, control.

    modal run reproduce/modal_zoo.py [--ntok 4] [--kernel k_step] [--fixed acquire]
"""
import pathlib
import subprocess
import time

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
NVBIT = "https://github.com/NVlabs/NVBit/releases/download/v1.8.2/nvbit-Linux-x86_64-1.8.2.tar.bz2"
ZOO_COMMIT = "95e315c"
ENGINE = "/opt/zoo/engines/qwen3-8b_h100/offline-a/engine"
MODEL_ID, MODEL_DIR = "Qwen/Qwen3-8B", "/models/qwen3-8b"


def sh(cmd, timeout):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, executable="/bin/bash")
        rc, out = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out = 124, (e.stdout or b"").decode(errors="replace") + (e.stderr or b"").decode(errors="replace")
    return rc, f"$ {cmd}\n[exit {rc} after {time.time() - t0:.0f} s]\n{out}"


BUILD = f"cd {ENGINE} && rm -f libmk.so && MK_DEFS=-lineinfo bash build.sh 2>&1 | grep -iE 'error|warning: R' | head; ls -la libmk.so"
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "bzip2", "build-essential")
    .pip_install("cmake>=3.29")
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==4.57.1", "accelerate", "safetensors", "numpy", "huggingface_hub")
    .run_commands(f"mkdir -p /opt/nvbit && curl -sL {NVBIT} | tar xj -C /opt/nvbit",
                  f"git clone -q https://github.com/dangxingyu/megakernel-zoo.git /opt/zoo && cd /opt/zoo && git checkout -q {ZOO_COMMIT}",
                  f"python -c \"from huggingface_hub import snapshot_download as s; s('{MODEL_ID}', local_dir='{MODEL_DIR}', "
                  "allow_patterns=['*.json', '*.safetensors', '*.txt', 'merges.txt', 'vocab.json'])\"",
                  BUILD)
    .add_local_dir(ROOT, "/work/syncsan", ignore=["build", "results", ".git"])
)
app = modal.App("syncsan-zoo", image=image)


@app.function(gpu="H100", cpu=8, memory=65536, timeout=45 * 60)
def run(mode: str, ntok: int, kernel: str, fixed: str, extra_env: str, ref: bytes) -> dict:
    """One mode per call (each under the 45-minute limit); the native call returns its outputs as the reference."""
    logs, summary = {}, []
    def step(name, cmd, timeout):
        rc, out = sh(cmd, timeout); logs[name] = out; summary.append(f"{name}: exit {rc}")
        print(f"=== {name}: exit {rc}\n" + out[-6000:], flush=True); return rc, out   # streamed: survives a timeout
    if mode == "native":
        step("gpu", "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader", 60)
    if mode != "native" or fixed:
        step("build_syncsan", "cd /work/syncsan && cmake -S . -B build -DNVBIT_ROOT=/opt/nvbit/nvbit_release_x86_64 "
             "-DNVBIT_TOOL_CUDA_ARCHITECTURES=90 && cmake --build build -j8 2>&1 | tail -3", 900)
    if fixed:
        step("patch", f"python /work/syncsan/reproduce/zoo/patch_sync.py {ENGINE}/mk.cu {fixed} && {BUILD}", 900)
    if ref:
        pathlib.Path("/tmp/ref.pt").write_bytes(ref)
    D = f"python /work/syncsan/reproduce/zoo/drive.py {ENGINE} {MODEL_DIR} {ntok}"
    T = "LD_PRELOAD=/work/syncsan/build/syncsan.so SYNCSAN_VERBOSE=1 SYNCSAN_SHARED=0" + (f" SYNCSAN_KERNEL={kernel}" if kernel else "") \
        + (f" {extra_env}" if extra_env else "")
    keep = lambda out: " | ".join(l for l in out.splitlines() if l.startswith(("DRIVE", "syncsan: WARNING", "syncsan: ADVERSARIAL"))
                                  or ("distinct racing" in l) or "Error" in l)[:900]
    plan = {"native": [("native", f"cd /tmp && env DRIVE_HF=1 {D} /tmp/ref.pt"), ("native2", f"cd /tmp && {D} /tmp/n2.pt /tmp/ref.pt")],
            "syncsan": [("syncsan", f"cd /tmp && env {T} {D} /tmp/s.pt /tmp/ref.pt")],
            "adversarial": [("adversarial", f"cd /tmp && env {T} SYNCSAN_ADVERSARIAL=1 {D} /tmp/a.pt /tmp/ref.pt")],
            "control": [("control", f"cd /tmp && env {T} SYNCSAN_ADVERSARIAL=control {D} /tmp/c.pt /tmp/ref.pt")]}
    for name, cmd in plan[mode]:
        rc, out = step(name, cmd, 600 if name.startswith("native") else 2400)
        summary[-1] += " | " + keep(out)
    r = pathlib.Path("/tmp/ref.pt")
    return {"summary": summary, "logs": logs, "ref": r.read_bytes() if mode == "native" and r.exists() else b""}


@app.local_entrypoint()
def main(ntok: int = 4, kernel: str = "k_step", fixed: str = "", modes: str = "syncsan,adversarial,control", extra_env: str = "SYNCSAN_SKIP_OPS==LDG.E.LU.128",
         tag: str = ""):
    t0 = time.time()
    out = ROOT / "results" / time.strftime(f"zoo{'_' + fixed if fixed else ''}{'_' + tag if tag else ''}_%Y%m%d_%H%M%S")
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
