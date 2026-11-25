"""AutoMegaKernel under syncsan on a rented H100 through Modal (https://modal.com).

Runs the toy model with --verify (the megakernel's tokens against an eager PyTorch run): natively, in adversarial mode,
under the control, and in adversarial mode after making the counter wait an acquire.

    modal run reproduce/modal/amk.py
"""
import pathlib
import subprocess

import modal

ROOT = pathlib.Path(__file__).resolve().parents[2]
NVBIT = "https://github.com/NVlabs/NVBit/releases/download/v1.8.2/nvbit-Linux-x86_64-1.8.2.tar.bz2"
AMK_COMMIT = "884534c"
FIX = r"""
import pathlib
p = pathlib.Path("vm/sync.cuh")
s = p.read_text()
t = s.replace("    return atomicAdd(c, 0u);",
              '    uint32_t v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(c) : "memory"); return v;')
assert t != s, "amk_acquire_load not found"
p.write_text(t)
"""

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "bzip2", "build-essential")
    .pip_install("cmake>=3.29", "uv")
    .run_commands(f"mkdir -p /opt/nvbit && curl -sL {NVBIT} | tar xj -C /opt/nvbit",
                  "git clone -q https://github.com/RightNow-AI/AutoMegaKernel.git /opt/amk && cd /opt/amk && "
                  f"git checkout -q {AMK_COMMIT} && uv sync -q")
    .add_local_dir(ROOT, "/work/syncsan", ignore=["build", ".git"])
)
app = modal.App("syncsan-amk", image=image)


def sh(cmd: str, timeout: int) -> str:
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, executable="/bin/bash")
        return p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return "timed out"


@app.function(gpu="H100", cpu=8, memory=32768, timeout=45 * 60)
def run() -> list[str]:
    sh("cd /work/syncsan && cmake -S src -B build -DNVBIT_ROOT=/opt/nvbit/nvbit_release_x86_64 "
       "-DNVBIT_TOOL_CUDA_ARCHITECTURES=90 && cmake --build build -j8", 900)
    gen = ".venv/bin/python amk_cli.py generate toy --gpu h100 --prompt-ids 1,2,3 --max-tokens 16 --verify"
    tool = "LD_PRELOAD=/work/syncsan/build/syncsan.so SYNCSAN_SHARED=0"
    keep = ("matches_eager", "syncsan: ADVERSARIAL", "distinct racing")
    lines = []
    for name, env, fix in (("native", "", False), ("adversarial", tool + " SYNCSAN_ADVERSARIAL=1", False),
                           ("control", tool + " SYNCSAN_ADVERSARIAL=control", False),
                           ("adversarial, acquire wait", tool + " SYNCSAN_ADVERSARIAL=1", True)):
        if fix:
            pathlib.Path("/tmp/fix_amk.py").write_text(FIX)
            lines.append("fix: " + sh("cd /opt/amk && python /tmp/fix_amk.py && grep -c ld.acquire vm/sync.cuh && rm -rf ~/.cache/torch_extensions", 60).strip())
        out = sh(f"cd /opt/amk && export PATH=/opt/amk/.venv/bin:$PATH && env {env} {gen}", 1200)
        lines.append(f"{name}: " + " | ".join(l.strip() for l in out.splitlines() if any(k in l for k in keep)))
    return lines


@app.local_entrypoint()
def main():
    print("\n".join(run.remote()))
