"""Hazy Research's low-latency Llama-1B megakernel under syncsan on a rented H100 through Modal (https://modal.com).

Builds the megakernel three ways (original, fences, acqrel; see patch_hazy.py), then runs: the native model, syncsan's
race report, adversarial mode and its control on the original, adversarial mode on the release/acquire build, and the
latency of each build. Llama-3.2-1B-Instruct comes from the unsloth mirror, which holds the same weights as meta-llama's.

    modal run reproduce/modal/hazy.py
"""
import pathlib
import subprocess

import modal

ROOT = pathlib.Path(__file__).resolve().parents[2]
NVBIT = "https://github.com/NVlabs/NVBit/releases/download/v1.8.2/nvbit-Linux-x86_64-1.8.2.tar.bz2"
MK_COMMIT = "7309cec"
MODEL = "unsloth/Llama-3.2-1B-Instruct"
MAKE = ("THUNDERKITTENS_ROOT=/opt/mk/ThunderKittens MEGAKERNELS_ROOT=/opt/mk PYTHON_VERSION=3.12 GPU=H100 make 2>&1 "
        "| grep -iE ' error' | head")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "bzip2", "build-essential")
    .pip_install("cmake>=3.29")
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==4.48.3", "pydra-config>=0.0.13", "accelerate", "tabulate", "tqdm", "einops", "pybind11",
                 "matplotlib", "psutil", "art", "openai", "huggingface_hub[cli]")
    .run_commands(f"mkdir -p /opt/nvbit && curl -sL {NVBIT} | tar xj -C /opt/nvbit",
                  "git clone -q https://github.com/HazyResearch/Megakernels.git /opt/mk && cd /opt/mk && "
                  f"git checkout -q {MK_COMMIT} && git submodule update --init --recursive -q && pip install -e . --no-deps -q",
                  f"huggingface-cli download {MODEL} --exclude '*.pth' 'original/*' > /dev/null",
                  f"cd /opt/mk/demos/low-latency-llama && {MAKE}")
    .add_local_file(ROOT / "reproduce" / "hazy" / "patch.py", "/opt/patch_hazy.py", copy=True)
    .run_commands(*[f"cp -r /opt/mk/demos/low-latency-llama /opt/mk/demos/{m} && rm -f /opt/mk/demos/{m}/*.so && "
                    f"python /opt/patch_hazy.py /opt/mk/demos/{m} {m} && cd /opt/mk/demos/{m} && {MAKE}"
                    for m in ("fences", "acqrel")])
    .add_local_dir(ROOT, "/work/syncsan", ignore=["build", ".git"])
)
app = modal.App("syncsan-hazy", image=image)


def sh(cmd: str, timeout: int) -> str:
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, executable="/bin/bash")
        return p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return "timed out"


@app.function(gpu="H100", cpu=8, memory=65536, timeout=60 * 60)
def run() -> list[str]:
    sh("cd /work/syncsan && cmake -S src -B build -DNVBIT_ROOT=/opt/nvbit/nvbit_release_x86_64 "
       "-DNVBIT_TOOL_CUDA_ARCHITECTURES=90 && cmake --build build -j8", 900)
    gen = f"python megakernels/scripts/generate.py mode=mk model={MODEL} num_warmup=0 num_iters=1 ntok=8"
    tool = "LD_PRELOAD=/work/syncsan/build/syncsan.so SYNCSAN_KERNEL=megakernel::mk SYNCSAN_SHARED=0"
    keep = ("Output text", "syncsan: ADVERSARIAL", "distinct racing", "Average time")
    lines = []
    runs = [("native", "", "low-latency-llama"),
            ("syncsan report", tool, "low-latency-llama"),
            ("adversarial, original", tool + " SYNCSAN_ADVERSARIAL=1", "low-latency-llama"),
            ("control, original", tool + " SYNCSAN_ADVERSARIAL=control", "low-latency-llama"),
            ("adversarial, release/acquire", tool + " SYNCSAN_ADVERSARIAL=1", "acqrel")]
    for name, env, build in runs:
        out = sh(f"cd /opt/mk && env {env} {gen} mk_dir=/opt/mk/demos/{build}", 1200)
        lines.append(f"{name}: " + " | ".join(l.strip() for l in out.splitlines() if l.startswith(keep) or "Output text" in l))
    for build in ("low-latency-llama", "fences", "acqrel"):   # latency, 128 tokens, average of 10
        out = sh(f"cd /opt/mk && python megakernels/scripts/generate.py mode=mk model={MODEL} mk_dir=/opt/mk/demos/{build} "
                 "ntok=128 num_warmup=3 num_iters=10", 900)
        lines.append(f"latency {build}: " + " | ".join(l.strip() for l in out.splitlines() if "Average time" in l))
    return lines


@app.local_entrypoint()
def main():
    print("\n".join(run.remote()))
