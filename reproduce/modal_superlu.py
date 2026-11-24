"""SuperLU_DIST GPU triangular solve (new 3D solve, SRC/cuda/pdgstrs_lsum_cuda.cu) under syncsan on a rented H100.

Builds (full SuperLU_DIST, double precision, CUDA, -lineinfo, 1 MPI rank):
  bug     4755a5e  HIP port: consumer spin-waits became atomicAdd(&fmod/bmod, 0) with no __threadfence()
  fix     587f38f  restores __threadfence() after the poll at all 7 sites
  fixrel  587f38f + reproduce/superlu/release_fence.sh (also a __threadfence() before each producer atomicSub)
Test: EXAMPLE/pddrive3d on example matrices and on reproduce/superlu/gen_mtx.py matrices (2D convection-diffusion, stacked
block-diagonal), solution error ||X - Xtrue||/||X|| printed by the driver, SUPERLU_ACC_SOLVE=1 (GPU trisolve), no
iterative refinement (-i 0). A run's args may start with VAR=value overrides (e.g. SUPERLU_RELAX=4 SUPERLU_MAXSUP=16
for many small supernodes, so many cross-block fmod/bmod dependencies). The batched drivers (-b) need MAGMA, so they are not used.

    modal run reproduce/modal_superlu.py [--native "..."] [--san "..."]
"""
import pathlib
import subprocess
import time

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
NVBIT = "https://github.com/NVlabs/NVBit/releases/download/v1.8.2/nvbit-Linux-x86_64-1.8.2.tar.bz2"
BUILDS = {"bug": "4755a5e116b6ebae8b92ff13e9b77f0b145d23bd", "fix": "587f38f6dc14c29c1f6f7f14d68dde3080a081c0",
          "fixrel": "587f38f6dc14c29c1f6f7f14d68dde3080a081c0"}
LIB = "/usr/lib/x86_64-linux-gnu"
CMAKE = ("cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx "
         "-DTPL_ENABLE_CUDALIB=ON -DCMAKE_CUDA_ARCHITECTURES=90 '-DCMAKE_CUDA_FLAGS=-lineinfo -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi' "
         "-DTPL_ENABLE_PARMETISLIB=OFF -DTPL_ENABLE_INTERNAL_BLASLIB=OFF "
         f"-DTPL_BLAS_LIBRARIES={LIB}/libopenblas.so -DTPL_ENABLE_LAPACKLIB=ON -DTPL_LAPACK_LIBRARIES={LIB}/liblapack.so "
         "-DXSDK_ENABLE_Fortran=OFF -Denable_python=OFF -Denable_tests=OFF "
         "-Denable_openmp=OFF")
ENV = ("OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 OMP_NUM_THREADS=1 SUPERLU_ACC_OFFLOAD=1 GPU3DVERSION=1 "
       "ANC25D=0 NEW3DSOLVE=1 NEW3DSOLVETREECOMM=1 SUPERLU_ACC_SOLVE=1 SUPERLU_BIND_MPI_GPU=1 SUPERLU_MAXSUP=256 "
       "SUPERLU_RELAX=64 SUPERLU_NUM_GPU_STREAMS=1")


def sh(cmd, timeout):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, executable="/bin/bash")
        rc, out = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out = 124, (e.stdout or b"").decode(errors="replace") + (e.stderr or b"").decode(errors="replace")
    return rc, f"$ {cmd}\n[exit {rc} after {time.time() - t0:.0f} s]\n{out}"


def build(b):
    d = f"/opt/slu_{b}"
    patch = " && bash /opt/release_fence.sh" if b == "fixrel" else ""
    return (f"cp -r /opt/superlu_dist {d} && cd {d} && git checkout -q {BUILDS[b]}{patch} && {CMAKE} > cmake.log 2>&1 "
            f"&& cmake --build build -j16 --target pddrive3d_block_diag pddrive3d > build.log 2>&1 || (tail -n 80 {d}/cmake.log {d}/build.log; false)")


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "bzip2", "build-essential", "libopenmpi-dev", "openmpi-bin", "libopenblas-dev", "liblapack-dev")
    .pip_install("cmake>=3.29")
    .run_commands(f"mkdir -p /opt/nvbit && curl -sL {NVBIT} | tar xj -C /opt/nvbit",
                  "git clone -q https://github.com/xiaoyeli/superlu_dist.git /opt/superlu_dist")
    .add_local_file(ROOT / "reproduce" / "superlu" / "release_fence.sh", "/opt/release_fence.sh", copy=True)
    .run_commands(*[build(b) for b in BUILDS])
    .add_local_dir(ROOT, "/work/syncsan", ignore=["build", "results", ".git"])
)
app = modal.App("syncsan-superlu", image=image)


@app.function(gpu="H100", cpu=8, memory=65536, timeout=45 * 60)
def run(native: list, san: str, kernel: str, builds: list, modes: list) -> dict:
    logs, summary = {}, []
    def step(name, cmd, timeout):
        rc, out = sh(cmd, timeout); logs[name] = out; summary.append(f"{name}: exit {rc}"); return rc, out
    step("gpu", "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; "
         + "; ".join(f"cd /opt/slu_{b} && git log --oneline -1 && grep -c '__threadfence();' SRC/cuda/pdgstrs_lsum_cuda.cu && "
                     f"grep -c slu_release_sub SRC/cuda/pdgstrs_lsum_cuda.cu" for b in BUILDS), 300)
    step("matrices", "cd /tmp && python3 /work/syncsan/reproduce/superlu/gen_mtx.py lap2d 64 lap64.mtx && "
         "python3 /work/syncsan/reproduce/superlu/gen_mtx.py lap2d 160 lap160.mtx && python3 /work/syncsan/reproduce/superlu/gen_mtx.py blockdiag 256 40 bd256.mtx && "
         "python3 /work/syncsan/reproduce/superlu/gen_mtx.py blockdiag 16 40 bd16.mtx", 300)
    step("build_syncsan", "cd /work/syncsan && cmake -S . -B build -DNVBIT_ROOT=/opt/nvbit/nvbit_release_x86_64 "
         "-DNVBIT_TOOL_CUDA_ARCHITECTURES=90 && cmake --build build -j8", 900)
    T = "LD_PRELOAD=/work/syncsan/build/syncsan.so SYNCSAN_VERBOSE=1 SYNCSAN_SHARED=0" + (f" SYNCSAN_KERNEL={kernel}" if kernel else "")
    keep = lambda out: " | ".join(l.strip() for l in out.splitlines() if any(k in l for k in (
        "Xtrue", "Berr", "berr", "ERROR", "rror", "Fatal", "instrumenting", "syncsan: ADVERSARIAL", "distinct racing", "Segmentation", "Aborted")))[:900]
    for b in builds:
        run_ = lambda env, args: f"cd /opt/slu_{b}/build/EXAMPLE && mpirun --allow-run-as-root -np 1 env {ENV} {env} {args}"
        for i, args in enumerate(native):
            rc, out = step(f"{b}_native{i}", run_("", args), 600); summary[-1] += " | " + keep(out)
        for name, env in (("syncsan", T), ("adversarial", T + " SYNCSAN_ADVERSARIAL=1"), ("control", T + " SYNCSAN_ADVERSARIAL=control")):
            if name in modes:
                rc, out = step(f"{b}_{name}", run_(env, san), 900); summary[-1] += " | " + keep(out)
    return {"summary": summary, "logs": logs}


@app.local_entrypoint()
def main(native: str = "SUPERLU_RELAX=4 SUPERLU_MAXSUP=16 ./pddrive3d -i 0 /tmp/lap160.mtx;"
                       "SUPERLU_RELAX=4 SUPERLU_MAXSUP=16 ./pddrive3d -i 0 /tmp/bd256.mtx;./pddrive3d -i 0 /tmp/lap160.mtx",
         san: str = "SUPERLU_RELAX=4 SUPERLU_MAXSUP=16 ./pddrive3d -i 0 /tmp/lap64.mtx", kernel: str = "lsum_",
         builds: str = "bug,fix,fixrel", modes: str = "syncsan,adversarial,control", tag: str = ""):
    t0 = time.time()
    res = run.remote([a for a in native.split(";") if a], san, kernel, builds.split(","), modes.split(","))
    out = ROOT / "results" / time.strftime(f"superlu{'_' + tag if tag else ''}_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    for name, text in res["logs"].items():
        (out / f"{name}.log").write_text(text)
    (out / "SUMMARY.txt").write_text(f"native: {native}\nsyncsan: {san}\nkernel: {kernel}\n" + "\n".join(res["summary"])
                                     + f"\nwall time {time.time() - t0:.0f} s\n")
    print("\n".join(res["summary"]))
    print(f"results in {out}")
