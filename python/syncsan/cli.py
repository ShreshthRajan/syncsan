"""syncsan command line.

    syncsan build                      download NVBit 1.8.2 and build the tool (needs nvcc and cmake >= 3.29), once
    syncsan run [options] -- CMD ...   run CMD with every (or the chosen) CUDA kernel checked for races
    syncsan where                      print the path of the built tool

Exit status of `run`: CMD's own, or --exitcode N when syncsan reported a race (for CI).
"""
import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import urllib.request

NVBIT_URL = "https://github.com/NVlabs/NVBit/releases/download/v1.8.2/nvbit-Linux-x86_64-1.8.2.tar.bz2"
CACHE = pathlib.Path(os.environ.get("SYNCSAN_HOME", pathlib.Path.home() / ".cache" / "syncsan"))
SRC = pathlib.Path(__file__).resolve().parent / "csrc"


def tool_path() -> pathlib.Path:
    return CACHE / "build" / "syncsan.so"


def build(arch: str) -> pathlib.Path:
    nvbit = CACHE / "nvbit_release_x86_64"
    if not nvbit.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        tar = CACHE / "nvbit.tar.bz2"
        print(f"syncsan: downloading NVBit 1.8.2 to {CACHE}", file=sys.stderr)
        urllib.request.urlretrieve(NVBIT_URL, tar)
        with tarfile.open(tar) as t:
            t.extractall(CACHE)
        tar.unlink()
    src = CACHE / "src"
    if src.exists():
        shutil.rmtree(src)
    shutil.copytree(SRC, src)
    b = CACHE / "build"
    subprocess.run(["cmake", "-S", str(src), "-B", str(b), f"-DNVBIT_ROOT={nvbit}", f"-DNVBIT_TOOL_CUDA_ARCHITECTURES={arch}"], check=True)
    subprocess.run(["cmake", "--build", str(b), "-j", str(os.cpu_count() or 4)], check=True)
    print(tool_path())
    return tool_path()


def run(a: argparse.Namespace) -> int:
    so = tool_path()
    if not so.exists():
        build(a.arch)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        print("syncsan run: give a command after --", file=sys.stderr)
        return 2
    env = dict(os.environ)
    env["LD_PRELOAD"] = f"{so}:{env['LD_PRELOAD']}" if env.get("LD_PRELOAD") else str(so)
    env["SYNCSAN_EXITCODE"] = str(a.exitcode)
    for flag, var in ((a.kernel, "SYNCSAN_KERNEL"), (a.sample, "SYNCSAN_SAMPLE"), (a.instrument, "SYNCSAN_INSTRUMENT"),
                      (a.async_, "SYNCSAN_ASYNC")):
        if flag:
            env[var] = str(flag)
    if a.verbose:
        env["SYNCSAN_VERBOSE"] = str(a.verbose)
    if a.check_nc:
        env["SYNCSAN_CHECK_NC"] = "1"
    return subprocess.run(cmd, env=env).returncode


def main() -> None:
    p = argparse.ArgumentParser(prog="syncsan", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="what", required=True)
    pb = sub.add_parser("build", help="download NVBit and build the tool")
    pb.add_argument("--arch", default="90", help="CUDA architecture of the tool's device code (90 = H100, 80 = A100)")
    sub.add_parser("where", help="print the built tool's path")
    pr = sub.add_parser("run", help="run a command under syncsan")
    pr.add_argument("--kernel", help="check only kernels whose name contains this")
    pr.add_argument("--sample", help="k or k:seed: check one plain load/store site in k (sync operations always checked)")
    pr.add_argument("--instrument", help="comma list of mem,bar,fence (default all)")
    pr.add_argument("--async", dest="async_", help="comma list of tma,bulk,wait (default all)")
    pr.add_argument("--check-nc", action="store_true", help="also check ld.global.nc (read-only) loads")
    pr.add_argument("--exitcode", type=int, default=0, help="exit with this code when a race is reported (0: CMD's own)")
    pr.add_argument("-v", "--verbose", type=int, default=0)
    pr.add_argument("--arch", default="90")
    pr.add_argument("cmd", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.what == "build":
        build(a.arch)
    elif a.what == "where":
        print(tool_path() if tool_path().exists() else "not built: run `syncsan build`")
    else:
        sys.exit(run(a))


if __name__ == "__main__":
    main()
