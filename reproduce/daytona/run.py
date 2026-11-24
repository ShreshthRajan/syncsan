"""Run a bash script on a Daytona GPU sandbox with the syncsan tree at /work/syncsan, then download /work/out.

    pip install "daytona[v0-sdk]"
    DAYTONA_API_KEY=... python reproduce/daytona/run.py [--gpu H100] [--image IMAGE] [--name tag] [--keep] SCRIPT.sh [ARGS...]

The script's output streams to results/<name>_<time>/run.log; everything the script writes under /work/out is
downloaded to that folder. The sandbox is deleted at the end unless --keep.
"""
import argparse
import io
import os
import pathlib
import sys
import tarfile
import time

from daytona import Daytona
from daytona.v0 import CreateSandboxFromImageParams, GpuType, Resources
from daytona.v0.common.process import SessionExecuteRequest

HERE = pathlib.Path(__file__).resolve().parent
TREE = HERE.parent.parent   # the repository root

ap = argparse.ArgumentParser()
ap.add_argument("script")
ap.add_argument("--gpu", default="H100")
ap.add_argument("--name", default="")
ap.add_argument("--keep", action="store_true")
ap.add_argument("--sandbox", default="", help="reuse a kept sandbox (its /opt installs stay)")
ap.add_argument("--image", default="pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
ap.add_argument("--hours", type=float, default=3.0, help="give up after this long")
ap.add_argument("--mem", type=int, default=64)
ap.add_argument("args", nargs=argparse.REMAINDER)
a = ap.parse_args()

name = a.name or pathlib.Path(a.script).stem
out = TREE / "results" / time.strftime(f"daytona_{name}_%Y%m%d_%H%M%S")
out.mkdir(parents=True, exist_ok=True)
log = open(out / "run.log", "w")
def say(s):
    print(s, flush=True); log.write(s + "\n"); log.flush()

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    t.add(TREE, arcname="syncsan", filter=lambda i: None if any(p in i.name for p in ("/build", "/results", "__pycache__", ".git/")) else i)
say(f"tree tarball {buf.tell() / 1e6:.1f} MB; creating {a.gpu} sandbox from {a.image}")

d = Daytona().v0
t0 = time.time()
if a.sandbox:
    sb = d.get(a.sandbox)
    sb.wait_for_sandbox_start(timeout=0)
else:
    sb = d.create(CreateSandboxFromImageParams(image=a.image, auto_delete_interval=0, auto_stop_interval=0,
                                               resources=Resources(cpu=8, memory=a.mem, disk=200, gpu=1, gpu_type=[GpuType(a.gpu)])),
                  timeout=0)
say(f"sandbox {sb.id} up after {time.time() - t0:.0f} s")
try:
    sb.fs.upload_file(buf.getvalue(), "/tmp/tree.tgz")
    sb.fs.upload_file(pathlib.Path(a.script).read_bytes(), "/tmp/job.sh")
    sb.process.exec("rm -rf /work/out /work/syncsan/daytona /work/syncsan/luce /work/syncsan/zoo; mkdir -p /work/out && tar xzf /tmp/tree.tgz -C /work", timeout=600)
    sess = "job%d" % int(time.time())
    sb.process.create_session(sess)
    cmd = "cd /work && bash /tmp/job.sh " + " ".join(a.args) + " > /work/out/job.log 2>&1; echo EXIT $? >> /work/out/job.log"
    r = sb.process.execute_session_command(sess, SessionExecuteRequest(command=cmd, run_async=True))
    seen = 0
    while time.time() - t0 < a.hours * 3600:
        time.sleep(30)
        txt = sb.process.exec("cat /work/out/job.log 2>/dev/null", timeout=60).result or ""
        if len(txt) > seen:
            new = txt[seen:]; seen = len(txt)
            print(new, end="", flush=True); log.write(new); log.flush()
        if "\nEXIT " in txt or txt.startswith("EXIT "):
            break
    else:
        say("TIMED OUT")
    sb.process.exec("cd /work && tar czf /tmp/out.tgz out", timeout=600)
    data = sb.fs.download_file("/tmp/out.tgz")
    tarfile.open(fileobj=io.BytesIO(data)).extractall(out)
    say(f"results in {out}; wall {time.time() - t0:.0f} s")
finally:
    if a.keep:
        say(f"kept sandbox {sb.id}")
    else:
        sb.delete()
        say("sandbox deleted")
