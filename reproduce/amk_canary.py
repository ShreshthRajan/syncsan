"""Native canary witness for AutoMegaKernel, no instrumentation. Every launch starts with the activation arena filled with
0xFF bytes (NaN in f32 and bf16) instead of zeros; every op writes its activation before a consumer reads it, so a correct
kernel never reads the NaN, and a consumer that reads before the producer's writes are visible puts NaN in the logits.
Counts the generations (of R) with a NaN in any output.

    cd /opt/amk && .venv/bin/python canary.py R [MAX_TOKENS]
"""
import sys

import torch

sys.path.insert(0, ".")
import vm.loader as L  # noqa: E402
from generate import generate  # noqa: E402

R = int(sys.argv[1]); T = int(sys.argv[2]) if len(sys.argv) > 2 else 16
_zeros = torch.zeros
def nan_zeros(*a, **k):
    t = _zeros(*a, **k)
    if k.get("dtype") is torch.uint8 and t.numel() > 256:   # the activation arena
        t.fill_(255)
    return t
_full, _steady = L.MegakernelVM._run_full_build, L.MegakernelVM._run_steady_state
def full(self, *a, **k):
    torch.zeros = nan_zeros
    try:
        return _full(self, *a, **k)
    finally:
        torch.zeros = _zeros
def steady(self, *a, **k):
    for t in getattr(self, "_keepalive", []):
        if t.dtype is torch.uint8 and t.numel() > 256:
            t.fill_(255)
    return _steady(self, *a, **k)
L.MegakernelVM._run_full_build, L.MegakernelVM._run_steady_state = full, steady
nan_runs = [0]
_run = L.MegakernelVM.run
def run(self, *a, **k):
    out = _run(self, *a, **k)
    if any(v.is_floating_point() and torch.isnan(v).any().item() for v in out.values()):
        nan_runs[0] += 1
    return out
L.MegakernelVM.run = run
hit = 0
for _ in range(R):
    before = nan_runs[0]
    generate("toy", "h100", [1, 2, 3], T, device="cuda")
    hit += nan_runs[0] > before
print(f"CANARY amk generations={R} with_nan={hit} nan_launches={nan_runs[0]}", flush=True)
