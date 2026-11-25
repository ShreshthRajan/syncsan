"""Native canary witness for Lucebox's decode megakernel, no instrumentation. Before every decode step the scratch buffers
named by CANARY are filled with NaN; every phase writes a scratch buffer before the next phase reads it, so a correct kernel
never reads the NaN, and a phase that reads before the writes are visible turns the hidden state into NaN. Counts the
decodes (of R, NTOK tokens each) in which a step's final-norm hidden state held a NaN.

    CANARY=_qkv_scratch,_attn_out,_mlp_inter python canary.py NTOK R
"""
import os
import sys

import torch

sys.path.insert(0, "/opt/luce/optimizations/megakernel")
from model import Decoder  # noqa: E402

ntok, reps = int(sys.argv[1]), int(sys.argv[2])
names = [n for n in os.environ.get("CANARY", "_qkv_scratch,_kv_scratch,_attn_out,_mlp_inter,_activations").split(",") if n]
dec = Decoder(verbose=False)
ids = dec.tokenizer.encode("The capital of France is", add_special_tokens=False)
_step = dec.step
bad_steps = [0]
def step(t):
    for n in names:
        getattr(dec, n).fill_(float("nan"))
    r = _step(t)
    if torch.isnan(dec._normalized).any().item():
        bad_steps[0] += 1
    return r
dec.step = step
hit = 0
for _ in range(reps):
    dec.reset()
    before = bad_steps[0]
    for t in ids[:-1]:
        dec.step(t)
    nid = ids[-1]
    for _ in range(ntok):
        nid = dec.step(nid)
    hit += bad_steps[0] > before
print(f"CANARY buffers={','.join(names)} decodes={reps} decodes_with_nan={hit} nan_steps={bad_steps[0]}/{reps * (ntok + len(ids) - 1)}",
      flush=True)
