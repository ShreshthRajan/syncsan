"""Native repeat test for the Lucebox megakernel: R greedy decodes of NTOK tokens; counts decodes whose tokens or final
hidden states differ from the first, and times decode steps.

    python stress.py NTOK R
"""
import sys
import time

import torch

sys.path.insert(0, "/opt/luce/optimizations/megakernel")
from model import Decoder  # noqa: E402

ntok, reps = int(sys.argv[1]), int(sys.argv[2])
dec = Decoder(verbose=False)
ids = dec.tokenizer.encode("The capital of France is", add_special_tokens=False)


def decode():
    dec.reset()
    for t in ids[:-1]:
        dec.step(t)
    nid, gen, hid = ids[-1], [], []
    for _ in range(ntok):
        nid = dec.step(nid)
        gen.append(nid)
        hid.append(dec._normalized.detach().clone())
    return gen, torch.stack(hid)


g0, h0 = decode()
torch.cuda.synchronize()
bad_tok = bad_bits = 0
t0 = time.time()
for _ in range(reps):
    g, h = decode()
    bad_tok += g != g0
    bad_bits += not torch.equal(h, h0)
torch.cuda.synchronize()
dt = time.time() - t0
print(f"STRESS {reps} decodes x {ntok} tokens: tokens differ in {bad_tok}, hidden states differ in {bad_bits}; "
      f"{dt / (reps * (ntok + len(ids) - 1)) * 1e6:.1f} us per step", flush=True)
