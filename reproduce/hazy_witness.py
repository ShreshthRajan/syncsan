"""Native hardware witness for Hazy's handoff races, no instrumentation (the method of HazyResearch/Megakernels#15).

Before every decode step, the buffer named by CANARY (attn_out, silu_out, ...) is filled with 1e4. Every op that reads
the buffer runs after the op that writes it, so a correct kernel never reads the canary; a consumer that reads the
buffer before the producer's writes are visible does. Each of NUM generations' tokens is recorded; the script prints how
many distinct token sequences came out and how often each did.

    CANARY=attn_out python witness.py mode=mk model=... ntok=64 num_warmup=0 num_iters=100 mk_dir=...
"""
import atexit
import collections
import os
import runpy
import sys

import torch
import megakernels.generators as G

CANARY = os.environ.get("CANARY", "")
seqs = []

_fill = G.MK_Generator.fill
def fill(self):
    _fill(self)
    if CANARY:
        getattr(self.schedule.globs, CANARY).fill_(1e4)
G.MK_Generator.fill = fill

# A step that read the canary shows it in the logits: 1e4 in an activation gives logits far above the normal range.
big_steps = []
_run = G.MK_Generator.run
def run(self, input_ids, pos_id):
    out = _run(self, input_ids, pos_id)
    m = float(self.schedule.globs.logits.float().abs().max())
    big_steps.append(m > 1e3 or m != m)
    return out
G.MK_Generator.run = run

_generate = G.MK_Generator.generate
hit_gens = []
def generate(self, output_tokens, *a, **k):
    start = len(big_steps)
    r = _generate(self, output_tokens, *a, **k)
    torch.cuda.synchronize()
    seqs.append(tuple(output_tokens[0].tolist()))
    hit_gens.append(any(big_steps[start:]))
    return r
G.MK_Generator.generate = generate

def report():
    c = collections.Counter(seqs)
    print(f"WITNESS canary={CANARY or 'none'} generations={len(seqs)} distinct={len(c)} "
          f"canary_steps={sum(big_steps)}/{len(big_steps)} canary_generations={sum(hit_gens)}/{len(hit_gens)}")
    for s, n in c.most_common(4):
        print(f"WITNESS {n} x {list(s[:16])}")
atexit.register(report)

sys.argv = ["generate.py"] + sys.argv[1:]
runpy.run_path("megakernels/scripts/generate.py", run_name="__main__")
