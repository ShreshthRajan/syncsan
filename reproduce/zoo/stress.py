"""Native repeat test for a megakernel-zoo offline engine: R greedy decodes of NTOK tokens; counts decodes whose tokens
or logits differ from the first, and times steps.

    python stress.py ENGINE_DIR MODEL_DIR NTOK R
"""
import sys
import time

import torch
from transformers import AutoTokenizer

eng_dir, model_dir, ntok, reps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, eng_dir)
from api import Engine  # noqa: E402

ids = AutoTokenizer.from_pretrained(model_dir).encode("The capital of France is", add_special_tokens=False)
eng = Engine(model_dir, max_batch=1, max_context=256)


def decode():
    eng.reset()
    x = None
    for t in ids:
        x = eng.step(torch.tensor([t], dtype=torch.int64, device="cuda"))
    gen, logits = [], []
    for _ in range(ntok):
        logits.append(x[0].clone())
        nxt = int(x[0].argmax())
        gen.append(nxt)
        x = eng.step(torch.tensor([nxt], dtype=torch.int64, device="cuda"))
    return gen, torch.stack(logits)


g0, l0 = decode()
torch.cuda.synchronize()
bad_tok = bad_bits = 0
t0 = time.time()
for _ in range(reps):
    g, l = decode()
    bad_tok += g != g0
    bad_bits += not torch.equal(l, l0)
torch.cuda.synchronize()
dt = time.time() - t0
print(f"STRESS {reps} decodes x {ntok} tokens: tokens differ in {bad_tok}, logits differ in {bad_bits}; "
      f"{dt / (reps * (ntok + len(ids))) * 1e6:.1f} us per step", flush=True)
