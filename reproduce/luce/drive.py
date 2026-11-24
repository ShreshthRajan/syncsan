"""Greedy decode with the Lucebox Qwen3.5-0.8B megakernel (decode path only: one decode_kernel + lm_head_kernel per
token); saves tokens and the final hidden state of every step, compares with a reference file when one exists.

    python drive.py NTOK OUT.pt [REF.pt] [--hf]
"""
import os
import sys

import torch

sys.path.insert(0, "/opt/luce/optimizations/megakernel")
from model import Decoder  # noqa: E402

ntok, out = int(sys.argv[1]), sys.argv[2]
ref = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else None
dec = Decoder(verbose=False)
tok = dec.tokenizer
ids = tok.encode("The capital of France is", add_special_tokens=False)
dec.reset()
for t in ids[:-1]:
    dec.step(t)
nid, gen, hid = ids[-1], [], []
for _ in range(ntok):
    nid = dec.step(nid)
    gen.append(nid)
    hid.append(dec._normalized.detach().clone().cpu())
H = torch.stack(hid)
torch.save({"tokens": gen, "normed": H}, out)
print("DRIVE tokens:", gen, "text:", repr(tok.decode(gen)), flush=True)
if ref and os.path.exists(ref):
    R = torch.load(ref)
    d = (H - R["normed"]).abs()
    print(f"DRIVE vs native: tokens {'MATCH' if gen == R['tokens'] else 'DIFFER'} ({gen} vs {R['tokens']}); "
          f"final-norm hidden {'BITWISE' if torch.equal(H, R['normed']) else 'differ'}, max|diff| {float(d.max()):.4g}, "
          f"steps differing {int((d.amax(1) > 0).sum())}/{len(gen)}", flush=True)
if "--hf" in sys.argv:   # reference: Hugging Face greedy decode of the same prompt
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="cuda")
    o = m.generate(torch.tensor([ids], device="cuda"), max_new_tokens=ntok, do_sample=False)[0, len(ids):].tolist()
    print(f"DRIVE HF greedy: {o} {repr(tok.decode(o))}; megakernel {'MATCHES' if o == gen else 'DIFFERS FROM'} HF", flush=True)
