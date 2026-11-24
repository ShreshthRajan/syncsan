"""Greedy decode with a megakernel-zoo offline engine through its step() contract; saves the logits, compares with a
reference file (bitwise and top-1) when one exists.

    python drive.py ENGINE_DIR MODEL_DIR NTOK OUT.pt [REF.pt]
"""
import os
import sys

import torch
from transformers import AutoTokenizer

eng_dir, model_dir, ntok, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
ref = sys.argv[5] if len(sys.argv) > 5 else None
sys.path.insert(0, eng_dir)
from api import Engine  # noqa: E402

tok = AutoTokenizer.from_pretrained(model_dir)
ids = tok.encode("The capital of France is", add_special_tokens=False)
eng = Engine(model_dir, max_batch=1, max_context=256)
eng.reset()
logits = []
x = None
for t in ids:
    x = eng.step(torch.tensor([t], dtype=torch.int64, device="cuda"))
gen = []
for _ in range(ntok):
    logits.append(x[0].clone())
    nxt = int(x[0].argmax())
    gen.append(nxt)
    x = eng.step(torch.tensor([nxt], dtype=torch.int64, device="cuda"))
torch.cuda.synchronize()
L = torch.stack(logits).cpu()
torch.save({"tokens": gen, "logits": L}, out)
print("DRIVE tokens:", gen, "text:", repr(tok.decode(gen)), flush=True)
if ref and os.path.exists(ref):
    R = torch.load(ref)
    d = (L - R["logits"]).abs()
    print(f"DRIVE vs native: tokens {'MATCH' if gen == R['tokens'] else 'DIFFER'} ({gen} vs {R['tokens']}); "
          f"logits {'BITWISE' if torch.equal(L, R['logits']) else 'differ'}, max|diff| {float(d.max()):.4g}, "
          f"positions differing {int((d.amax(1) > 0).sum())}/{len(gen)}", flush=True)
if os.environ.get("DRIVE_HF") == "1":   # reference: Hugging Face BF16 greedy decode of the same prompt
    from transformers import AutoModelForCausalLM
    del eng
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda")
    o = m.generate(torch.tensor([ids], device="cuda"), max_new_tokens=ntok, do_sample=False)[0, len(ids):].tolist()
    print(f"DRIVE HF greedy: {o} {repr(tok.decode(o))}; engine {'MATCHES' if o == gen else 'DIFFERS FROM'} HF", flush=True)
