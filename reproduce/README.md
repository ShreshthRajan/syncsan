# Reproducing the results

These scripts rent a GPU through [Modal](https://modal.com) (`pip install modal`, `modal setup`). An H100 costs about
$4 an hour there.

- `modal_cases.py` builds syncsan and runs the 37 known-answer cases in `tests/` (about 2 minutes).
- `modal_hazy.py` builds Hazy Research's Llama megakernel in its original form and with each fix, then prints the race
  report, the adversarial and control outputs, and the latency of each build (about 30 minutes, most of it the image
  build). The table in [docs/hazy-megakernel.md](../docs/hazy-megakernel.md) comes from this script.
- `modal_amk.py` runs AutoMegaKernel's toy model with `--verify` natively, in adversarial mode, under the control, and
  with an acquire in its counter wait (about 15 minutes). The table in [docs/automegakernel.md](../docs/automegakernel.md)
  comes from this script.
- `patch_hazy.py DIR fences|acqrel` writes the two fixes into a copy of `demos/low-latency-llama`.
- `modal_superlu.py` builds SuperLU_DIST at the buggy commit, the official fix and the fix plus a producer fence, and
  runs the GPU triangular solve natively, under syncsan, in adversarial mode and under the control (about 30 minutes).
  The table in [docs/superlu.md](../docs/superlu.md) comes from this script.
- `modal_cutlass.py` builds CUTLASS 3.6.0 and 3.7.0 with the reproducer from issue #2027 and runs it natively and under
  syncsan. The table in [docs/cutlass-streamk.md](../docs/cutlass-streamk.md) comes from this script.
- `modal_zoo.py` and `modal_luce.py` run megakernel-zoo's Qwen3-8B engine and Lucebox's Qwen3.5-0.8B megakernel
  natively, under syncsan, in adversarial mode and under the control; `--fixed acquire` (zoo) or `--fixed both`
  (Lucebox) applies the fix in `zoo/patch_sync.py` or `luce/patch_sync.py`. The tables in
  [docs/megakernel-zoo.md](../docs/megakernel-zoo.md) and [docs/lucebox.md](../docs/lucebox.md) come from these scripts.
- `hazy_witness.py` is the native canary test in [docs/hazy-megakernel.md](../docs/hazy-megakernel.md): run it in place
  of `megakernels/scripts/generate.py`, with `CANARY=silu_out` or `CANARY=attn_out`, inside the image `modal_hazy.py`
  builds.
