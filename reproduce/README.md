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
