# Daytona runners

`run.py` rents one GPU sandbox on [Daytona](https://www.daytona.io), uploads this repository to `/work/syncsan`, runs a
job script there, downloads what the script writes to `/work/out` into `results/`, and deletes the sandbox. Daytona's
lower tiers reach only allow-listed hosts, and `pypi.nvidia.com` is not one of them, so the jobs use the
`pytorch/pytorch` images, which ship torch.

- `mirage.sh VARIANTS REPS NTOK RUNSAN SANVARIANTS SANTOK` builds Mirage MPK at ed771bd (with `mirage_codegen_fix.py`),
  runs the Qwen3 demo natively `REPS` times per variant (`orig`, `nocap`, `nocap_fix`), compares outputs, and with
  `RUNSAN=1` runs syncsan on `worker_kernel`. [docs/mirage.md](../../docs/mirage.md):
  `python reproduce/daytona/run.py --image pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel reproduce/daytona/mirage.sh orig 0 2 1 orig 2`.
- `luce.sh MODES FIX NTOK REPS` runs Lucebox (modes `native,stress,syncsan,adversarial,control`; `FIX` `none`, `acquire`,
  `reset` or `both`). The RTX 5090 rows in [docs/lucebox.md](../../docs/lucebox.md):
  `python reproduce/daytona/run.py --gpu RTX-5090 --image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel reproduce/daytona/luce.sh native,stress,syncsan,adversarial,control none 16 20000`.
- `zoo.sh MODES FIX NTOK REPS` runs megakernel-zoo's Qwen3-8B engine the same way (H100).
- `concurrent_test.sh` runs the two-stream regression test and the known-answer cases.
