# Reproducing the results

Each result reruns on a rented GPU through [Modal](https://modal.com) (`pip install modal && modal setup`) or
[Daytona](https://www.daytona.io) (`DAYTONA_API_KEY=... python reproduce/daytona/run.py [--gpu H100|RTX-5090] SCRIPT.sh ARGS`).
Daytona's `run.py` uploads this repository, runs the job script, and saves its output under `results/`.

| Result | Modal | Daytona | Write-up |
|---|---|---|---|
| Known-answer tests | `modal run reproduce/modal/cases.py` | `tests_job.sh` | [tests/](../tests/) |
| Hazy Llama-1B | `modal run reproduce/modal/hazy.py` | `hazy_only.sh`, `hazy_witness.sh` (NaN canary) | [hazy-megakernel](../docs/findings/hazy-megakernel.md) |
| AutoMegaKernel | `modal run reproduce/modal/amk.py` | `hazy_amk.sh`, `amk_canary.sh` | [automegakernel](../docs/findings/automegakernel.md) |
| SuperLU_DIST | `modal run reproduce/modal/superlu.py` | `superlu.sh` | [superlu](../docs/findings/superlu.md) |
| megakernel-zoo | `modal run reproduce/modal/zoo.py [--fixed acquire]` | `zoo.sh MODES FIX NTOK REPS` | [megakernel-zoo](../docs/findings/megakernel-zoo.md) |
| Lucebox | `modal run reproduce/modal/luce.py [--fixed both]` | `luce.sh MODES FIX NTOK REPS`, `luce_canary.sh` | [lucebox](../docs/findings/lucebox.md) |
| Mirage MPK | | `mirage.sh` | [mirage](../docs/findings/mirage.md) |
| CUTLASS Stream-K | `modal run reproduce/modal/cutlass.py` | | [cutlass-streamk](../docs/findings/cutlass-streamk.md) |
| Magni VM | | `magni.sh` | patches in [magni/](magni/) |
| cuHadron | | `cuhadron.sh` | [tests/cuhadron](../tests/cuhadron/) |

The per-project folders hold the patches and drivers the jobs use: `hazy/patch.py DIR fences|acqrel` writes the two
Hazy fixes, `*/canary.py` are the NaN canary tests, and `zoo/`, `luce/` drive each engine's decode. Jobs that need
torch use the `pytorch/pytorch` images (`--image pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel`), since Daytona's lower
tiers cannot reach `pypi.nvidia.com`. `compile_check.sh` builds everything on a CPU sandbox (`--gpu none`).
