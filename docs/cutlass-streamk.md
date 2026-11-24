# CUTLASS Stream-K: a known bug, found and cleared

CUTLASS issue [#2027](https://github.com/NVIDIA/cutlass/issues/2027) reported nondeterministic results from the SM90
Stream-K GEMM, fixed in CUTLASS 3.7.0. syncsan reports the bug's instructions in 3.6.0 and nothing in 3.7.0. Results are
for the issue's reproducer (fp32, 1024 x 1024 x K, Stream-K forced, checked against CUTLASS's reference GEMM) on an
H100.

| Version | Wrong results natively | Racing pairs |
|---|---|---|
| 3.6.0, K = 4096 | 50 of 50 runs | 110 to 114 (varies by run), one pattern |
| 3.7.0, K = 4096 | 0 of 140 runs (all K tested) | 0 |
| either, K = 1024 (no tile gets three splits) | 0 | 0 |

All pairs in 3.6.0 are the middle split's `atomicAdd` in `BlockStripedReduce::reduce`
(`block_striped.h:229` < `sm90_tile_scheduler_stream_k.hpp:504`) against the final split's read in `load_add`
(`block_striped.h:181` < `stream_k.hpp:530`), which follows `wait_lt(..., 1)` at line 525. Adversarial mode adds nothing
here, because the buggy version already fails natively. [reproduce/modal_cutlass.py](../reproduce/modal_cutlass.py)
reruns the table.
