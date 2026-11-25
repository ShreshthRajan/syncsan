# A missing acquire in AutoMegaKernel

[AutoMegaKernel](https://github.com/RightNow-AI/AutoMegaKernel) runs a model's forward pass as one persistent kernel
whose instructions wait on counters. Its consumers wait with a relaxed atomic, so the PTX memory model does not order
their reads after the producers' writes. Under adversarial mode the toy model's output stops matching the eager
reference, and a one-line change to the wait fixes it. Results are for commit 884534c on an H100, with the toy model and
`--verify`, which compares the megakernel's tokens with an eager PyTorch run of the same model.

## The handoff

`amk_signal` orders a producer's outputs before its counter increment with `__threadfence()` and `atomicAdd`, which is a
release. `amk_wait_all` spins on `amk_acquire_load`, which is `atomicAdd(c, 0u)`, and then calls `__syncthreads()`. The
comment on `amk_acquire_load` says the RMW "returns the current value with .gpu acquire semantics", but PTX treats an
`atom` without a `.sem` qualifier as relaxed, and no fence follows the spin, so the consumer side has no acquire.
syncsan reports 24 racing instruction pairs, all in `amk_megakernel`, between producers' stores and consumers' loads.

## What the output depends on

| Build | Mode | Racing pairs | Loads given an older value | `matches_eager` |
|---|---|---|---|---|
| original | native | | | true |
| original | syncsan | 24 | | true |
| original | adversarial | 24 | 54,144 | false (generated tokens are all 0) |
| original | control (same mechanism, current values) | 24 | | true |
| `amk_acquire_load` as `ld.acquire.gpu` | adversarial | 0 | 0 | true |

The fix replaces the body of `amk_acquire_load` with `asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) :
"l"(c) : "memory")`; `atom.acquire.gpu` would also work. No wrong output has been observed natively. The table is from
syncsan 0.9 with shared-memory checking on; 0.5 gave the same counts with it off. All 54,144 substituted words were last
written by an access of the same width; syncsan left 5 instructions unchecked because of the sm_90 NVBit hazard and
warns that an unseen write there could make a value one the model forbids.
[reproduce/daytona/hazy_amk.sh](../reproduce/daytona/hazy_amk.sh) reruns the table on Daytona,
[reproduce/modal_amk.py](../reproduce/modal_amk.py) on Modal.

On hardware: with the activation arena filled with NaN before every launch (`reproduce/amk_canary.py`), 0 of 200
generations of the original and 0 of 200 of the fixed build read it on an H100, and a control build whose waits return at
once read it in 20 of 20 (`results/amk_canary_20261124`).
