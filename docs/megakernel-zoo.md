# A missing acquire in megakernel-zoo's Qwen3-8B engine

[megakernel-zoo](https://github.com/dangxingyu/megakernel-zoo)'s `qwen3-8b_h100/offline-a` engine (95e315c) runs a
decode step of Qwen3-8B as one persistent kernel, `k_step`. Blocks hand work to each other through a grid counter,
`P.sync`. The producers publish correctly: `__threadfence()`, then `atomicAdd(P.sync)` (`mk.cu:1712`, `:1722`). The
waiters spin on the counter with a `volatile` load (`grid_barrier` at `:971`, the coordinator loop at `:1775`) and then
call `__syncthreads()`. No fence follows the spin, so the waiter has no acquire, and the PTX memory model does not order
its reads after the producers' writes.

## Results

H100, 4 greedy tokens of "The capital of France is", weights as stored in the checkpoint (BF16):

| Build | Mode | Racing pairs | Output |
|---|---|---|---|
| 95e315c | native | |  Paris. The capital", the same as Hugging Face BF16 greedy for these 4 tokens (at 16 tokens the engine departs from it at token 13); two runs bitwise equal |
| 95e315c | syncsan | 9 | |
| 95e315c | adversarial | | "eses:es" (another run: "::s:"); logits differ by up to 23.6 to 27.9, at all 4 positions |
| 95e315c | control | | bitwise equal to native |
| `__threadfence()` after each spin | adversarial | 1 (the hint below) | bitwise equal to native |

Eight pairs come from the missing acquire:

- the epilogue's q, k and v stores (`:952`, reached from `:1710`) against attention's `__ldcg` loads (`:2415`);
- the `ssq` partial sums (`:938`, and `:694` in `op_embed`) against `rms_inv` (`:985`), in both directions across the
  barrier.

`__ldcg` compiles to a strong relaxed load, so it may legally return the older value when nothing orders it after the
store.

The ninth pair is a deliberate race. `att_done` is written with a plain store (`:2510`) and polled with `__ldcg`
(`:3076`). The source calls it "a hint only: a stale or late stamp changes the timing, not the result", and the poll is
bounded.

## Fix

Add `__threadfence()` after each spin on `P.sync`: in `grid_barrier`, in the coordinator warp's barrier and in the
`QKV_NOWAIT` wait. [reproduce/zoo/patch_sync.py](../reproduce/zoo/patch_sync.py) makes the change. `fence.acq_rel.gpu`
or an `ld.acquire.gpu` poll would also work.

## On hardware

2,000 native decodes of 16 tokens on an H100 gave the same tokens and bitwise-identical logits every time, in the
original and the fixed build, so the race has not shown up on hardware. A step took 5,143 µs in the original and 5,063
µs with the fix, timed from Python, one run each on separate H100s, which is within the spread of one run. `modal run
reproduce/modal_zoo.py --modes stress --ntok 16` reruns it (add `--fixed acquire`).

## Run settings

`SYNCSAN_SHARED=0` and `SYNCSAN_SKIP_OPS==LDG.E.LU.128`, which leaves the engine's weight loads (`WLOAD`) unchecked.
Weights are never written during a step, so skipping them can hide no race in the activations. With the weight loads
checked, every mode ran past the 40-minute limit: the shadow table fills with weight words. Every other load and store
is checked. [reproduce/modal_zoo.py](../reproduce/modal_zoo.py) reruns the table (`--fixed acquire` for the last row).
