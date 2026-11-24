# Two grid-barrier races in Lucebox's Qwen3.5-0.8B megakernel

[Lucebox](https://github.com/Luce-Org/lucebox)'s decode megakernel (`optimizations/megakernel`, a70ab3c, BF16, sm_90)
runs a decode step of Qwen3.5-0.8B in one kernel, `decode_kernel`, and separates its phases with a software grid
barrier, `AtomicGridSync::sync` (`kernel.cu:144-162`). The barrier has two races under the PTX memory model:

- **No acquire on the waiter.** The arrivers release correctly, with a fence and then `atomicAdd(counter)` (`:151`).
  The waiters spin on `volatile *generation` (`:157`) with no fence after the spin, so their later reads are not ordered
  after the other blocks' writes.
- **The counter reset.** The last arriver resets the counter with a plain store, `*counter = 0` (`:153`). Its fence
  comes after that store, so the reset is not ordered after the other blocks' `atomicAdd`. The race is write-write, so
  adversarial mode cannot show what it does. RESOLVE reported the same reset race.

## Results

H100, 8 greedy tokens of "The capital of France is":

| Build | Mode | Racing pairs | Output |
|---|---|---|---|
| a70ab3c | native | | " Paris.\nThe capital of France is", the same as Hugging Face; bitwise repeatable |
| a70ab3c | syncsan | 107 | |
| a70ab3c | adversarial | | "_x offID干的目nung)Ventas" (another run: " costftenVEDingssienneszatccion的那样"), wrong at all 8 steps |
| a70ab3c | control | | bitwise equal to native |
| fence after the spin | adversarial | 9 (the reset, one per barrier site) | bitwise equal to native |
| fence after the spin and `atomicExch` reset | adversarial | 0 | bitwise equal to native |

About 98 pairs come from the missing acquire: almost every matvec, DeltaNet and attention output is read by another
block after a barrier. Examples are `:335` against `:490`, `:502`, `:514` and `:525`; `:582` against `:593`; `:734`
and `:753` against `:764`; `:372` against `:609` and `:779`; and the in-place RoPE writes at `:522` and `:525` against
the q stores.

## Fix

Add `fence.acq_rel.gpu` (the file's `GPU_MEMORY_FENCE()`) after the generation spin, and reset the counter with
`atomicExch(counter, 0u)`. [reproduce/luce/patch_sync.py](../reproduce/luce/patch_sync.py) makes the change (`acquire`,
`reset` or `both`).

## On hardware

2,000 native decodes of 16 tokens on an H100 gave the same tokens and bitwise-identical hidden states every time, in the
original and the fixed build, so the race has not shown up on hardware. A step took 1,235 µs in the original and 1,254 µs
with the fix, timed from Python, one run each on separate H100s, so the fix costs nothing measurable.
`modal run reproduce/modal_luce.py --modes stress --ntok 16` reruns it (add `--fixed both`).

The same holds on a GeForce RTX 5090 (Blackwell, sm_120), the class of GPU Lucebox is written for. 20,000 native decodes
of 16 tokens were all identical. syncsan reports 46 racing pairs there (the 5090 runs fewer blocks than the H100).
Adversarial mode gives wrong tokens at 16 of 16 steps ("ываютτεύtret otrasikan…"), the control is bitwise equal to
native, and with both fixes syncsan reports 0 pairs and the adversarial output is bitwise equal to native.

## Run settings

`SYNCSAN_SHARED=0`, `SYNCSAN_KERNEL=decode_kernel` and `SYNCSAN_SKIP_OPS==LDG.E.NA.128`, which leaves the weight loads
(`load_128bit`) unchecked. Weights are never written during decode, so skipping them can hide no race in the
activations. With the weight loads checked, one token took 1,099 s and the shadow table overflowed with weight words.
`lm_head_kernel` was not checked; its waiter does have `__threadfence()` after the spin.
[reproduce/modal_luce.py](../reproduce/modal_luce.py) reruns the table (`--fixed acquire` and `--fixed both` for the last two rows).
