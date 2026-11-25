# Races in Hazy Research's Llama-1B megakernel

Every handoff of activations between SMs in the [low-latency Llama
megakernel](https://github.com/HazyResearch/Megakernels/tree/7309cec/demos/low-latency-llama) races under the PTX memory
model, and its output depends on how those races resolve. Results are for commit 7309cec, Llama-3.2-1B-Instruct, on an
H100 80GB and a B200.

[HazyResearch/Megakernels#15](https://github.com/HazyResearch/Megakernels/issues/15) (Tanish Gudise, Oct 27, 2026) found
this pattern first, by hand, on the up/gate to down-projection handoff (`silu_out`), and measured it on hardware: 26 of
1,000 uninstrumented H100 generations had a token more than one logit off, against 0 of 1,000 with a release/acquire
patch. This page covers the other handoffs with the same pattern.

## The handoff

A producer's storer lane writes its output with a TMA store, waits for the store with `tma::store_async_wait()`, and
increments a counter with `atomicAdd`. A consumer's lane spins on the counter with a `volatile` load and then reads the
data with a TMA load, a `cp.async` or a plain `ld.global.v4`. The `atomicAdd` is relaxed, the spin load is not followed
by a fence, and the compiled kernel contains no `MEMBAR`, so nothing orders the consumer's reads after the TMA engine's
writes. syncsan 0.9 reports 26 racing instruction pairs in an 8-token run on H100: 25 in global memory, for example the
ones below, and one write-write pair in shared memory (a `STS` against a TMA load into the same buffer) that is also
reported in the fixed build and comes from a second race (below):

| Write | Read |
|---|---|
| TMA store of Q, K and V (`rms_matvec_rope_append.cu:130`, `:148`) | TMA loads of the K and V cache (`attention_partial.cu:374`, `:379`) and `cp.async.cg` of Q (`attention_partial.cu:271`) |
| TMA store of `attn_out` (`attention_partial.cu:577`) | o_proj's `ld.global.v4` of its input (`matvec_adds.cu:145`) |
| TMA store of `silu_out` (`upgate.cu:109`), the handoff of #15 | the down-projection's `ld.global.v4` (`matvec_adds.cu:145`) |
| TMA reduce into the residual `hidden_states` (`matvec_adds.cu:78`) | the next op's RMS norm (`upgate.cu:162`, `rms_matvec_rope_append.cu:222`, `rms_lm_head.cu:109`) |

The authors left `fence.acq_rel.gpu` commented out at five of these handoffs, with the note "possible we need sc here
but I don't think so". In `attention_partial.cu` the commented fence sits before `tma::store_async_wait()`. A fence that
runs before the TMA store completes does not order that store, so enabling the fence where it is leaves the `attn_out`
handoff unordered, and syncsan still reports it.

## What the output depends on

In adversarial mode each racing load returns the value its word held before the last write. Hazy's handoffs are TMA box
stores read by 16-byte loads, so every substituted word was last written by an access of another width. PTX models
vector and bulk accesses as accesses of their element type, which the machine code does not show, so syncsan cannot tell
whether these are uniform-size races (where the model allows the stale value) or mixed-size ones (where it defines no
outcome). The wrong output below shows that the races lie on the data path to the output; the evidence that they matter
on hardware is #15's native measurement. `attn_out` is rewritten every layer, so for the o_proj read the older value is
the previous layer's attention output. 8 tokens on an H100:

| Build | Mode | Loads given an older value | Output |
|---|---|---|---|
| original | native | | "Why did the cookie go to the" |
| original | adversarial | 14.9M | "iteriteriter abstractجرineronde" |
| original | control (same redirection, current values) | 14.9M | "Why did the cookie go to the" |
| release/acquire counters | adversarial | 0 | "Why did the cookie go to the" |

The wrong text differs between runs, because which loads race depends on timing; earlier runs printed "
insideuccarnochumptlink circular" and "wedchuntghtunday造成uarios". The TMA and `cp.async.cg` reads go through L2, which
is coherent across SMs, and are less exposed on current hardware. The `ld.global.v4` can be served from the reading SM's
L1, which is not coherent, if that SM still holds lines of `attn_out` from an earlier layer of the same launch. No wrong
output has been observed natively.

A native run with no instrumentation, using #15's method, fills `silu_out` with 1e4 before every decode step and counts
generations outside the two token sequences both builds produce most often: 43 of 300 for the original against 6 of 300
for the release/acquire build (Fisher exact p = 1.7e-8). The same test on `attn_out` gave 23 against 11 (p = 0.05),
which does not separate the builds at this sample size.

## On hardware

A native test with no instrumentation, following #15's method with a stronger canary: before every decode step,
`reproduce/hazy/canary.py` fills one buffer with NaN. Every op that reads the buffer runs after the op that writes it,
so a correct kernel never reads the NaN; one stale read anywhere turns the logits into NaN. 1,000 generations of 64
tokens per build, H100 (`results/hazy/canary_nan_20261124`):

| Buffer | Original | Release/acquire build |
|---|---|---|
| `silu_out` (up/gate to down-projection, #15's handoff) | NaN in 1,000 of 1,000 generations | 0 of 1,000 |
| `attn_out` (attention to o_proj) | 0 of 1,000 | 0 of 1,000 |

So on an H100 the down-projection reads `silu_out` before the up/gate op's writes are visible in every generation,
nearly always on the first decode step; #15's 2.6% counts the generations in which the stale value also changed a token.
The `attn_out` handoff, read through TMA and `cp.async`, did not fail.
[reproduce/daytona/hazy_witness.sh](../../reproduce/daytona/hazy_witness.sh) reruns the table.

## A second race: the QOL page

With `skip_attn_reduction` set, which is the default (`scheduler.py:40`), the attention storer waits on `O_arrived`,
stores `O`, and finishes the QOL page (`attention_partial.cu:647`) without waiting on `L_arrived`. The consumer stores
the LSE into `L_smem` in that page after arriving on `O_arrived` (`:543`). Nothing orders that store before the page is
handed to the next instruction, whose loader TMA-loads into it; syncsan reports the consumer's `STS` against that TMA
load (`upgate.cu:48` via `rms_lm_head.cu:99`). In the skip path nothing reads `L_smem`, so skipping the store, or
waiting on `L_arrived` before finishing the page, removes the race. No wrong output has been observed from it. Reported
as [HazyResearch/Megakernels#17](https://github.com/HazyResearch/Megakernels/issues/17).

## The fix

Release/acquire on the counters removes every report:

128 tokens, average of 10 iterations; each fixed build is compared with the original in the same H100 container, one row
per container:

| Build | Pairs reported | Original | Fixed build |
|---|---|---|---|
| `fence.acq_rel.gpu` after each `store_async_wait()` and after each counter wait | 0 | 126.7, 126.8, 126.9 ms | 146.3, 146.5, 146.3 ms (+15%) |
| counters as `red.release.gpu.global.add`, polls as `ld.acquire.gpu` | 0 | 124.7, 126.9 ms | 136.2, 138.2 ms (+9.2%, +9.0%) |

(A third release/acquire timing, 126.7 ms, came from a run where the patch failed to build and the original ran instead;
syncsan still reported the original's pairs there, so it is left out.)

[reproduce/hazy/patch.py](../../reproduce/hazy/patch.py) produces both fixed builds (`fences` and `acqrel`), and
[reproduce/modal/hazy.py](../../reproduce/modal/hazy.py) reruns the table. The issue upstream is
[HazyResearch/Megakernels#16](https://github.com/HazyResearch/Megakernels/issues/16).
