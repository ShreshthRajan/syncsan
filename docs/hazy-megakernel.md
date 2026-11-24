# Races in Hazy Research's Llama-1B megakernel

Every handoff of activations between SMs in the
[low-latency Llama megakernel](https://github.com/HazyResearch/Megakernels/tree/7309cec/demos/low-latency-llama)
races under the PTX memory model, and its output depends on how those races resolve. Results are for commit 7309cec,
Llama-3.2-1B-Instruct, on an H100 80GB and a B200.

[HazyResearch/Megakernels#15](https://github.com/HazyResearch/Megakernels/issues/15) (Tanish Gudise, Oct 27, 2026)
found this pattern first, by hand, on the up/gate to down-projection handoff (`silu_out`), and measured it on
hardware: 26 of 1,000 uninstrumented H100 generations had a token more than one logit off, against 0 of 1,000 with a
release/acquire patch. This page covers the other handoffs with the same pattern.

## The handoff

A producer's storer lane writes its output with a TMA store, waits for the store with `tma::store_async_wait()`, and
increments a counter with `atomicAdd`. A consumer's lane spins on the counter with a `volatile` load and then reads the
data with a TMA load, a `cp.async` or a plain `ld.global.v4`. The `atomicAdd` is relaxed, the spin load is not followed
by a fence, and the compiled kernel contains no `MEMBAR`, so nothing orders the consumer's reads after the TMA engine's
writes. syncsan reports 25 racing instruction pairs in a 4-token run on H100 (versions before 0.6.0 stored only the
first 7, the same on H100 and B200, because the record buffer filled with their repeats), for example:

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

In adversarial mode each racing load returns the value its word held before the last write, which the PTX model
allows because no happens-before edge orders the load after that write. `attn_out` is rewritten every layer, so for
the o_proj read the older value is the previous layer's attention output. 8 tokens on an H100:

| Build | Mode | Loads given an older value | Output |
|---|---|---|---|
| original | native | | "Why did the cookie go to the" |
| original | adversarial | 13.2M | " insideuccarnochumptlink circular" |
| original | control (same redirection, current values) | 13.2M | "Why did the cookie go to the" |
| release/acquire counters | adversarial | 0 | "Why did the cookie go to the" |

The wrong text differs between runs, because which loads race depends on timing; an earlier run printed
"wedchuntghtunday造成uarios". The TMA and `cp.async.cg` reads go through L2, which is coherent across SMs, so on current hardware they most likely
see the new data. The `ld.global.v4` can be served from the reading SM's L1, which is not coherent, if that SM still
holds lines of `attn_out` from an earlier layer of the same launch. No wrong output has been observed natively.

A native run with no instrumentation, using #15's method, fills `silu_out` with 1e4 before every decode step and counts
generations outside the two token sequences both builds produce most often: 43 of 300 for the original against 6 of
300 for the release/acquire build (Fisher exact p = 1.7e-8). The same test on `attn_out` gave 23 against 11 (p = 0.05),
which does not separate the builds at this sample size.

## The fix

Release/acquire on the counters removes every report:

| Build | Pairs reported | 128 tokens, average of 10, one H100 run |
|---|---|---|
| original | 7 (counted before 0.6.0) | 127.0 ms |
| `fence.acq_rel.gpu` after each `store_async_wait()` and after each counter wait | 0 | 146.7 ms (+15.5%) |
| counters as `red.release.gpu.global.add`, polls as `ld.acquire.gpu` | 0 | 138.0 ms (+8.7%) |

[reproduce/patch_hazy.py](../reproduce/patch_hazy.py) produces both fixed builds (`fences` and `acqrel`), and
[reproduce/modal_hazy.py](../reproduce/modal_hazy.py) reruns the table. The issue upstream is
[HazyResearch/Megakernels#16](https://github.com/HazyResearch/Megakernels/issues/16).
