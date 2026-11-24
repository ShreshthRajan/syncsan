# syncsan

syncsan finds data races in GPU megakernels and shows what they do to the output. It runs unmodified CUDA binaries
under [NVBit](https://github.com/NVlabs/NVBit), tracks happens-before as the PTX memory model defines it, and reports
every pair of accesses that no barrier, release/acquire pair, fence or async-copy wait orders. In adversarial mode,
each load that is not ordered after the last write to its word returns the value that word held before that write,
which the PTX model still allows it to return. A race then becomes a wrong output on a real GPU.

## Example: Hazy Research's Llama-1B megakernel

The [low-latency Llama megakernel](https://github.com/HazyResearch/Megakernels) hands activations between SMs with a
TMA store, a relaxed `atomicAdd` on a counter, a `volatile` spin and a read, with no gpu-scope fence, so the PTX model
does not order the read after the store. syncsan reports 25 racing instruction pairs in one run, covering these
handoff: Q, K and V into attention, attention output into o_proj, `silu_out` into the down-projection, and the residual
stream into the next RMS norm. On
an H100, 8 tokens:

| Build | Mode | Loads given an older value | Output |
|---|---|---|---|
| original | native | | "Why did the cookie go to the" |
| original | adversarial | 13.2M | " insideuccarnochumptlink circular" |
| original | control (same redirection, current values) | 13.2M | "Why did the cookie go to the" |
| release/acquire counters | adversarial | 0 | "Why did the cookie go to the" |

The wrong text changes from run to run (another run printed "wedchuntghtunday造成uarios"); it is wrong in every run.
The control run redirects the same loads but hands them their current values, which rules out the redirection itself
as the cause. Making the counters `red.release.gpu` and `ld.acquire.gpu` removes every report and
costs about 9% in latency. The analysis and the fix are in [docs/hazy-megakernel.md](docs/hazy-megakernel.md) and in
[HazyResearch/Megakernels#16](https://github.com/HazyResearch/Megakernels/issues/16). The same missing release/acquire
was found earlier by hand in [#15](https://github.com/HazyResearch/Megakernels/issues/15), on the up/gate to
down-projection handoff, where it changes tokens in 2.6% of generations on real H100s.

## Example: AutoMegaKernel

[AutoMegaKernel](https://github.com/RightNow-AI/AutoMegaKernel)'s consumers wait for their inputs with
`atomicAdd(c, 0)`, which a source comment describes as an acquire. PTX treats it as relaxed, and no fence follows the
spin, so syncsan reports 24 racing pairs. Its `--verify` flag compares the megakernel's tokens with an eager PyTorch run
(toy model, H100):

| Build | Mode | Loads given an older value | `--verify` |
|---|---|---|---|
| original | native | | pass |
| original | adversarial | 54,144 | fail |
| original | control | | pass |
| wait as `ld.acquire.gpu` | adversarial | 0 | pass |

Details in [docs/automegakernel.md](docs/automegakernel.md).

## Install

syncsan needs Linux on x86-64, an NVIDIA GPU of compute capability 8.0 or newer (tested on A100, H100 and B200), CUDA
12 with `nvcc`, and CMake 3.29 or newer.

```
pip install git+https://github.com/ShreshthRajan/syncsan
syncsan build --arch 90        # once: downloads NVBit 1.8.2 and compiles the tool (90 = H100, 100 = B200, 80 = A100)
```

## Usage

```
syncsan run -- python generate.py                         # check every kernel
syncsan run --kernel megakernel -- ./app                  # check only kernels whose name contains "megakernel"
syncsan run --exitcode 1 -- ./kernel_tests                # exit with 1 when a race is found (for CI)
SYNCSAN_ADVERSARIAL=1 syncsan run -- python generate.py   # adversarial mode
```

Build the program with `-lineinfo` so that reports name source lines. A report gives both accesses with their SASS,
their warp or copy engine, and the inlined call chain down to the operation that ran:

```
syncsan: RACE (write then read) at 0x2ba22b84c900
  block 18 copy engine (issued by warp 17): UTMASTG.4D [UR8], [UR16], desc[UR14]  [mk+0x1d1d0 tma.cuh:74 < tma.cuh:248 < attention_partial.cu:577 < attention_partial.cu:612 < storer.cuh:6]
  block 55 warp 3: @!P3 LDG.E.128 R8, desc[UR6][R8.64]  [mk+0x21e40 util.cuh:158 < global_to_shared.cuh:27 < matvec_adds.cu:145 < consumer.cuh:6]
  no barrier, release/acquire pair, async-copy wait or atomic orders these two accesses
```

`syncsan run --help` lists the options. The most useful ones: `--sample k` checks one plain load or store site in `k`
(synchronization is always checked, so sampling can miss a race but cannot invent one), and `SYNCSAN_SHARED=0` turns
off shared-memory checking. [examples/github-actions.yml](examples/github-actions.yml) runs syncsan in CI on a
self-hosted GPU runner.

## What it checks

syncsan checks global memory between thread blocks (flags, counters and locks built from relaxed or strong accesses,
fences, release/acquire and `fence.sc`), shared memory between warps (full and named barriers, mbarriers), and the
Hopper and Blackwell copy engine. TMA tensor copies, `cp.async.bulk` and `cp.async` are modeled as accesses by the
block's copy engine that become visible to a warp once it waits on them, with TMA coordinates decoded from the tensor
maps the program creates. [docs/how-it-works.md](docs/how-it-works.md) describes the model and its limits.

On NVIDIA's [cuHadron](https://github.com/NVlabs/cuHadron) race benchmark (SM90), syncsan detects 9 of the 21
single-GPU races with no false positives on the fixed builds. compute-sanitizer from CUDA 12.4 detects 6 others, all in
shared memory. cuHadron has no test of synchronization between thread blocks, so [cuhadron/](cuhadron/) adds six;
syncsan detects all six and compute-sanitizer none. syncsan does not check races between lanes of one warp,
distributed shared memory, or ordering between kernels, streams and host copies.

## Other results

**SuperLU_DIST.** The GPU triangular solve's fix for a dropped fence (commit 587f38f) restored the consumer's acquire
but left the producer's release out: syncsan still reports 4 racing pairs, adversarial mode gives a wrong solution
(error about 0.7), and one `__threadfence()` before the producer's `atomicSub` clears both. The original bug (4755a5e) is
flagged at the same lines and fails natively. Details in [docs/superlu.md](docs/superlu.md).

**CUTLASS Stream-K.** On CUTLASS issue #2027, syncsan reports the known race between the middle split's `atomicAdd`
and the final split's read in 3.6.0, which fails natively, and nothing in the fixed 3.7.0. Details in
[docs/cutlass-streamk.md](docs/cutlass-streamk.md).


syncsan found an admission race in the [Magni VM](https://github.com/Bobliuuu/mvm) megakernel in one run, after 3,200
test runs had passed; the fix is submitted as [Bobliuuu/mvm#97](https://github.com/Bobliuuu/mvm/pull/97). CUB's
decoupled look-back scan, select and radix-histogram kernels check clean. In
[open-flapping-airplanes](https://github.com/Bobliuuu/open-flapping-airplanes), which trains and serves a model and runs
Doom in one kernel, the training handoffs are ordered by `__threadfence()`; the reported races are inference reading the
weights while training updates them, which the demo does on purpose, a plain-store reset of a task counter, and the
displayed loss.

## Cost

Most of the slowdown comes from NVBit's call at each instrumented access, not from the checks. Hazy's 4-token script
takes 54 s under syncsan with every check on, against 14 s natively. A training megakernel whose 54 µs step is almost
all global accesses runs about 9,000 times slower; `--sample` is the remedy there.

## Prior work

Adversarial memory comes from Jumble (Flanagan and Freund, PLDI 2010), which returned stale values allowed by the Java
memory model to show that a race was harmful. Bugaroo (ISSRE 2018) brought stale values to GPUs through a recompiled
binary and does not limit them to values the model allows. Happens-before race detection on GPUs goes back to iGUARD
(SOSP 2021); Triton's GSan does it for global memory in Triton programs. syncsan does both on any CUDA binary,
including CUDA C++, ThunderKittens, CUTLASS and inline PTX, with fences and Hopper async copies modeled.
[docs/prior-work.md](docs/prior-work.md) covers the rest.

## Reproducing the results

[reproduce/](reproduce/) runs the known-answer tests and the Hazy experiments on a rented H100 through Modal.
`tests/run_tests.sh` runs the 37 known-answer cases on a local GPU.

## Citing

See [CITATION.cff](CITATION.cff).

## License

Apache 2.0.
