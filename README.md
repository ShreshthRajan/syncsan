# syncsan

A race detector for GPU megakernels. It runs your unmodified CUDA program and reports every pair of memory accesses
that nothing orders: no barrier, no fence, no release/acquire, no async-copy wait. Then it can show you what each race
does to the output.

Megakernels pass data between thread blocks through flags and counters in global memory. A missing fence there
usually works on today's GPUs, so tests pass, but the PTX memory model allows the consumer to read stale data.
syncsan finds those handoffs. Its adversarial mode feeds each unordered load the stale value the model permits, so
a race shows up as wrong output.

## Results

| Project | What syncsan found | Status |
|---|---|---|
| [SuperLU_DIST](https://github.com/xiaoyeli/superlu_dist) | A fix restored the consumer's acquire but not the producer's release. Adversarial solve error 0.7. | [Confirmed and fixed upstream](https://github.com/xiaoyeli/superlu_dist/issues/261) |
| [Hazy Research Llama-1B megakernel](https://github.com/HazyResearch/Megakernels) | 25 racing pairs in the activation handoffs. On a real H100, `silu_out` is read stale in 1,000 of 1,000 generations. | [Reported](https://github.com/HazyResearch/Megakernels/issues/16) |
| [Magni VM](https://github.com/Bobliuuu/mvm) | An admission race that 3,200 passing test runs missed. | [Fix submitted](https://github.com/Bobliuuu/mvm/pull/97) |
| [AutoMegaKernel](https://github.com/RightNow-AI/AutoMegaKernel) | `atomicAdd(c, 0)` used as an acquire. Adversarial mode fails `--verify`. | |
| [megakernel-zoo](https://github.com/dangxingyu/megakernel-zoo), [Lucebox](https://github.com/Luce-Org/lucebox) | Spin-waits with no acquire. Adversarial output is wrong text. | |
| [Mirage MPK](https://github.com/mirage-project/mirage) | A shared-memory race in the argmax reduction. | [Reported](https://github.com/mirage-project/mirage/issues/843) |
| [CUTLASS](https://github.com/NVIDIA/cutlass) 3.6.0 Stream-K | The known race from issue #2027, and nothing in the fixed 3.7.0. | Already fixed |

Each finding has a write-up in [docs/findings/](docs/findings/) and a script in [reproduce/](reproduce/). The
summary of every run is in [results/](results/).

On the Hazy megakernel (H100, 8 tokens):

| Build | Mode | Output |
|---|---|---|
| original | native | "Why did the cookie go to the" |
| original | adversarial | "iteriteriter abstractجرineronde" |
| original | control (same loads, current values) | "Why did the cookie go to the" |
| release/acquire counters | adversarial | "Why did the cookie go to the" |

## Install

You need Linux on x86-64, an NVIDIA GPU of compute capability 8.0 or newer (tested on H100, B200 and RTX 5090), CUDA
12 with `nvcc`, and CMake 3.29 or newer.

```
pip install git+https://github.com/ShreshthRajan/syncsan
syncsan build --arch 90     # once; 90 = H100, 100 = B200, 120 = RTX 50xx, 80 = A100
```

`syncsan build` downloads [NVBit](https://github.com/NVlabs/NVBit), NVIDIA's binary instrumentation library.

## Usage

```
syncsan run -- python generate.py                         # check every kernel
syncsan run --kernel megakernel -- ./app                  # check kernels whose name contains "megakernel"
syncsan run --exitcode 1 -- ./kernel_tests                # fail CI when a race is found
SYNCSAN_ADVERSARIAL=1 syncsan run -- python generate.py   # give racing loads stale values
```

Compile with `-lineinfo` so reports name source lines:

```
syncsan: RACE (write then read) at 0x2ba22b84c900
  block 18 copy engine (issued by warp 17): UTMASTG.4D ...  [tma.cuh:74 < attention_partial.cu:577 < storer.cuh:6]
  block 55 warp 3: LDG.E.128 R8, ...                        [util.cuh:158 < matvec_adds.cu:145 < consumer.cuh:6]
  no barrier, release/acquire pair, async-copy wait or atomic orders these two accesses
```

`--sample k` checks one in `k` plain memory accesses for speed. `syncsan run --help` lists the rest, and
[examples/github-actions.yml](examples/github-actions.yml) runs syncsan in CI.

## What it checks

- Global memory between thread blocks: flags, counters, locks, fences, release/acquire.
- Shared memory between warps: barriers, named barriers, mbarriers.
- Hopper and Blackwell async copies: TMA, `cp.async.bulk`, `cp.async`, `fence.proxy.async`.

It does not check races between lanes of one warp, distributed shared memory, or ordering across kernels and
streams. A Llama-1B decode runs about 4x slower under it; kernels that do little besides memory access run far
slower, which `--sample` offsets. [docs/how-it-works.md](docs/how-it-works.md) explains the model and its limits,
and [docs/prior-work.md](docs/prior-work.md) compares it with Barracuda, iGUARD, SuperCollider and compute-sanitizer.

## Tests

`tests/run_tests.sh sm_90` runs 51 known-answer cases on a local GPU, and `modal run reproduce/modal/cases.py` runs
them on a rented one. [tests/cuhadron/](tests/cuhadron/) adds six cross-block synchronization tests to NVIDIA's
cuHadron race benchmark.

## Citing and license

See [CITATION.cff](CITATION.cff). Apache 2.0.
