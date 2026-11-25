# syncsan

syncsan finds data races in GPU megakernels. It runs your CUDA program unchanged under
[NVBit](https://github.com/NVlabs/NVBit) and reports every pair of memory accesses that no barrier, fence,
release/acquire or async-copy wait puts in order. With `SYNCSAN_ADVERSARIAL=1` it also hands each racy load the stale
value the PTX memory model allows, so you can see what the race does to your output.

Megakernels need this more than most code. They run a whole model in one launch and pass activations between SMs
through counters in global memory. A missing fence in one of those handoffs rarely fails on today's GPUs, so it
survives testing.

## Quick start

You need Linux, an NVIDIA GPU from the A100 onward, CUDA 12 and CMake 3.29.

```
pip install git+https://github.com/ShreshthRajan/syncsan
syncsan build --arch 90          # once; use 100 for B200, 120 for RTX 50xx, 80 for A100
syncsan run -- python generate.py
```

Build your kernels with `-lineinfo` so reports point at source lines. This report comes from Hazy Research's Llama
megakernel, where a TMA store in attention races with the matrix-vector load that reads it on another SM:

```
syncsan: RACE (write then read) at 0x2ba22b84c900
  block 18 copy engine (issued by warp 17): UTMASTG.4D ...  [tma.cuh:74 < attention_partial.cu:577 < storer.cuh:6]
  block 55 warp 3: LDG.E.128 R8, ...                        [util.cuh:158 < matvec_adds.cu:145 < consumer.cuh:6]
  no barrier, release/acquire pair, async-copy wait or atomic orders these two accesses
```

`--kernel NAME` checks only the kernels whose name contains `NAME`, `--exitcode 1` makes a race fail your CI
([example workflow](examples/github-actions.yml)), and `--sample k` checks one in `k` plain accesses when the full
check is too slow. `syncsan run --help` lists the rest.

## What it has found

The clearest case is SuperLU_DIST. A commit that fixed a dropped fence in its GPU triangular solve restored the
consumer's acquire but left out the producer's release. syncsan flagged the producer's decrement, and in adversarial
mode the solve came back with an error of 0.7. On the fixed commit the error is 6e-15. The maintainers confirmed it and fixed it the same day
([xiaoyeli/superlu_dist#261](https://github.com/xiaoyeli/superlu_dist/issues/261)).

In Hazy Research's [Llama-1B megakernel](https://github.com/HazyResearch/Megakernels), activations move between SMs
through a TMA store, a relaxed `atomicAdd` on a counter and a `volatile` spin, with no gpu-scope fence. syncsan
reports 25 racing pairs, and in adversarial mode the model stops making sense (H100, 8 tokens):

| Build | Mode | Output |
|---|---|---|
| original | native | "Why did the cookie go to the" |
| original | adversarial | "iteriteriter abstractجرineronde" |
| original | control | "Why did the cookie go to the" |
| release/acquire counters | adversarial | "Why did the cookie go to the" |

The control run redirects the same loads but hands them their current values, so the redirection itself is not what
breaks the output. This race also fails on real hardware. When `silu_out` is filled with NaN before each step, the
original megakernel reads the NaN in all 1,000 generations and the fixed one in none. Making the counters
release/acquire fixes it at a 9% latency cost
([HazyResearch/Megakernels#16](https://github.com/HazyResearch/Megakernels/issues/16)).

The other finds:

- [AutoMegaKernel](https://github.com/RightNow-AI/AutoMegaKernel) waits with `atomicAdd(c, 0)`, which a comment calls
  an acquire but PTX treats as relaxed. In adversarial mode its own `--verify` check fails.
- [megakernel-zoo](https://github.com/dangxingyu/megakernel-zoo) and [Lucebox](https://github.com/Luce-Org/lucebox)
  release correctly but spin on a `volatile` load with no fence after it. In both, the adversarial output is garbage
  and a fence after the spin fixes it.
- [Mirage MPK](https://github.com/mirage-project/mirage)'s argmax reduction has no barrier between warp 0 reading the
  per-warp maxima and the other warps writing the next row's
  ([mirage-project/mirage#843](https://github.com/mirage-project/mirage/issues/843)).
- [Magni VM](https://github.com/Bobliuuu/mvm) had an admission race that 3,200 passing test runs never hit
  ([Bobliuuu/mvm#97](https://github.com/Bobliuuu/mvm/pull/97)).
- On CUTLASS 3.6.0, syncsan finds the known Stream-K race from issue #2027, and it finds nothing in the fixed 3.7.0.

Each one has a write-up in [docs/findings](docs/findings/) and a script in [reproduce](reproduce/) that reruns it on a
rented GPU. The summary of every run is in [results](results/).

## What it checks

syncsan follows the PTX memory model for global memory between thread blocks and for shared memory between warps,
including mbarriers and the Hopper and Blackwell copy engine (TMA, `cp.async.bulk`, `cp.async`). It does not check
races between lanes of one warp, distributed shared memory, or ordering across kernels and streams. It is slow: a
4-token decode of the Hazy megakernel takes 54 s under syncsan and 14 s without it.
[docs/how-it-works.md](docs/how-it-works.md) describes the model and its limits, and
[docs/prior-work.md](docs/prior-work.md) compares syncsan with Barracuda, iGUARD, SuperCollider and compute-sanitizer.

## Tests

`tests/run_tests.sh sm_90` runs 51 cases with known answers on a local GPU. [tests/cuhadron](tests/cuhadron/) adds six
cross-block synchronization tests to NVIDIA's [cuHadron](https://github.com/NVlabs/cuHadron) race benchmark.

## License

Apache 2.0. To cite syncsan, see [CITATION.cff](CITATION.cff).
