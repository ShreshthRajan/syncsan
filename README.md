# syncsan

syncsan finds data races in GPU megakernels and shows what they do to the output. It runs unmodified CUDA binaries under
[NVBit](https://github.com/NVlabs/NVBit), tracks happens-before as the PTX memory model defines it, and reports every
pair of accesses that no barrier, release/acquire pair, fence or async-copy wait orders. In adversarial mode, each load
that is not ordered after the last write to its word returns the value that word held before that write. For a race
between two same-size accesses that syncsan tracked, the PTX model allows that value, and syncsan counts at exit the
substitutions where this does not hold; the conditions are in [docs/how-it-works.md](docs/how-it-works.md). A race then
becomes a wrong output from a real GPU, under an execution the model permits.

## Example: Hazy Research's Llama-1B megakernel

The [low-latency Llama megakernel](https://github.com/HazyResearch/Megakernels) hands activations between SMs with a TMA
store, a relaxed `atomicAdd` on a counter, a `volatile` spin and a read, with no gpu-scope fence, so the PTX model does
not order the read after the store. syncsan reports 25 racing instruction pairs in global memory, covering these
handoffs: Q, K and V into attention, attention output into o_proj, `silu_out` into the down-projection, and the residual
stream into the next RMS norm, and one pair in shared memory from a second race, in how the QOL page is recycled
([docs/hazy-megakernel.md](docs/hazy-megakernel.md)). On an H100, 8 tokens:

| Build | Mode | Loads given an older value | Output |
|---|---|---|---|
| original | native | | "Why did the cookie go to the" |
| original | adversarial | 14.9M | "iteriteriter abstractجرineronde" |
| original | control (same redirection, current values) | 14.9M | "Why did the cookie go to the" |
| release/acquire counters | adversarial | 0 | "Why did the cookie go to the" |

The wrong text changes from run to run (earlier runs printed " insideuccarnochumptlink circular" and
"wedchuntghtunday造成uarios"); every run was wrong. Hazy's handoffs are TMA box stores read by 16-byte loads. PTX models a
vector or bulk access as accesses of its element type, and the machine code does not show element types, so syncsan
cannot tell whether these races are uniform-size (where the model allows the stale value) or mixed-size (where it
defines no outcome); on hardware, a NaN canary in `silu_out` is read in 1,000 of 1,000 generations of the original and 0
of 1,000 of the fixed build (the handoff #15 found), and a canary in `attn_out` in none of either. Hazy's native decode
is not bitwise repeatable across runs, so the comparison is on the text. The control run redirects the same loads but
hands them their current values, which rules out the redirection itself as the cause. Making the counters
`red.release.gpu` and `ld.acquire.gpu` removes every global-memory report and costs 9% in latency (two runs, each
against the original in the same container). The analysis and the fix are in
[docs/hazy-megakernel.md](docs/hazy-megakernel.md) and in
[HazyResearch/Megakernels#16](https://github.com/HazyResearch/Megakernels/issues/16). The same missing release/acquire
was found earlier by hand in [#15](https://github.com/HazyResearch/Megakernels/issues/15), on the up/gate to
down-projection handoff, where it changes tokens in 2.6% of generations on real H100s.

## Example: AutoMegaKernel

[AutoMegaKernel](https://github.com/RightNow-AI/AutoMegaKernel)'s consumers wait for their inputs with `atomicAdd(c,
0)`, which a source comment describes as an acquire. PTX treats it as relaxed, and no fence follows the spin, so syncsan
reports 24 racing pairs. Its `--verify` flag compares the megakernel's tokens with an eager PyTorch run (toy model,
H100):

| Build | Mode | Loads given an older value | `--verify` |
|---|---|---|---|
| original | native | | pass |
| original | adversarial | 54,144 | fail |
| original | control | | pass |
| wait as `ld.acquire.gpu` | adversarial | 0 | pass |

Every substituted word here was last written by an access of the same width as the load. syncsan left 5 instructions
unchecked because of the sm_90 NVBit hazard, so it warns that an unseen write could make a value one the model forbids.
Details in [docs/automegakernel.md](docs/automegakernel.md).

## Install

syncsan needs Linux on x86-64, an NVIDIA GPU of compute capability 8.0 or newer (tested on H100, B200 and RTX 5090),
CUDA 12 with `nvcc` (12.8 or newer for sm_120), and CMake 3.29 or newer. `syncsan build` downloads NVBit, which NVIDIA
distributes under its own license.

```
pip install git+https://github.com/ShreshthRajan/syncsan
syncsan build --arch 90        # once: downloads NVBit 1.8.2 and compiles the tool (90 = H100, 100 = B200, 120 = RTX 50xx, 80 = A100)
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
(synchronization is always checked), and `SYNCSAN_SHARED=0` turns off shared-memory checking.
`SYNCSAN_SKIP_OPS=PREFIX,...` (or `=OPCODE` for an exact match) leaves those memory opcodes unchecked, such as a model's
weight loads, which no step writes. [examples/github-actions.yml](examples/github-actions.yml) runs syncsan in CI on a
self-hosted GPU runner.

## What it checks

syncsan checks global memory between thread blocks (flags, counters and locks built from relaxed or strong accesses,
fences, release/acquire and `fence.sc`), shared memory between warps (full and named barriers, mbarriers), and the
Hopper and Blackwell copy engine, including the `fence.proxy.async` a bulk or TMA store needs after generic writes to
the shared memory it reads. TMA tensor copies, `cp.async.bulk` and `cp.async` are modeled as accesses by the block's
copy engine that become visible to a warp once it waits on them, with TMA coordinates decoded from the tensor maps the
program creates. [docs/how-it-works.md](docs/how-it-works.md) describes the model and its limits.

On NVIDIA's [cuHadron](https://github.com/NVlabs/cuHadron) race benchmark (SM90), syncsan detects 9 of the 21 single-GPU
races with no false positives on the fixed builds; compute-sanitizer from CUDA 12.4 detects 5 others, all in shared
memory. SuperCollider (PLDI 2026), whose authors wrote cuHadron, reports all but 3 of its 27 tests, so it covers more of
the benchmark than syncsan. cuHadron has no test of release/acquire or fence synchronization between thread blocks, so
[cuhadron/](cuhadron/) adds six, all detected by syncsan; compute-sanitizer does not check global memory and misses them
by design.

A release or acquire by one thread covers the other warps of its block only across a barrier or mbarrier, so a missing
`__syncthreads()` before thread 0's flag store, or after its acquire, is reported (launches of up to 320 blocks; larger
ones fall back to one clock per block). syncsan checks one kernel at a time and does not check races between lanes of
one warp, distributed shared memory, or ordering between kernels, streams and host copies. The full list of what can
hide a race or produce a false report is in [docs/how-it-works.md](docs/how-it-works.md).

## On hardware

Adversarial mode shows what a race permits. Whether a race also fails on today's GPUs is a separate measurement: before
every decode step or launch, the buffers a race involves are filled with NaN; a correct kernel overwrites them before any
read, so a NaN in the output means a consumer read before the producer's writes were visible. Each test has a positive
control, a build with the wait removed, to show the canary can fire. No instrumentation runs during these tests.

| Kernel, buffer | GPU | Original | Fixed | Control (wait removed) |
|---|---|---|---|---|
| Hazy, `silu_out` (#15's handoff) | H100 | 1,000 of 1,000 generations | 0 of 1,000 | (the original) |
| Hazy, `attn_out` | H100 | 0 of 1,000 | 0 of 1,000 | |
| AutoMegaKernel, activation arena | H100 | 0 of 200 | 0 of 200 | 20 of 20 |
| Lucebox, five scratch buffers | RTX 5090 | 0 of 1,000 | 0 of 1,000 | 20 of 20 |

The CUTLASS 3.6.0 race (50 of 50 runs wrong) and the SuperLU_DIST race before its fix (wrong solutions) also fail
natively. The other races syncsan reports have not failed in these tests; for them the evidence is what the PTX model
allows. Summaries are in `results/*canary*` and `results/hazy_witness_nan_20261124`.

## Other results

**megakernel-zoo and Lucebox.** Two LLM decode megakernels release correctly (a fence, then `atomicAdd` on a grid
counter) but spin on the counter with a `volatile` load and no fence after it, so the waiter has no acquire. In
[megakernel-zoo](https://github.com/dangxingyu/megakernel-zoo)'s Qwen3-8B engine syncsan reports 9 racing pairs and
adversarial mode turns " Paris. The capital" into "_ and of["; in [Lucebox](https://github.com/Luce-Org/lucebox)'s
Qwen3.5-0.8B megakernel it reports 72 pairs on an RTX 5090 and the adversarial output is wrong at every step. In both,
the control is bitwise equal to native, and a fence after the spin restores the correct output. Lucebox's barrier also
resets its counter with a plain store that races with the other blocks' `atomicAdd`. Details in
[docs/megakernel-zoo.md](docs/megakernel-zoo.md) and [docs/lucebox.md](docs/lucebox.md).

**Mirage MPK.** In Mirage's persistent megakernel (Qwen3-1.7B, H100), syncsan reports a shared-memory race in the argmax
reduction: it runs once per batch row with no barrier between warp 0 reading the per-warp maxima and the other warps
writing the next row's, so a warp that runs ahead could change another row's token. It needs several rows in one task
(prefill, batching, speculative decoding) and did not show up in our greedy runs. Mirage runs its worker and scheduler
kernels at once, which exposed a syncsan bug (fixed in 0.8.0). Details in [docs/mirage.md](docs/mirage.md).

These results, the Hazy rerun and the reruns on RTX 5090 are on syncsan 0.9 (`results/*_0.9_*`); CUTLASS below was measured with 0.6; the SuperLU_DIST fix check is on 0.10.

**SuperLU_DIST.** The GPU triangular solve's fix for a dropped fence (commit 587f38f) restored the consumer's acquire
but left the producer's release out: syncsan reported 4 racing pairs, and adversarial mode gave a wrong solution (error
about 0.7). The SuperLU_DIST developers confirmed it in [xiaoyeli/superlu_dist#261](https://github.com/xiaoyeli/superlu_dist/issues/261)
and fixed it the same day in 239280c, with a release fence before each producer's decrement in the double, single and
complex solves. On 239280c syncsan reports 0 pairs and the adversarial solve is correct (error 6e-15). The original bug
(4755a5e) is flagged at the same lines and fails natively. Details in [docs/superlu.md](docs/superlu.md).

**CUTLASS Stream-K.** On CUTLASS issue #2027, syncsan reports the known race between the middle split's `atomicAdd` and
the final split's read in 3.6.0, which fails natively, and nothing in the fixed 3.7.0. Details in
[docs/cutlass-streamk.md](docs/cutlass-streamk.md).

**Magni VM.** syncsan found an admission race in the [Magni VM](https://github.com/Bobliuuu/mvm) megakernel in one run,
after 3,200 test runs had passed; the fix is submitted as [Bobliuuu/mvm#97](https://github.com/Bobliuuu/mvm/pull/97). On syncsan 0.10 the stage suite
shows 64 racing pairs before the fix and 0 after (`results/magni_0.10_20261124`, `reproduce/daytona/magni.sh`).

## Cost

NVBit's call at each instrumented access accounts for most of the slowdown. Hazy's 4-token script takes 54 s under
syncsan with every check on, against 14 s natively. A training megakernel whose 54 µs step is almost all global accesses
runs about 9,000 times slower; `--sample` is the remedy there.

## Prior work

Adversarial memory comes from Jumble (Flanagan and Freund, PLDI 2010), which returned stale values allowed by the Java
memory model to show that a race was harmful. Bugaroo (ISSRE 2018) brought stale values to GPUs through a recompiled
binary and does not limit them to values the model allows. Happens-before race detection on GPU binaries goes back to
Barracuda (PLDI 2017), with CURD (PLDI 2018) and iGUARD (SOSP 2021) after it; Triton's GSan does it for global memory in
Triton programs, and SuperCollider (PLDI 2026) finds races by delaying reads in a recompiled binary. syncsan adds a
model of Hopper and Blackwell async copies (TMA, bulk copies, `cp.async`, mbarriers) to binary happens-before checking,
and turns the races it finds into outputs the model allows. [docs/prior-work.md](docs/prior-work.md) covers the rest.

## Reproducing the results

[reproduce/](reproduce/) reruns the Hazy, AutoMegaKernel, megakernel-zoo, Lucebox, Mirage, SuperLU_DIST and CUTLASS
results on rented GPUs through Modal or Daytona. The cuHadron and Magni VM runs are not scripted here yet.
`tests/run_tests.sh` runs 49 known-answer checks (race and clean cases, adversarial and control cases, the two-stream
check) on a local GPU.

## Citing

See [CITATION.cff](CITATION.cff).

## License

Apache 2.0.
