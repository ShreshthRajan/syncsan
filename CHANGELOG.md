# Changelog

## 0.10.3 (2026-11-24)

- Fixed a missed race: a release published the block clock at the releasing store instead of at its fence, so accesses
  between the fence and the store counted as released (a spinlock whose only fence follows the lock). The block clock
  now ticks at each fence and the release publishes the fence-time value; an SC fence publishes before the tick.
  `spinlock_no_release` / `spinlock_released` cases; 51 known-answer checks on H100. SuperLU_DIST unchanged (4 pairs on
  587f38f, 0 on 239280c). Found by the puzzles built on syncsan.

## 0.10.2 (2026-11-24)

- SuperLU_DIST: the developers confirmed the incomplete fix syncsan found and fixed it in 239280c; syncsan reports 0
  pairs there and the adversarial solve is correct (`reproduce/daytona/superlu.sh`).

## 0.10.1 (2026-11-24)

- Native canary tests with positive controls (`reproduce/hazy/canary.py`, `reproduce/amk/canary.py`,
  `reproduce/luce/canary.py`): Hazy's `silu_out` handoff reads stale data on an H100 in 1,000 of 1,000 generations
  before the release/acquire fix and 0 of 1,000 after; Hazy's `attn_out`, AutoMegaKernel and Lucebox did not fail.
- Magni VM rerun on 0.10: 64 racing pairs before the fix, 0 after.

## 0.10.0 (2026-11-24)

- `fence.proxy.async`: a bulk or TMA store that reads shared memory written by a warp with no proxy fence in the block
  between the write and the copy's issue is reported (`proxy_fence_missing` / `proxy_fence_present`). 49 known-answer
  checks pass on H100 (`results/tests/0.10_h100_20261124`).
- Hazy: the shared-memory pair is a second race, the QOL page finished before the consumer's `L_smem` store when
  `skip_attn_reduction` is set ([docs/findings/hazy-megakernel.md](docs/findings/hazy-megakernel.md)).

## 0.9.0 (2026-11-24)

- Warp-level ordering between blocks (launches of up to 320 blocks): a release by one thread publishes another warp's
  accesses only if that warp synchronized with it first, and an acquire by one thread orders another warp's reads only
  through a barrier or mbarrier. A missing `__syncthreads()` before a one-thread flag store, or after a one-thread
  acquire, is now reported; block-level clocks missed both (`tests/race_cases.cu` `mp_block_*`). Warps of one block are
  ordered by the shared clocks (full and named barriers, mbarriers) in global memory too. `SYNCSAN_WARP_ORDER=0` keeps
  the old per-block clocks.
- Adversarial mode enforces what it can of its legality conditions: a store smaller than 4 bytes voids the word's
  recorded value, a record read while it changes is left alone, `--sample` is ignored, the exit summary counts
  substituted words whose last write had another access width, and a warning names unseen writes. Adversarial and control cases in
  `tests/run_tests.sh`.
- Fixed: with `SYNCSAN_SHARED=0` the per-launch reset failed and nothing was checked (introduced and fixed during 0.9
  development; covered by a new test).
- Reruns on 0.9 with shared-memory checking on: Hazy (26 pairs, one shared-memory pair unexplained; every substituted
  word was last written by an access of another width), AutoMegaKernel (unchanged), cuHadron (unchanged: 9 of 21, the six scopedsync tests, 0
  false positives on single-GPU fixed builds), Lucebox on RTX 5090 (72 pairs, adversarial wrong, all same-size).
  Summaries in `results/*_0.9_*`.
- Daytona runners for the tests, Hazy, AutoMegaKernel, cuHadron, Magni VM, Lucebox and megakernel-zoo.

## 0.8.1 (2026-11-24)

Documentation corrections after a review against the run logs.

- The stale-value claim now states its conditions (same-size races, every write to the word seen).
- Limits are split into what can hide a race and what can produce a false report; new entries include the block-wide
  release and acquire, the 16-bit barrier count, unchecked concurrent kernels and grids over 4,096 blocks.
- Prior work: Barracuda (PLDI 2017) and CURD (PLDI 2018) before iGUARD; SuperCollider's cuHadron result beside syncsan's.
- Corrected numbers: Hazy fix latencies (per-container pairs, +9.0% and +9.2%), compute-sanitizer on cuHadron (5 others),
  CUTLASS 3.6.0 pairs (110). megakernel-zoo's engine departs from Hugging Face at token 13 of 16.
- `tests/run_tests.sh` finds NVBit where `syncsan build` put it.

## 0.8.0 (2026-11-24)

- One kernel at a time: a launch that starts while a checked kernel runs on another stream runs unchecked, with a
  warning. Before, its reset cleared the running kernel's state and produced false reports (Mirage MPK runs its worker
  and scheduler kernels at once). Regression test `tests/concurrent_streams.cu`, run by `tests/run_tests.sh`.
- Mirage MPK: a shared-memory race in the argmax reduction.
- Lucebox on an RTX 5090 (sm_120): adversarial wrong tokens, control bitwise, fixed build clean; 20,000 native decodes
  identical on the 5090 as on the H100.
- Daytona runners in `reproduce/daytona/`.

## 0.7.0 (2026-11-23)

- megakernel-zoo (Qwen3-8B) and Lucebox (Qwen3.5-0.8B): missing acquires after grid-counter spins, shown as wrong
  tokens in adversarial mode and fixed by a fence after the spin; Lucebox's plain-store counter reset. Results and
  reproductions.
- `SYNCSAN_SKIP_OPS` documented.
- The source distribution includes the CUDA sources, so `python -m build` builds the wheel.

## 0.6.0 (2026-11-23)

- One race record per instruction pair, deduplicated on the GPU. The record buffer used to fill with repeats of the
  first pairs, so later pairs went unprinted; Hazy now shows 25 pairs, among them the handoff of Megakernels#15.
- CTA-scope fences (`MEMBAR.ALL.CTA`) no longer count as releases or acquires between blocks.
- SuperLU_DIST (an incomplete fix) and CUTLASS Stream-K (a known bug, flagged and cleared) results and reproductions.

## 0.5.0 (2026-11-23)

- Adversarial mode replaces each eligible load and performs it itself, so it covers any plain or relaxed global load,
  not only loads that overwrite their own address register.
- AutoMegaKernel: adversarial mode makes `--verify` fail; an acquire in its counter wait fixes it.
- Source chains are computed once per module (a megakernel with thousands of linked functions timed out before).

## 0.4.0 (2026-11-23)

First public release.

- Happens-before checking of global memory between thread blocks: release/acquire, fences, `fence.sc`, atomics.
- Shared memory between warps: full and named barriers, mbarrier arrive and `try_wait`.
- Async copies as copy-engine accesses: TMA tensor loads, stores and reductions (decoded from `cuTensorMapEncodeTiled`),
  `cp.async.bulk`, `cp.async`.
- Adversarial mode (`SYNCSAN_ADVERSARIAL=1`) and its control.
- H100 (sm_90) and B200 (sm_100); 37 known-answer cases.
- `pip` package with `syncsan build` and `syncsan run`.
