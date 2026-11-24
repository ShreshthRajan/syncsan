# Changelog

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
