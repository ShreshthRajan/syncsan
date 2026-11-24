# Changelog

## 0.4.0 (2026-11-23)

First public release.

- Happens-before checking of global memory between thread blocks: release/acquire, fences, `fence.sc`, atomics.
- Shared memory between warps: full and named barriers, mbarrier arrive and `try_wait`.
- Async copies as copy-engine accesses: TMA tensor loads, stores and reductions (decoded from `cuTensorMapEncodeTiled`),
  `cp.async.bulk`, `cp.async`.
- Adversarial mode (`SYNCSAN_ADVERSARIAL=1`) and its control.
- H100 (sm_90) and B200 (sm_100); 37 known-answer cases.
- `pip` package with `syncsan build` and `syncsan run`.
