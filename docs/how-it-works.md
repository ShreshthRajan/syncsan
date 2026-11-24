# How syncsan works

syncsan instruments every memory access, barrier, fence and async copy of the kernels it checks, keeps a vector clock
for every participant, and compares each access with the last write and the last read of the same 4-byte word. Two
accesses race when at least one is a write, they are not both synchronizing operations (strong or atomic), and no
happens-before edge orders them.

## Happens-before

Each warp keeps a vector clock over the warps of its block (and the block's copy engine). Full and named barriers are
edges among their participants, an mbarrier arrive is a release into the mbarrier's clock, and a successful `try_wait`
is an acquire. Two warps of one block are ordered, in global and in shared memory, when such an edge separates their
accesses.

Across blocks, each block keeps a vector clock over all blocks. A strong store or an atomic that follows a fence in the
same thread is a release: the location takes the block's clock. A strong load or an atomic that a fence follows is an
acquire: the location's clock joins the acquiring warp's record of what it has acquired, and barriers and mbarriers
carry that record to the other warps of the block. These are the patterns that `st.release`, `ld.acquire`, `volatile`
with `__threadfence()`, and atomics with fences compile to on sm_80 through sm_120. `fence.sc` (what `__threadfence()`
compiles to) is totally ordered, and each SC fence synchronizes with the later ones, so syncsan keeps one SC clock per
launch.

A release by one thread publishes another warp's accesses only if that warp synchronized with the releasing thread after
them. Each release records, per warp of its block, the latest access it covers; a reader in another block is ordered
after a write only if it acquired a release that covers the writing warp. So a missing `__syncthreads()` between a
block's writes and its thread 0's flag store is a race, and so is a missing `__syncthreads()` between thread 0's acquire
and the other warps' reads. This warp-level ordering applies to launches of up to 320 blocks (megakernels launch one or
two blocks per SM); larger launches use one clock per block, where a release or acquire counts for the whole block.
`SYNCSAN_WARP_ORDER=0` selects the per-block clocks for any launch.

## Async copies

The copy engine of each block is modeled as extra participants with clocks of their own. Issuing a TMA tensor copy, a
`cp.async.bulk` or a `cp.async` starts the copy from what the issuing warp has seen; the bytes it reads or writes are
checked as the copy engine's accesses. A warp is ordered after the copy once it waits for it: `cp.async.bulk.wait_group`
or `cp.async.wait_all` for copies that complete on the issuing thread, a successful `try_wait` on the copy's mbarrier
for copies that complete there. The rest of the block is ordered after the copy at the waiting warp's next barrier,
other blocks at its next release. TMA coordinates become byte ranges through the tensor maps the program encodes, which
syncsan records by intercepting `cuTensorMapEncodeTiled`.

**Proxy fences.** A bulk or TMA store reads shared memory through the async proxy, so a warp's ordinary write to that
memory must be followed by `fence.proxy.async` before the copy is issued (CUTLASS's `tma_store_fence`). syncsan counts
each block's shared-memory proxy fences (`FENCE.VIEW.ASYNC.S`) and reports a copy that reads a word whose last write,
otherwise ordered before the copy, came after the block's last proxy fence: then no thread fenced in between, so no
causality chain holds a fence. A fence by an unrelated thread in between hides the report.

## Adversarial mode

`SYNCSAN_ADVERSARIAL=1` changes what racing loads return. Each write records the value it replaces. syncsan removes each
plain or relaxed global load from the kernel (NVBit's `nvbit_remove_orig`) and performs it itself, then writes the
destination registers: a word whose load is not ordered after the word's last write gets the value that write replaced,
and every other word gets what memory holds. When the race is between two accesses of the same size and syncsan saw
every write to the word, the PTX model allows the older value: a load cannot return a value older than the last write
that happens-before it, and no write after the replaced one does. syncsan enforces what it can of this: a store smaller
than 4 bytes marks the word's recorded value unusable, a record read while it changes is left alone, `--sample` is
ignored in adversarial mode, and the summary at exit counts the substituted words whose last write had another access
width and warns when writes went unseen (skipped opcodes, a full shadow, writes left unchecked by the sm_90 hazard,
kernels left unchecked). PTX models a vector or bulk access as accesses of its element type, which the machine code does
not show, so a different width is a mixed-size race (for which the model defines no outcome) only when the element types
differ. When the width count is zero and no warning is printed, every substituted value is one the model allows, given
the orderings syncsan models (an ordering it misses, listed under false reports below, would make the value one the
model forbids); otherwise a wrong output shows that the race lies on the data path to the output. Loads that directly
follow another load in flight are left in place, because a call placed there corrupts the earlier load on sm_90.
`SYNCSAN_ADVERSARIAL=control` performs the same replacement with current values, to check that the mechanism alone does
not change the output.

## Limits

**What can hide a race:**
- In launches of more than 320 blocks, a release or acquire by one thread counts for its whole block.
- Coverage of a release is cumulative over the block's releases: once a release covers a warp's access, later releases
  of that block count as covering it too.
- A wait counts every copy the block has issued as done.
- Each word keeps only its last reader.
- Grids of more than 8,192 blocks share clock slots; grids of more than 4,096 blocks are not checked on the global side
  of async copies.
- A copy's shared-memory side is checked at one word per 128 bytes, and TMA loads are checked only on pages the launch
  has written.
- Accesses smaller than 4 bytes, `--sample`, `SYNCSAN_SKIP_OPS`, a full shadow table (counted at exit as untracked
  accesses), and kernels launched while another checked kernel runs.
- `.cta`-scoped strong accesses between blocks are treated as synchronizing.

**What can produce a false report:**
- Ordering between warps of one block through a shared-memory flag instead of a barrier or mbarrier, and, in launches
  of more than 320 blocks, through anything other than a barrier.
- Synchronization through a kernel that ran unchecked because another checked kernel was running.
- A full sync table or clock pool (syncsan warns at exit).
- In launches of more than 320 blocks, more than 65,536 barriers passed by one warp, where the 16-bit barrier count
  wraps.
- When two writes to one word are recorded at once, a report can name the wrong writing instruction.

**Not checked:** races between lanes of one warp, distributed shared memory, ordering between kernels, streams and host
copies, multi-GPU programs, programmatic dependent launch, im2col and multicast TMA, WGMMA and tcgen05 operand reads of
shared memory, `fence.proxy.async` for global memory, and the proxy fence a copy that writes shared memory needs before
generic reads (completion through `wait_group` or an mbarrier already provides it).

## Cost

NVBit replaces each instrumented instruction with a call that saves and restores the kernel's live registers. On a
training megakernel that uses about 255 registers, an empty call per access already costs about 6,300 times the native
step time, and syncsan's checks add 1.4 times on top. Hazy's Llama megakernel, which streams its weights through TMA,
runs its whole 4-token script 3.9 times slower with every check on. `--sample k` instruments one plain load or store
site in `k`.
