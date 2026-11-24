# How syncsan works

syncsan instruments every memory access, barrier, fence and async copy of the kernels it checks, keeps a vector clock
for every participant, and compares each access with the last write and the last read of the same 4-byte word. Two
accesses race when at least one is a write, they are not both synchronizing operations (strong or atomic), and no
happens-before edge orders them.

## Happens-before

Within a block, warps are ordered by barriers. Each warp counts the full-block barriers it has passed, and two warps'
accesses to global memory are ordered when a barrier separates them.

Across blocks, each block keeps a vector clock over all blocks. A strong store or an atomic that follows a fence in the
same thread is a release: the location takes the block's clock. A strong load or an atomic that a fence follows is an
acquire: the block's clock joins the location's. These are the patterns that `st.release`, `ld.acquire`, `volatile` with
`__threadfence()`, and atomics with fences compile to on sm_80 through sm_100. `fence.sc` (what `__threadfence()`
compiles to) is totally ordered, and each SC fence synchronizes with the later ones, so syncsan keeps one SC clock per
launch.

Shared memory has its own per-warp clocks inside each block. Full and named barriers are edges among their participants,
an mbarrier arrive is a release into the mbarrier's clock, and a successful `try_wait` is an acquire.

## Async copies

The copy engine of each block is modeled as extra participants with clocks of their own. Issuing a TMA tensor copy, a
`cp.async.bulk` or a `cp.async` starts the copy from what the issuing warp has seen; the bytes it reads or writes are
checked as the copy engine's accesses. A warp is ordered after the copy once it waits for it: `cp.async.bulk.wait_group`
or `cp.async.wait_all` for copies that complete on the issuing thread, a successful `try_wait` on the copy's mbarrier
for copies that complete there. The rest of the block is ordered after the copy at the waiting warp's next barrier,
other blocks at its next release. TMA coordinates become byte ranges through the tensor maps the program encodes, which
syncsan records by intercepting `cuTensorMapEncodeTiled`.

## Adversarial mode

`SYNCSAN_ADVERSARIAL=1` changes what racing loads return. Each write records the value it replaces. syncsan removes each
plain or relaxed global load from the kernel (NVBit's `nvbit_remove_orig`) and performs it itself, then writes the
destination registers: a word whose load is not ordered after the word's last write gets the value that write replaced,
and every other word gets what memory holds. When the race is between two accesses of the same size and syncsan saw
every write to the word, the PTX model allows the older value: a load cannot return a value older than the last write
that happens-before it, and no write after the replaced one does. The value can be one the model forbids when a write to
the word went unseen: a store smaller than 4 bytes, a store left out by `--sample` or `SYNCSAN_SKIP_OPS`, a store left
unchecked because of the sm_90 hazard below, a store dropped because the shadow table was full, a store by a kernel
running unchecked on another stream, or two writes to the word recorded at the same moment. The model also says nothing
about races between accesses of different sizes, such as a TMA box store read by 16-byte loads. A wrong output in
adversarial mode therefore shows that the race lies on the data path to the output; for races outside these cases it
also shows an output the model permits. Loads that directly follow another load in flight are left in place, because a
call placed there corrupts the earlier load on sm_90. `SYNCSAN_ADVERSARIAL=control` performs the same replacement with
current values, to check that the mechanism alone does not change the output.

## Limits

**What can hide a race:**
- A release or acquire by one thread counts for its whole block, so a missing `__syncthreads()` between the block's
  writes and one thread's release, or between one thread's acquire and the block's reads, is not reported.
- Named barriers over a subset of warps count as full barriers for global memory, and in global memory warps of one
  block that have passed different numbers of barriers count as ordered.
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
- In global memory, ordering between warps of one block by anything other than a barrier (an mbarrier or a flag).
- Synchronization through a kernel that ran unchecked because another checked kernel was running.
- A full sync table or clock pool (syncsan warns at exit).
- More than 65,536 barriers passed by one warp, where the 16-bit barrier count wraps.
- When two writes to one word are recorded at once, a report can name the wrong writing instruction.

**Not checked:** races between lanes of one warp, distributed shared memory, ordering between kernels, streams and host
copies, multi-GPU programs, programmatic dependent launch, im2col and multicast TMA, WGMMA and tcgen05 operand reads of
shared memory, and `fence.proxy.async` (generic writes are taken as visible to the copy engine).

## Cost

NVBit replaces each instrumented instruction with a call that saves and restores the kernel's live registers. On a
training megakernel that uses about 255 registers, an empty call per access already costs about 6,300 times the native
step time, and syncsan's checks add 1.4 times on top. Hazy's Llama megakernel, which streams its weights through TMA,
runs its whole 4-token script 3.9 times slower with every check on. `--sample k` instruments one plain load or store
site in `k`.
