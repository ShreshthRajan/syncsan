# Mirage MPK: a shared-memory race in the argmax reduction

[Mirage](https://github.com/mirage-project/mirage)'s MPK compiles a model into a persistent megakernel: worker blocks
run tasks (attention, linear layers, norms, argmax) that a separate scheduler kernel hands them through queues. Checked
at ed771bd on an H100 with Qwen3-1.7B (2 tokens, `SYNCSAN_KERNEL=worker_kernel`), syncsan reports 9 racing instruction
pairs in `worker_kernel`. The shadow table filled during the run and 110M accesses went untracked, so the check is
partial.

## The race

`block_reduce_max_idx` (`include/mirage/persistent_kernel/tasks/ampere/argmax.cuh:34-72`) has each warp's lane 0 write
its maximum into `smem_vals[warp]` and `smem_idxs[warp]`, then `__syncthreads()`, then warp 0 reads all of them.
`argmax_partial_kernel` and `argmax_reduce_kernel` call it once per batch row in a loop (`for (int batch_idx = 0;
batch_idx < num_active_tokens; batch_idx++)`). Nothing orders warp 0's reads for row `b` before another warp's writes
for row `b + 1`, so a warp that runs ahead can overwrite `smem_vals[w]` before warp 0 reads it, and row `b` can get the
wrong token. syncsan reports the pair (`argmax.cuh:62` read, `argmax.cuh:49` write) and the same pair in the sm100 copy
(`argmax_sm100.cuh:69` and `:56`, a named barrier in place of `__syncthreads()`).

The race needs more than one row in one task: prefill chunks, several requests in a batch, or speculative decoding,
where each verified position is a row. The window is small, because warp 0 reads right after the barrier while the other
warps first scan their part of the next row, and greedy decoding was deterministic in our runs (3 runs of 473 tokens
each, Qwen3-8B). A barrier at the end of `block_reduce_max_idx`, or two alternating buffers, removes it.

## The other reports

- 5 pairs at worker start-up (one of these records names an unrelated `LDG` as the writer: two writes to the word were
  recorded at once) (`persistent_kernel.cuh:989-990`, `:1021`): every thread stores the same values into
  `worker_queues[0]` and `worker_queue_ids[0]`, and thread 0 reads them without a barrier. Benign: the values are equal.
- 2 pairs between a TMA load of a linear layer's input in one block (`linear_swapAB_hopper.cuh:265`) and a later store by
  `rmsnorm_hopper.cuh:149` in another block. Mirage orders these tasks through its scheduler kernel, which syncsan does
  not check (below), so these two are not verified.

## A syncsan bug Mirage exposed

Mirage launches `worker_kernel` and `scheduler_kernel` at the same time on two streams. syncsan reset its tables at
every checked launch, so the scheduler's launch cleared the worker's state while the worker ran, and the first Mirage
run reported 643 pairs, nearly all false. syncsan now checks one kernel at a time: a launch that starts while a checked
kernel runs on another stream runs unchecked, with a warning (`tests/concurrent_streams.cu` is the regression test).

## Also found

At ed771bd the Hopper paged-attention code generator (`src/kernel/task_register.cc`, about line 1509) passes 15 template
arguments to a format string with 14 markers, so every Qwen3 run on an H100 aborts while transpiling. Adding the missing
marker fixes it; an upstream "Fix Hopper paged-attention codegen" change exists. The OUTPUT_SIZE cap in
`demo/qwen3/demo.py` ("not same output for same prompt") never applies to Qwen3-8B's layer sizes, and runs with it
removed were deterministic too, as were runs that also restored the store wait the swapAB linear kernel has commented
out (`reproduce/daytona/mirage_fix.py`); that wait guards an output slot the next TMA store may still be reading.
