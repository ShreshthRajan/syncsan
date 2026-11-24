#include <cuda.h>
// Known-answer kernels for syncsan. ./race_cases <case>; each case's expected verdict is in run_tests.sh.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

// Warp 1 writes, warp 0 reads the same words, with or without a barrier between.
__global__ void cross_warp(int* buf, int* out, int barrier) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (warp == 1) buf[lane] = lane + 1;
  if (barrier) __syncthreads();
  if (warp == 0) out[lane] = buf[lane];
}

// Shaped like Magni VM's admission: the block copies a record in 16-byte chunks (thread t owns chunks t, t+B, ...), then
// reads 4-byte words of it at another thread's index. Without the barrier, a word can be read before its owner stored it.
__global__ void stage_then_read(const int4* src, int4* stage, int* out, int n4, int barrier) {
  stage += (size_t)blockIdx.x * n4; out += (size_t)blockIdx.x * 4 * n4;   // each block its own stage buffer
  for (int k = threadIdx.x; k < n4; k += blockDim.x) stage[k] = src[k];
  if (barrier) __syncthreads();
  const int* words = reinterpret_cast<const int*>(stage);
  for (int e = threadIdx.x; e < 4 * n4 - 37; e += blockDim.x) out[e] = __ldcg(words + e + 37);
}

// The same word written by every warp of a block, but with atomics only: never a race.
__global__ void atomics_only(int* counter) { atomicAdd(counter, 1); }

// Different blocks touch the same words with nothing ordering them.
__global__ void cross_block(int* buf, int* out) {
  if (blockIdx.x == 1) buf[threadIdx.x] = 7;
  if (blockIdx.x == 0) out[threadIdx.x] = buf[threadIdx.x];
}

__device__ __forceinline__ unsigned ld_relaxed(const unsigned* p) { unsigned v; asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_relaxed(unsigned* p, unsigned v) { asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void st_release(unsigned* p, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }

// Message passing between two blocks, one thread each: block 1 writes 32 data words then a flag, block 0 waits for the
// flag then reads the data. mode 0: relaxed flag load (no acquire); 1: ld.acquire; 2: volatile load then __threadfence.
__global__ void mp(unsigned* flag, unsigned* data, unsigned* out, int mode) {
  if (threadIdx.x != 0) return;
  if (blockIdx.x == 1) {
    for (int i = 0; i < 32; ++i) data[i] = i + 1;                        // plain stores
    st_release(flag, 1);
  } else {
    if (mode == 0) while (ld_relaxed(flag) == 0) {}                      // relaxed: no acquire
    else if (mode == 1) while (ld_acquire(flag) == 0) {}
    else { while (*(volatile unsigned*)flag == 0) {} __threadfence(); }
    for (int i = 0; i < 32; ++i) out[i] = data[i];                       // plain loads
  }
}

// The producer publishes with a relaxed store and no fence, so the consumer's acquire has nothing to synchronize with.
// A kernel of its own: with a release anywhere in the same kernel the compiler may hoist that fence above both stores.
__global__ void mp_no_release(unsigned* flag, unsigned* data, unsigned* out) {
  if (threadIdx.x != 0) return;
  if (blockIdx.x == 1) { for (int i = 0; i < 32; ++i) data[i] = i + 1; st_relaxed(flag, 1); }
  else { while (ld_acquire(flag) == 0) {} for (int i = 0; i < 32; ++i) out[i] = data[i]; }
}

// Shaped like Magni VM's indegree handoff: block 1 writes 32 words, fences, then adds 1 to a counter with atom.acq_rel;
// block 0 spins adding 0 with atom.acq_rel until it sees 1, then reads the words. acq_rel = 1: clean; 0: both adds relaxed, a race.
__device__ __forceinline__ unsigned atom_add_acqrel(unsigned* p, unsigned v) { unsigned r; asm volatile("atom.acq_rel.gpu.global.add.u32 %0,[%1],%2;" : "=r"(r) : "l"(p), "r"(v) : "memory"); return r; }
__device__ __forceinline__ unsigned atom_add_relaxed(unsigned* p, unsigned v) { unsigned r; asm volatile("atom.relaxed.gpu.global.add.u32 %0,[%1],%2;" : "=r"(r) : "l"(p), "r"(v) : "memory"); return r; }
__global__ void rmw_handoff(unsigned* counter, unsigned* data, unsigned* out, int acq_rel) {
  if (threadIdx.x != 0) return;
  if (blockIdx.x == 1) {
    for (int i = 0; i < 32; ++i) data[i] = i + 1;
    if (acq_rel) { __threadfence(); atom_add_acqrel(counter, 1); } else atom_add_relaxed(counter, 1);
  } else {
    if (acq_rel) while (atom_add_acqrel(counter, 0) == 0) {} else while (atom_add_relaxed(counter, 0) == 0) {}
    for (int i = 0; i < 32; ++i) out[i] = data[i];
  }
}

// The last-block reduction (CUDA's threadFenceReduction, PyTorch's reduce_kernel, open-flapping-airplanes' signal_ret):
// each block's thread 0 writes a partial, every thread runs __threadfence() (fence.sc) and the block syncs, then thread 0
// bumps a counter with a relaxed atomic; the block that sees the last ticket syncs and reads every partial. With a
// __threadfence() in the reader after the sync (fenced = 1) the SC fences order the partials before the reads: clean.
// fenced = 0 drops every fence, so nothing orders the partials before the reads in any execution: a race. (Dropping only
// the reader's fence is not a deterministic race: whether the reader's earlier fence orders a writer's depends on the
// order the fences happened to take, so a sound happens-before detector may rightly see none.)
__global__ void last_block(unsigned* partial, unsigned* count, unsigned* out, int fenced) {
  __shared__ bool last;
  if (threadIdx.x == 0) partial[blockIdx.x] = blockIdx.x + 1;
  if (fenced) __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(count, 1u) == gridDim.x - 1;
  __syncthreads();
  if (!last) return;
  if (fenced) __threadfence();
  if (threadIdx.x < gridDim.x) out[threadIdx.x] = partial[threadIdx.x];
}

// Data-as-flag: the producer writes each word with a strong store and the consumer polls each word with a strong load
// until it leaves the sentinel. Every access is synchronizing, so it's clean however it interleaves.
__global__ void data_as_flag(unsigned* data, unsigned* out) {
  if (threadIdx.x >= 32) return;
  if (blockIdx.x == 1) st_relaxed(data + threadIdx.x, threadIdx.x + 1);
  else { unsigned v; while ((v = ld_relaxed(data + threadIdx.x)) == 0) {} out[threadIdx.x] = v; }
}

// cp.async.bulk shared->global by thread 0 (warp 0); warp 1 reads the destination. mode 0: reads before the wait (racy);
// 1: wait, barrier, read (clean); 2: barrier, read, then wait (racy: the copy may still be writing after the barrier).
__global__ void bulk_store(int* g, int* out, int mode) {
  __shared__ alignas(128) int s[64];
  const int t = threadIdx.x;
  s[t] = t;
#if __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  const unsigned sa = (unsigned)__cvta_generic_to_shared(s);
  if (t == 0) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 256;" :: "l"(g), "r"(sa) : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  }
  if (mode == 0) { if (t >= 32) out[t] = g[t]; if (t == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
  else if (mode == 1) { if (t == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); __syncthreads(); if (t >= 32) out[t] = g[t]; }
  else { __syncthreads(); if (t >= 32) out[t] = g[t]; if (t == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
#endif
}

// cp.async (LDGSTS) reads g in warp 0 while warp 1 writes it: racy without the barrier between them.
__global__ void ldgsts(int* g, int* out, int barrier) {
  __shared__ int s[32];
  const int t = threadIdx.x;
  if (barrier) { if (t >= 32) g[t - 32] = t; __syncthreads(); }
  if (t < 32) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"((unsigned)__cvta_generic_to_shared(s + t)), "l"(g + t) : "memory");
    asm volatile("cp.async.wait_all;" ::: "memory");
    out[t] = s[t];
  }
  if (!barrier && t >= 32) g[t - 32] = t;
}

// More blocks than syncsan's 8,192 slots: each block writes its own words; with racy, the last block also writes word 0,
// which block 0 writes too.
__global__ void big_grid(int* g, int racy) {
  const unsigned b = blockIdx.x;
  g[b * blockDim.x + threadIdx.x] = (int)b;
  if (racy && b == gridDim.x - 1 && threadIdx.x == 0) g[0] = -1;
}

// TMA tensor copies over a 64x64 int32 tensor in buf, 16x16 box at (col 16, row 32). Warp 1 touches row 32, cols 16..47.
// tma_store: mode 0 reads before the wait (racy), 1 waits then barrier then reads (clean), 2 barrier then reads then waits
// (racy). tma_load: warp 1 overwrites the source; wait = 1 waits on the mbarrier first (clean: the copy is done), 0 doesn't.
__global__ void tma_store(const __grid_constant__ CUtensorMap tm, int* g, int* out, int mode) {
#if __CUDA_ARCH__ >= 900
  __shared__ alignas(128) int s[16 * 16];
  const int t = threadIdx.x;
  for (int i = t; i < 256; i += blockDim.x) s[i] = i;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (t == 0) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                 :: "l"(&tm), "r"(16), "r"(32), "r"((unsigned)__cvta_generic_to_shared(s)) : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  }
  int* row = g + 32 * 64 + 16;
  if (mode == 0) { if (t >= 32) out[t] = row[t - 32 < 16 ? t - 32 : 0]; if (t == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
  else if (mode == 1) { if (t == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); __syncthreads(); if (t >= 32) out[t] = row[(t - 32) % 16]; }
  else { __syncthreads(); if (t >= 32) out[t] = row[(t - 32) % 16]; if (t == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
#endif
}
__global__ void tma_load(const __grid_constant__ CUtensorMap tm, int* g, int* out, int wait) {
#if __CUDA_ARCH__ >= 900
  __shared__ alignas(128) int s[16 * 16];
  __shared__ alignas(8) unsigned long long mbar;
  const int t = threadIdx.x;
  const unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar);
  if (t == 0) { asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mb)); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  if (t < 16) g[32 * 64 + 16 + t] = t;   // the source was produced earlier in the launch (TMA loads are checked where it wrote)
  __syncthreads();
  if (t == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], 1024;" :: "r"(mb) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"((unsigned)__cvta_generic_to_shared(s)), "l"(&tm), "r"(16), "r"(32), "r"(mb) : "memory");
  }
  if (wait || t < 32) asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.shared.b64 p, [%0], 0; @!p bra W; }" :: "r"(mb) : "memory");
  if (t >= 32) g[32 * 64 + 16 + (t - 32) % 16] = -t;   // overwrites the copy's source
  out[t] = s[t];
#endif
}

// ---- shared memory ----
// Warp 0 writes s, warp 1 reads it. mode 0: no ordering (racy); 1: __syncthreads (clean); 2: named barrier 1 over the two
// warps, producer bar.arrive / consumer bar.sync (clean); 3: consumer reads before its bar.sync (racy); 4: mbarrier,
// producer arrives, consumer try_waits then reads (clean); 5: consumer reads without waiting (racy).
__global__ void smem_handoff(int* out, int mode) {
  __shared__ int s[32];
  __shared__ alignas(8) unsigned long long mbar;
  const int t = threadIdx.x, lane = t & 31;
  const unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar);
  if (t == 0) asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mb) : "memory");
  __syncthreads();
  if (t < 32) {
    s[lane] = lane + 1;
    if (mode == 1) __syncthreads();
    else if (mode == 2 || mode == 3) asm volatile("bar.arrive 1, 64;" ::: "memory");
    else if (mode >= 4) { __syncwarp(); if (lane == 0) asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(mb) : "memory"); }
  } else {
    int v = 0;
    if (mode == 1) __syncthreads();
    else if (mode == 2) asm volatile("bar.sync 1, 64;" ::: "memory");
    else if (mode == 3) { v = s[lane]; asm volatile("bar.sync 1, 64;" ::: "memory"); }
    else if (mode == 4) asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.shared.b64 p, [%0], 0; @!p bra W; }" :: "r"(mb) : "memory");
    if (mode != 3) v = s[lane];
    out[lane] = v;
    if (mode == 5) asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.shared.b64 p, [%0], 0; @!p bra W; }" :: "r"(mb) : "memory");
  }
}

// cp.async into shared memory: read before cp.async.wait_all (racy) or after it (clean); same thread.
__global__ void cpasync_smem(const int* g, int* out, int wait_first) {
  __shared__ int s[64];
  const int t = threadIdx.x;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"((unsigned)__cvta_generic_to_shared(s + t)), "l"(g + t) : "memory");
  int v;
  if (wait_first) { asm volatile("cp.async.wait_all;" ::: "memory"); v = s[t]; }
  else { v = s[t]; asm volatile("cp.async.wait_all;" ::: "memory"); }
  out[t] = v;
}

// cp.async.bulk shared -> global reads s while warp 1 overwrites it: racy without wait_group.read + barrier first.
__global__ void bulk_store_smem(int* g, int fixed) {
  __shared__ alignas(128) int s[64];
  const int t = threadIdx.x;
  s[t] = t;
#if __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (t == 0) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 256;" :: "l"(g), "r"((unsigned)__cvta_generic_to_shared(s)) : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    if (fixed) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
  }
  if (fixed) __syncthreads();
  if (t >= 32) s[t - 32] = -t;
  if (!fixed && t == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
#endif
}

int main(int argc, char** argv) {
  const char* c = argc > 1 ? argv[1] : "";
  int *buf, *out; int4 *src, *stage;
  const int n4 = 4096;
  CK(cudaMalloc(&buf, 1 << 20)); CK(cudaMalloc(&out, 1 << 20)); CK(cudaMalloc(&src, n4 * 16)); CK(cudaMalloc(&stage, 4 * n4 * 16));
  CK(cudaMemset(buf, 0, 1 << 20)); CK(cudaMemset(src, 1, n4 * 16));
  if (!strcmp(c, "cross_warp_racy")) cross_warp<<<1, 64>>>(buf, out, 0);
  else if (!strcmp(c, "cross_warp_barrier")) cross_warp<<<1, 64>>>(buf, out, 1);
  else if (!strcmp(c, "stage_racy")) stage_then_read<<<4, 256>>>(src, stage, out, n4, 0);
  else if (!strcmp(c, "stage_barrier")) stage_then_read<<<4, 256>>>(src, stage, out, n4, 1);
  else if (!strcmp(c, "atomics_only")) atomics_only<<<2, 256>>>(buf);
  else if (!strcmp(c, "cross_block")) cross_block<<<2, 64>>>(buf, out);
  else if (!strcmp(c, "mp_relaxed_flag")) mp<<<2, 32>>>((unsigned*)buf, (unsigned*)buf + 64, (unsigned*)out, 0);
  else if (!strcmp(c, "mp_acquire_flag")) mp<<<2, 32>>>((unsigned*)buf, (unsigned*)buf + 64, (unsigned*)out, 1);
  else if (!strcmp(c, "mp_volatile_fence")) mp<<<2, 32>>>((unsigned*)buf, (unsigned*)buf + 64, (unsigned*)out, 2);
  else if (!strcmp(c, "mp_no_release")) mp_no_release<<<2, 32>>>((unsigned*)buf, (unsigned*)buf + 64, (unsigned*)out);
  else if (!strcmp(c, "rmw_acqrel")) rmw_handoff<<<2, 32>>>((unsigned*)buf, (unsigned*)buf + 64, (unsigned*)out, 1);
  else if (!strcmp(c, "rmw_relaxed")) rmw_handoff<<<2, 32>>>((unsigned*)buf, (unsigned*)buf + 64, (unsigned*)out, 0);
  else if (!strcmp(c, "last_block_fenced")) last_block<<<64, 128>>>((unsigned*)buf, (unsigned*)buf + 1024, (unsigned*)out, 1);
  else if (!strcmp(c, "last_block_nofence")) last_block<<<64, 128>>>((unsigned*)buf, (unsigned*)buf + 1024, (unsigned*)out, 0);
  else if (!strcmp(c, "data_as_flag")) data_as_flag<<<2, 32>>>((unsigned*)buf, (unsigned*)out);
  else if (!strcmp(c, "bulk_store_before_wait")) bulk_store<<<1, 64>>>(buf, out, 0);
  else if (!strcmp(c, "bulk_store_waited")) bulk_store<<<1, 64>>>(buf, out, 1);
  else if (!strcmp(c, "bulk_store_barrier_first")) bulk_store<<<1, 64>>>(buf, out, 2);
  else if (!strcmp(c, "ldgsts_racy")) ldgsts<<<1, 64>>>(buf, out, 0);
  else if (!strcmp(c, "ldgsts_barrier")) ldgsts<<<1, 64>>>(buf, out, 1);
  else if (!strcmp(c, "big_grid_racy")) big_grid<<<16384, 16>>>(buf, 1);
  else if (!strcmp(c, "big_grid_clean")) big_grid<<<16384, 16>>>(buf, 0);
  else if (!strncmp(c, "smem_", 5)) {
    const char* m[] = {"smem_racy", "smem_barrier", "smem_named_barrier", "smem_named_barrier_racy", "smem_mbarrier", "smem_mbarrier_racy"};
    int mode = -1;
    for (int i = 0; i < 6; ++i) if (!strcmp(c, m[i])) mode = i;
    if (mode < 0) { printf("unknown case %s\n", c); return 2; }
    smem_handoff<<<1, 64>>>(out, mode);
  }
  else if (!strcmp(c, "cpasync_smem_racy")) cpasync_smem<<<1, 64>>>(buf, out, 0);
  else if (!strcmp(c, "cpasync_smem_waited")) cpasync_smem<<<1, 64>>>(buf, out, 1);
  else if (!strcmp(c, "bulk_store_smem_racy")) bulk_store_smem<<<1, 64>>>(buf, 0);
  else if (!strcmp(c, "bulk_store_smem_waited")) bulk_store_smem<<<1, 64>>>(buf, 1);
  else if (!strncmp(c, "tma_", 4)) {
    CUtensorMap tm; cuuint64_t dims[2] = {64, 64}, strides[1] = {64 * 4}; cuuint32_t box[2] = {16, 16}, es[2] = {1, 1};
    if (cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_INT32, 2, buf, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                               CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
      printf("cuTensorMapEncodeTiled failed\n"); return 1;
    }
    if (!strcmp(c, "tma_store_before_wait")) tma_store<<<1, 64>>>(tm, buf, out, 0);
    else if (!strcmp(c, "tma_store_waited")) tma_store<<<1, 64>>>(tm, buf, out, 1);
    else if (!strcmp(c, "tma_store_barrier_first")) tma_store<<<1, 64>>>(tm, buf, out, 2);
    else if (!strcmp(c, "tma_load_nowait")) tma_load<<<1, 64>>>(tm, buf, out, 0);
    else if (!strcmp(c, "tma_load_waited")) tma_load<<<1, 64>>>(tm, buf, out, 1);
    else { printf("unknown case %s\n", c); return 2; }
  }
  else { printf("unknown case %s\n", c); return 2; }
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  int first = 0;   // what the kernel's out[0] holds (for SYNCSAN_DELAY witness runs)
  CK(cudaMemcpy(&first, out, sizeof first, cudaMemcpyDeviceToHost));
  printf("case %s done, out[0] = %d\n", c, first);
  return 0;
}
