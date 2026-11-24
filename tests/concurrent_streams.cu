// Two kernels running at once on two streams, as Mirage's worker and scheduler kernels do. `worker` is race free: warp 0
// writes shared memory, a barrier, warp 1 reads it, many times, while it waits for `helper` (launched second, on another
// stream) to set a flag. Before syncsan checked one kernel at a time, the helper's launch reset the worker's state
// mid-run and syncsan reported races in the worker. Expected: no races, and a warning that the helper ran unchecked.
#include <cstdio>
__global__ void worker(volatile int* flag, float* out) {
  __shared__ float s[64];
  float acc = 0;
  for (int it = 0; it < 2000 || !*flag; ++it) {
    if (threadIdx.x < 32) s[threadIdx.x] = it + threadIdx.x;
    __syncthreads();
    if (threadIdx.x >= 32) acc += s[threadIdx.x - 32];
    __syncthreads();
    if (it > 1000000) break;
  }
  out[blockIdx.x * 64 + threadIdx.x] = acc;
}
__global__ void helper(volatile int* flag) {
  __shared__ int t[32];
  t[threadIdx.x] = threadIdx.x;
  __syncthreads();
  if (threadIdx.x == 0) { __nanosleep(1000000); *flag = 1 + t[1]; }
}
int main() {
  int* flag; float* out; cudaMalloc(&flag, 4); cudaMemset(flag, 0, 4); cudaMalloc(&out, 8 * 64 * 4);
  cudaStream_t a, b; cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking); cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking);
  worker<<<8, 64, 0, a>>>(flag, out);
  helper<<<1, 32, 0, b>>>(flag);
  cudaError_t e = cudaDeviceSynchronize();
  printf("concurrent_streams: %s\n", cudaGetErrorString(e));
}
