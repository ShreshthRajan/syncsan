// The most common megakernel bug, in 30 lines. Block 1 writes 256 values, fences, and bumps a counter. Block 0 spins
// on the counter with a volatile load and then reads the values. The producer side is correct. The consumer has no
// acquire: nothing after its spin orders the reads after block 1's writes, so the PTX memory model lets it read stale
// data. Build with -DFIXED to add the missing fence after the spin.
//
//   nvcc -arch=sm_90 -lineinfo -o handoff examples/handoff.cu && ./handoff        # correct on real hardware
//   syncsan run -- ./handoff                                                     # the race report
//   SYNCSAN_ADVERSARIAL=1 syncsan run -- ./handoff                               # the wrong sum the model allows
//   nvcc -arch=sm_90 -lineinfo -DFIXED -o handoff examples/handoff.cu && syncsan run -- ./handoff   # clean
#include <cstdio>

__global__ void handoff(unsigned* counter, int* data, int* sum) {
  int t = threadIdx.x;
  if (blockIdx.x == 1) {
    data[t] = t + 1;
    __syncthreads();
    if (t == 0) { __threadfence(); atomicAdd(counter, 1u); }   // release: fence, then signal
  } else {
    if (t == 0) {
      while (*(volatile unsigned*)counter == 0) {}               // wait
#ifdef FIXED
      __threadfence();                                         // acquire: the fence after the wait
#endif
    }
    __syncthreads();
    atomicAdd(sum, data[t]);
  }
}

int main() {
  unsigned* counter; int *data, *sum;
  cudaMalloc(&counter, 4); cudaMalloc(&data, 256 * 4); cudaMalloc(&sum, 4);
  cudaMemset(counter, 0, 4); cudaMemset(data, 0, 256 * 4); cudaMemset(sum, 0, 4);
  handoff<<<2, 256>>>(counter, data, sum);
  int h = 0;
  cudaMemcpy(&h, sum, 4, cudaMemcpyDeviceToHost);
  printf("sum = %d (expected %d): %s\n", h, 256 * 257 / 2, h == 256 * 257 / 2 ? "correct" : "WRONG");
  return h == 256 * 257 / 2 ? 0 : 1;
}
