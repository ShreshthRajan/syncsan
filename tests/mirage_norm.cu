// Mirage's norm_hopper.cuh reduction pattern, reduced: 4 warps of a 256-thread block loop over heads; each warp writes
// its partial into reduce[warp] through a generic pointer, named barrier 1 over 128 threads, every thread reads
// reduce[tid] (tid < 4), thread 0 writes reduce[0], barrier 1 again, everyone reads reduce[0]. The other 4 warps wait at
// __syncthreads. Expected under syncsan: only the read of reduce[0] (after the second barrier) against warp 0's write of
// reduce[0] at the next head (WAR, no barrier between them). mode 1 adds a third barrier at the end of each head: clean.
// mode 2 is Mirage's RoPE path, whose barrier inside the row loop orders the read: clean.
#include <cstdio>
__device__ __forceinline__ void wg_sync(unsigned id) { asm volatile("bar.sync %0, 128;" ::"r"(id)); }
__global__ void norm(float* out, int mode, int heads, const float* w, const float* cs) {
  __shared__ float red_s[8];
  __shared__ float x_s[4 * 128];
  float* reduce = red_s;                       // generic pointer, as Mirage's smem_ptr helpers give
  asm volatile("" : "+l"(reduce));
  const int warp = threadIdx.x / 32;
  if (warp < 4) {
    float acc = 0;
    for (int h = 0; h < heads; ++h) {
      float sum = threadIdx.x * 0.5f + h;
      for (int o = 16; o > 0; o /= 2) sum += __shfl_xor_sync(~0u, sum, o);
      if (threadIdx.x % 32 == 0) reduce[warp] = sum;
      wg_sync(1);
      sum = threadIdx.x < 4 ? reduce[threadIdx.x] : 0.f;
      for (int o = 16; o > 0; o /= 2) sum += __shfl_xor_sync(~0u, sum, o);
      if (threadIdx.x == 0) reduce[0] = sum;
      wg_sync(1);
      const float rms = reduce[0];
      acc += rms;
      if (mode == 1) wg_sync(1);
      if (mode == 2) {   // Mirage's RoPE path: scale the row, barrier, rotate (the barrier orders the read of reduce[0])
        float* x = x_s + (h % 4) * 128;
        for (int i = threadIdx.x; i < 128; i += 128) {
          float v = x[i] * rms * w[i];
          x[i] = v;
          wg_sync(1);
          float c = cs[h * 128 + i], sn = cs[4096 + h * 128 + i];
          float v2 = x[i < 64 ? i + 64 : i - 64];
          float r = i < 64 ? v * c - v2 * sn : v * c + v2 * sn;
          wg_sync(1);
          x[i] = r;
        }
      }
    }
    out[blockIdx.x * 256 + threadIdx.x] = acc;
  }
  __syncthreads();
}
int main(int argc, char** argv) {
  int mode = argc > 1 ? atoi(argv[1]) : 0;
  float *d, *w, *cs; cudaMalloc(&d, 512 * sizeof(float)); cudaMalloc(&w, 128 * sizeof(float)); cudaMalloc(&cs, 8192 * sizeof(float));
  cudaMemset(w, 0, 128 * sizeof(float)); cudaMemset(cs, 0, 8192 * sizeof(float));
  norm<<<2, 256>>>(d, mode, 8, w, cs);
  cudaError_t e = cudaDeviceSynchronize();
  float h; cudaMemcpy(&h, d, sizeof h, cudaMemcpyDeviceToHost);
  printf("mode %d: %s out[0]=%g\n", mode, cudaGetErrorString(e), h);
}
