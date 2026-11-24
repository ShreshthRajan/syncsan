// CUTLASS issue #2027 reproducer (Stream-K nondeterministic-reduction fixup), adapted from the issue's program:
// one SM90 TMA warp-specialized cooperative fp32 GEMM with the Stream-K scheduler forced, all-ones inputs, checked
// element-wise against CUTLASS's device reference GEMM. Changes from the issue: --mode=nd|det picks one reduction
// mode per process, --iters repeats the launch, and the report counts mismatching elements.
//   nvcc -std=c++17 -arch=sm_90a -O3 -lineinfo --expt-relaxed-constexpr -I$CUTLASS/include -I$CUTLASS/tools/util/include
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"

using namespace cute;

// Types at namespace scope: inside a class template GCC 11 rejects CUTLASS's epilogue builder (dependent-name bug).
struct Runner {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;
  using T = float;
  static constexpr int Align = 16 / sizeof(T);
  using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<T, T, T, T, cutlass::FloatRoundStyle::round_to_nearest>;
  using EpiOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto, T, T, T, LayoutC, Align, T, LayoutD, Align,
      cutlass::epilogue::TmaWarpSpecializedCooperative, DefaultOperation>::CollectiveOp;
  using MainOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, T, LayoutA, Align, T, LayoutB, Align, T,
      Shape<_128,_128,_64>, Shape<_2,_1,_1>,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename EpiOp::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, MainOp, EpiOp,
                                                          cutlass::gemm::StreamKScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  int run(bool Deterministic, int M, int N, int K, int iters, const cutlass::KernelHardwareInfo& hw) {
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
    cutlass::DeviceAllocation<T> A(size_t(M) * K), B(size_t(N) * K), C(size_t(M) * N), D(size_t(M) * N), R(size_t(M) * N);
    std::vector<T> ones(std::max(size_t(M) * K, size_t(N) * K), 1.f);
    A.copy_from_host(ones.data()); B.copy_from_host(ones.data()); C.copy_from_host(ones.data());

    cutlass::reference::device::GemmComplex({M, N, K}, T(1), cutlass::TensorRef(A.get(), LayoutA::packed({M, K})),
        cutlass::ComplexTransform::kNone, cutlass::TensorRef(B.get(), LayoutB::packed({K, N})), cutlass::ComplexTransform::kNone,
        T(0), cutlass::TensorRef(C.get(), LayoutC::packed({M, N})), cutlass::TensorRef(R.get(), LayoutD::packed({M, N})),
        T(0), 1, M * K, K * N, M * N, M * N);
    cudaDeviceSynchronize();
    std::vector<T> hR(size_t(M) * N), hD(size_t(M) * N);
    cudaMemcpy(hR.data(), R.get(), hR.size() * sizeof(T), cudaMemcpyDeviceToHost);

    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
                                  {A.get(), sA, B.get(), sB}, {{}, C.get(), sC, D.get(), sD}, hw};
    args.epilogue.thread.alpha = 1.f;
    args.epilogue.thread.beta = 0.f;
    using P = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams;
    args.scheduler.reduction_mode = Deterministic ? P::ReductionMode::Deterministic : P::ReductionMode::Nondeterministic;
    args.scheduler.decomposition_mode = P::DecompositionMode::StreamK;
    Gemm op;
    cutlass::device_memory::allocation<uint8_t> ws(Gemm::get_workspace_size(args));
    if (op.can_implement(args) != cutlass::Status::kSuccess) { printf("can_implement failed\n"); return 2; }
    int bad_iters = 0;
    for (int it = 0; it < iters; ++it) {
      cudaMemset(D.get(), 0xff, hD.size() * sizeof(T));
      op.initialize(args, ws.get());
      op.run();
      cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess) { printf("CUDA error %s\n", cudaGetErrorString(e)); return 3; }
      cudaMemcpy(hD.data(), D.get(), hD.size() * sizeof(T), cudaMemcpyDeviceToHost);
      size_t bad = 0, first = 0; float maxd = 0;
      for (size_t i = 0; i < hD.size(); ++i) {
        float d = std::abs(hD[i] - hR[i]);
        if (!(d == 0.f)) { if (!bad) first = i; ++bad; if (d > maxd || d != d) maxd = d; }
      }
      if (bad) {
        ++bad_iters;
        printf("iter %d: MISMATCH %zu / %zu elements, max |diff| %g, first at %zu (ref %g, got %g)\n", it, bad, hD.size(), maxd,
               first, hR[first], hD[first]);
      }
    }
    printf("mode=%s M=%d N=%d K=%d sms=%d: %d / %d iterations wrong -> %s\n", Deterministic ? "det" : "nd", M, N, K, hw.sm_count,
           bad_iters, iters, bad_iters ? "FAILED" : "PASSED");
    return bad_iters ? 1 : 0;
  }
};

int main(int argc, char const** argv) {
  cutlass::CommandLine cmd(argc, argv);
  int m, n, k, iters, sms; std::string mode;
  cmd.get_cmd_line_argument("m", m, 1024);
  cmd.get_cmd_line_argument("n", n, 1024);
  cmd.get_cmd_line_argument("k", k, 4096);
  cmd.get_cmd_line_argument("iters", iters, 1);
  cmd.get_cmd_line_argument("sms", sms, 0);
  cmd.get_cmd_line_argument("mode", mode, std::string("nd"));
  cutlass::KernelHardwareInfo hw;
  hw.device_id = 0;
  hw.sm_count = sms ? sms : cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  return Runner().run(mode == "det", m, n, k, iters, hw);
}
