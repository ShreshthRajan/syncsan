#!/bin/bash
# Build syncsan and its known-answer cases, run each case under the tool, and compare the count of RACE reports.
# Usage (on a machine with an NVIDIA GPU): NVBIT_ROOT=/path/to/nvbit_release_x86_64 tests/run_tests.sh [sm_90]
set -uo pipefail
cd "$(dirname "$0")/.."
ARCH="${1:-sm_90}"   # the TMA, bulk-copy and mbarrier cases need sm_90 or newer
cmake -S . -B build -DNVBIT_ROOT="$NVBIT_ROOT" -DNVBIT_TOOL_CUDA_ARCHITECTURES="${ARCH#sm_}" > build.log 2>&1 && cmake --build build -j >> build.log 2>&1 || { tail -40 build.log; exit 1; }
nvcc -O2 -arch="$ARCH" -lineinfo -o build/race_cases tests/race_cases.cu -lcuda || exit 1
fail=0
check() {  # case expect(racy|clean)
  out=$(LD_PRELOAD=$PWD/build/syncsan.so build/race_cases "$1" 2>&1); n=$(grep -c "syncsan: RACE" <<< "$out")
  if { [ "$2" = racy ] && [ "$n" -gt 0 ]; } || { [ "$2" = clean ] && [ "$n" -eq 0 ]; }; then echo "PASS $1 ($n reports, expected $2)"
  else echo "FAIL $1 ($n reports, expected $2)"; echo "$out" | tail -20; fail=1; fi
}
check cross_warp_racy racy
check cross_warp_barrier clean
check stage_racy racy
check stage_barrier clean
check atomics_only clean
check cross_block racy
check mp_relaxed_flag racy
check mp_acquire_flag clean
check mp_volatile_fence clean
check mp_no_release racy
check data_as_flag clean
check rmw_acqrel clean
check rmw_relaxed racy
check last_block_fenced clean
check last_block_nofence racy
check bulk_store_before_wait racy    # sm_90: cp.async.bulk shared->global
check bulk_store_waited clean
check bulk_store_barrier_first racy
check ldgsts_racy racy
check ldgsts_barrier clean
check big_grid_racy racy             # 16,384 blocks, folded onto 8,192 slots
check big_grid_clean clean
check tma_store_before_wait racy     # sm_90: TMA tensor store (UTMASTG) through a decoded tensor map
check tma_store_waited clean
check tma_store_barrier_first racy
check tma_load_nowait racy           # TMA tensor load (UTMALDG) completed through an mbarrier
check tma_load_waited clean
check smem_racy racy                 # shared memory (v4)
check smem_barrier clean
check smem_named_barrier clean
check smem_named_barrier_racy racy
check smem_mbarrier clean
check smem_mbarrier_racy racy
check cpasync_smem_racy racy
check cpasync_smem_waited clean
check bulk_store_smem_racy racy
check bulk_store_smem_waited clean
exit $fail
