#!/bin/bash
# Build syncsan and its known-answer cases, run each case under the tool, and compare the count of RACE reports.
# Usage (on a machine with an NVIDIA GPU, after `syncsan build` or with NVBIT_ROOT set): tests/run_tests.sh [sm_80]
set -uo pipefail
cd "$(dirname "$0")/.."
NVBIT_ROOT="${NVBIT_ROOT:-${SYNCSAN_HOME:-$HOME/.cache/syncsan}/nvbit_release_x86_64}"   # where `syncsan build` puts NVBit
ARCH="${1:-sm_80}"
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
check proxy_fence_missing racy       # generic writes, then a bulk store with no fence.proxy.async
check proxy_fence_present clean

# a release or acquire by one thread covers the other warps of its block only across a barrier
check spinlock_no_release racy       # the fence before the critical section does not release it
check spinlock_released clean
check mp_block_barriers clean
check mp_block_no_producer_barrier racy
check mp_block_no_consumer_barrier racy

# adversarial mode: what out[0] must hold (a racing load gets the value before the write; an ordered one, the current
# value; a load after the thread's own sub-word store is left alone), and the control leaves every value current
adv() {  # case mode(1|control) expected-out0
  out=$(SYNCSAN_ADVERSARIAL=$2 LD_PRELOAD=$PWD/build/syncsan.so build/race_cases "$1" 2>&1)
  got=$(sed -n 's/.*out\[0\] = \(-\{0,1\}[0-9]*\).*/\1/p' <<< "$out" | tail -1)
  if [ "$got" = "$3" ]; then echo "PASS adversarial=$2 $1 (out[0] = $got)"; else echo "FAIL adversarial=$2 $1 (out[0] = $got, expected $3)"; echo "$out" | tail -8; fail=1; fi
}
adv adv_mp_relaxed 1 0
adv adv_mp_acquire 1 1
adv adv_mp_relaxed control 1
adv adv_subword 1 30583

# global checks with shared-memory checking off (SYNCSAN_SHARED=0) and with block-level clocks (SYNCSAN_WARP_ORDER=0)
for env in SYNCSAN_SHARED=0 SYNCSAN_WARP_ORDER=0; do
  n=$(env $env LD_PRELOAD=$PWD/build/syncsan.so build/race_cases mp_relaxed_flag 2>&1 | grep -c "syncsan: RACE")
  if [ "$n" -gt 0 ]; then echo "PASS $env mp_relaxed_flag ($n reports, expected racy)"; else echo "FAIL $env mp_relaxed_flag (0 reports, expected racy)"; fail=1; fi
done

# two kernels at once on two streams: the first is checked, the second runs unchecked, and no race is reported
nvcc -O2 -arch="$ARCH" -lineinfo -o build/concurrent_streams tests/concurrent_streams.cu || exit 1
out=$(LD_PRELOAD=$PWD/build/syncsan.so build/concurrent_streams 2>&1)
if [ "$(grep -c "syncsan: RACE" <<< "$out")" = 0 ] && grep -q "runs unchecked" <<< "$out"; then echo "PASS concurrent_streams (0 reports, second kernel unchecked)"; else echo "FAIL concurrent_streams"; echo "$out" | tail -20; fail=1; fi
exit $fail
