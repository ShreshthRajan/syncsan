# syncsan with two kernels running at once on two streams (tests/concurrent_streams.cu), plus the known-answer cases.
. /work/syncsan/reproduce/daytona/common.sh
cd /work/syncsan/tests && nvcc -arch=sm_$ARCH -lineinfo -o /tmp/conc concurrent_streams.cu
step "concurrent, old behavior"; LD_PRELOAD=$SAN SYNCSAN_ALLOW_CONCURRENT=1 SYNCSAN_VERBOSE=1 /tmp/conc 2>&1 | grep -E "RACE|unchecked|distinct racing|concurrent_streams" | head -12
step concurrent; LD_PRELOAD=$SAN SYNCSAN_VERBOSE=1 /tmp/conc 2>&1 | grep -E "RACE|WARNING|unchecked|distinct racing|concurrent_streams" | head -20
step known-answer; cd /work/syncsan && NVBIT_ROOT=/opt/nvbit/nvbit_release_x86_64 bash tests/run_tests.sh sm_$ARCH 2>&1 | tail -8
