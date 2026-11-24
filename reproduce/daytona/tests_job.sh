# Known-answer tests on this GPU (all cases, the two-stream check, the block-wide and adversarial cases).
. /work/syncsan/reproduce/daytona/common.sh
step known-answer
cd /work/syncsan && NVBIT_ROOT=/opt/nvbit/nvbit_release_x86_64 bash tests/run_tests.sh sm_$ARCH 2>&1 | tee /work/out/tests.log | grep -vE "^PASS" ; echo "passed: $(grep -c ^PASS /work/out/tests.log)  failed: $(grep -c ^FAIL /work/out/tests.log)"
