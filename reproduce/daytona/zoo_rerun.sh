# megakernel-zoo on this syncsan version (warp-level ordering, sub-word guard), and the known-answer tests, on an H100.
. /work/syncsan/reproduce/daytona/common.sh
step known-answer
(cd /work/syncsan && NVBIT_ROOT=/opt/nvbit/nvbit_release_x86_64 bash tests/run_tests.sh sm_$ARCH > /work/out/tests.log 2>&1; grep -aE "^FAIL|mp_block" /work/out/tests.log; echo "passed: $(grep -c ^PASS /work/out/tests.log)  failed: $(grep -c ^FAIL /work/out/tests.log)")
bash /work/syncsan/reproduce/daytona/zoo.sh native,syncsan,adversarial,control none 4 0
