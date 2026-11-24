# NVIDIA's cuHadron race benchmark (SM90) under syncsan and compute-sanitizer racecheck: the racy builds (should report)
# and the FIXED builds (should not), then the scopedsync category this repository adds. Arg: TOOLS (syncsan,compute_sanitizer)
TOOLS=${1:-syncsan,compute_sanitizer}
CATDIR=${CATDIR:-/work/syncsan/cuhadron}
. /work/syncsan/reproduce/daytona/common.sh
step cuhadron
git clone -q https://github.com/NVlabs/cuHadron.git /opt/cuhadron && (cd /opt/cuhadron && git log --oneline -1)
bash $CATDIR/add_category.sh /opt/cuhadron > /dev/null
declare -A CMD=([compute_sanitizer]="compute-sanitizer --tool=racecheck --error-exitcode=1" [syncsan]="env LD_PRELOAD=$SAN SYNCSAN_EXITCODE=1")
for tool in ${TOOLS//,/ }; do
  for fixed in "" "--fixed"; do
    step "$tool ${fixed:-racy} (all)"
    (cd /opt/cuhadron && make clean >/dev/null 2>&1; export RUN_CMD="${CMD[$tool]}" ARCH=sm_$ARCH; bash ./run_tests.sh $fixed all 2>&1 | tee /work/out/${tool}${fixed}.log | grep -aE "PASS|FAIL|Summary|passed|failed|SKIP" | tail -40)
    step "$tool ${fixed:-racy} (scopedsync)"
    (cd /opt/cuhadron && export RUN_CMD="${CMD[$tool]}" ARCH=sm_$ARCH; for t in scopedsync/*.cu; do bash ./run_tests.sh $fixed ${t%.cu}; done 2>&1 | tee /work/out/${tool}${fixed}_scopedsync.log | grep -aE "PASS|FAIL" | tail -12)
  done
done
