# Magni VM (github.com/Bobliuuu/mvm) admission race: the stage suite under syncsan, before (test) and after (fix) the
# barrier fix (Bobliuuu/mvm#97). magni/0001 adds the stage suite, magni/0002 is the fix. Each nvcc of test_exec.cu takes minutes.
. /work/syncsan/reproduce/daytona/common.sh
PATCHES=${PATCHES:-/work/syncsan/reproduce/magni}
step mvm
G="git -c user.name=t -c user.email=t@t"
git clone -q https://github.com/Bobliuuu/mvm.git /opt/mvm && cd /opt/mvm && git checkout -q 60c42de && \
  $G checkout -q -b test && $G am -q $PATCHES/0001-*.patch && $G checkout -q -b fix && $G am -q $PATCHES/0002-*.patch && \
  sed 's/ -DMVM_EXTRA_OPS_DEF=[^ ]* -DMVM_EXTRA_OPS_CUH=[^ ]*//' tests/gpu/build_core.sh > /opt/build_core_noextra.sh
for rev in test fix; do
  step "build $rev"; (cd /opt/mvm && git checkout -q $rev && MVM_ARCH=sm_$ARCH bash /opt/build_core_noextra.sh /opt/test_exec_$rev > /work/out/build_$rev.log 2>&1; ls -la /opt/test_exec_$rev)
  step "native stage ($rev)"; MVM_TEST_QUICK=1 /opt/test_exec_$rev stage 2>&1 | grep -aE "== stage" | head -3
  step "syncsan stage ($rev)"; LD_PRELOAD=$SAN MVM_TEST_QUICK=1 SYNCSAN_KERNEL=mvm_exec_kernel SYNCSAN_VERBOSE=1 /opt/test_exec_$rev stage 2>&1 | tee /work/out/syncsan_stage_$rev.log | grep -aE "== stage|distinct racing" | head -4
done
