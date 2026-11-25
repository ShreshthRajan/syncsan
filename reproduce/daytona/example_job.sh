# Runs examples/handoff.cu natively, under syncsan, in adversarial mode, and fixed, as the README's quick start shows.
. /work/syncsan/reproduce/daytona/common.sh
step example
cd /work/syncsan
for v in racy fixed; do
  F=""; [ $v = fixed ] && F=-DFIXED
  nvcc -arch=sm_$ARCH -lineinfo $F -o /tmp/handoff_$v examples/handoff.cu
  echo "--- $v native";      /tmp/handoff_$v
  echo "--- $v syncsan";     LD_PRELOAD=$SAN /tmp/handoff_$v 2>&1 | grep -aE "RACE|block|sum =|syncsan: [0-9]" | head -8
  echo "--- $v adversarial"; SYNCSAN_ADVERSARIAL=1 LD_PRELOAD=$SAN /tmp/handoff_$v 2>&1 | grep -aE "sum =|substitut" | head -3
done 2>&1 | tee /work/out/example.log
