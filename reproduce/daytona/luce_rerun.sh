# Lucebox on this syncsan version (warp-level ordering, sub-word guard), plus the block-wide cases under both models.
. /work/syncsan/reproduce/daytona/common.sh
nvcc -O2 -arch=sm_$ARCH -lineinfo -o /tmp/rc /work/syncsan/tests/race_cases.cu -lcuda
for c in mp_block_barriers mp_block_no_producer_barrier mp_block_no_consumer_barrier; do
  echo "$c block-level: $(SYNCSAN_WARP_ORDER=0 LD_PRELOAD=$SAN /tmp/rc $c 2>&1 | grep -c 'syncsan: RACE') warp-level: $(LD_PRELOAD=$SAN /tmp/rc $c 2>&1 | grep -c 'syncsan: RACE')"
done
bash /work/syncsan/reproduce/daytona/luce.sh native,syncsan,adversarial,control none 16 0
