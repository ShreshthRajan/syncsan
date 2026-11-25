"""Restore the commented-out store wait in Mirage's Hopper swapAB linear kernel, so an output slot is not rewritten
while the TMA store issued Kstages iterations earlier may still be reading it. The original commented order
(barrier, then wait) would not work: only the issuing lane has store groups to wait on."""
import sys

p = sys.argv[1]
s = open(p).read()
old = "      // wg_sync<THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(1);\n      // store_async_wait<Kstages - 1>();\n"
assert old in s
# wait_group is per thread: the lane that committed the store waits, then the barrier releases the other threads
new = ("      if ((warp_idx & 3) == 0 && lane_idx == 0) {\n        store_async_wait<Kstages - 1>();\n      }\n"
       "      wg_sync<THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(1);\n")
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("fix applied")
