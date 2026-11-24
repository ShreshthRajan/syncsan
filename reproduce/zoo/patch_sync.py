"""Patch the megakernel-zoo Qwen3-8B engine (mk.cu): `acquire` adds __threadfence() after each spin on the grid
counter P.sync (grid_barrier, the coordinator-warp barrier, the QKV_NOWAIT wait), the consumer side of the
fence-then-atomicAdd release the engine already has."""
import pathlib
import re
import sys

p, mode = pathlib.Path(sys.argv[1]), sys.argv[2]
s = p.read_text()
n0 = s.count("__threadfence();")
s = s.replace("  while ((int)(*(volatile unsigned*)P.sync - target) < 0) polls++;\n",
              "  while ((int)(*(volatile unsigned*)P.sync - target) < 0) polls++;\n  __threadfence();\n")
s = s.replace("    TS();                                 // 6: all blocks arrived\n",
              "    __threadfence();\n    TS();                                 // 6: all blocks arrived\n")
s = s.replace("    while ((int)(*(volatile unsigned*)P.sync - prev) < 0) { }\n",
              "    while ((int)(*(volatile unsigned*)P.sync - prev) < 0) { }\n    __threadfence();\n")
assert s.count("__threadfence();") == n0 + 3, s.count("__threadfence();") - n0
p.write_text(s)
print("patched", mode)
