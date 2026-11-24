"""Patch Lucebox's AtomicGridSync (kernel.cu): `acquire` adds fence.acq_rel.gpu after the generation spin (the waiter's
missing acquire); `reset` makes the last arriver's `*counter = 0` an atomicExch; `both` does both."""
import pathlib
import sys

p, mode = pathlib.Path(sys.argv[1]), sys.argv[2]
s = p.read_text()
old_spin = "                while (*vgen <= my_gen) {}\n"
old_reset = "                *counter = 0;\n"
assert old_spin in s and old_reset in s
if mode in ("acquire", "both"):
    s = s.replace(old_spin, old_spin + "                GPU_MEMORY_FENCE();\n")
if mode in ("reset", "both"):
    s = s.replace(old_reset, "                atomicExch(counter, 0u);\n")
p.write_text(s)
print("patched", mode)
