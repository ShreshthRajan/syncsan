"""Mirage ed771bd's Hopper multitoken paged-attention codegen (src/kernel/task_register.cc, about line 1509) passes 15
template arguments to a format string with 14 markers, so every Hopper Qwen3 run aborts while transpiling. Add the
missing marker (for PAGE_STRIDE_ROWS). Upstream knows: "Fix Hopper paged-attention codegen and build sm90 kernels in CI"."""
import sys

p = sys.argv[1]
s = open(p).read()
old = ('code.e("kernel::multitoken_paged_attention_hopper_impl<bfloat16, $, $, $, $, "\n'
       '         "$, $, $, $, $, "\n'
       '         "$, $, $, $, $>(",')
new = old.replace('"$, $, $, $, $>(",', '"$, $, $, $, $, $>(",')
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, new))
print("codegen fix applied")
