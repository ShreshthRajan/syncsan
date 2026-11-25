# AutoMegaKernel native canary witness (NaN activation arena), original and acquire-fixed builds. Arg: R
R=${1:-200}
. /work/syncsan/reproduce/daytona/common.sh
step amk-setup
git clone -q https://github.com/RightNow-AI/AutoMegaKernel.git /opt/amk && cd /opt/amk && git checkout -q 884534c && timeout 900 uv sync -q 2>&1 | tail -3
cp /work/syncsan/reproduce/amk/canary.py /opt/amk/canary.py
step "amk canary original"
(cd /opt/amk && export PATH=/opt/amk/.venv/bin:$PATH && timeout 2400 .venv/bin/python canary.py $R 2>&1 | tee /work/out/canary_orig.log | grep -aE "CANARY|Error|Traceback" | tail -4)
step "amk canary fixed"
cd /opt/amk && python3 - <<'PY'
import pathlib; p = pathlib.Path('vm/sync.cuh'); s = p.read_text()
s2 = s.replace('    return atomicAdd(c, 0u);', '    uint32_t v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(c) : "memory"); return v;')
assert s2 != s; p.write_text(s2); print('patched')
PY
rm -rf ~/.cache/torch_extensions
(cd /opt/amk && export PATH=/opt/amk/.venv/bin:$PATH && timeout 2400 .venv/bin/python canary.py $R 2>&1 | tee /work/out/canary_fixed.log | grep -aE "CANARY|Error|Traceback" | tail -4)

step "amk canary positive control (waits removed)"
cd /opt/amk && git checkout -q vm/sync.cuh && python3 - <<'PY'
import pathlib; p = pathlib.Path('vm/sync.cuh'); s = p.read_text()
s2 = s.replace('    return atomicAdd(c, 0u);', '    return 0xFFFFFFFFu;   // canary control: every wait returns at once')
assert s2 != s; p.write_text(s2); print('patched')
PY
rm -rf ~/.cache/torch_extensions
(cd /opt/amk && export PATH=/opt/amk/.venv/bin:$PATH && timeout 1200 .venv/bin/python canary.py 20 2>&1 | tee /work/out/canary_control.log | grep -aE "CANARY|Error|Traceback" | tail -4)
