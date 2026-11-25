# Mirage MPK Qwen3 demo (H100). Native determinism under three builds, then syncsan on the small model.
# Args: VARIANTS (comma list of orig,nocap,nocap_fix,fix) REPS NTOK SAN (0|1)
VARIANTS=${1:-orig,nocap,nocap_fix}; REPS=${2:-5}; NTOK=${3:-64}; RUNSAN=${4:-0}; SANVARIANTS=${5:-orig}; SANTOK=${6:-2}
. /work/syncsan/reproduce/daytona/common.sh
step known-answer; (cd /work/syncsan && NVBIT_ROOT=/opt/nvbit/nvbit_release_x86_64 bash tests/run_tests.sh sm_$ARCH 2>&1 | tail -4)
step mirage-install
if [ ! -f /opt/.mirage_done ]; then
$PIPI "transformers==4.57.1" accelerate safetensors huggingface_hub z3-solver cuda-python pytest
git clone -q --recursive https://github.com/mirage-project/mirage /opt/mirage && (cd /opt/mirage && git checkout -q ed771bd && git submodule update -q --init --recursive)
export MIRAGE_HOME=/opt/mirage
python /work/syncsan/reproduce/mirage/codegen_fix.py /opt/mirage/src/kernel/task_register.cc
cp -r /opt/mirage /opt/mirage.pristine
(cd /opt/mirage && pip install -e . -v > /work/out/mirage_build.log 2>&1; tail -3 /work/out/mirage_build.log)
python -c "import mirage; print('mirage ok')" || { echo "mirage did not build"; tail -40 /work/out/mirage_build.log; exit 1; }
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-8B'); s('Qwen/Qwen3-1.7B')" >/dev/null 2>&1
touch /opt/.mirage_done
fi
export MIRAGE_HOME=/opt/mirage
DEMO=/opt/mirage/demo/qwen3/demo.py
LIN=/opt/mirage/include/mirage/persistent_kernel/tasks/hopper/linear_swapAB_hopper.cuh
restore() { cp /opt/mirage.pristine/demo/qwen3/demo.py $DEMO; cp /opt/mirage.pristine/include/mirage/persistent_kernel/tasks/hopper/linear_swapAB_hopper.cuh $LIN; }
apply() {
  restore
  case $1 in *nocap*) sed -i 's|    if size / 96 > 400:|    if False and size / 96 > 400:|' $DEMO; grep -c "if False and size" $DEMO;; esac
  case $1 in *fix*) python /work/syncsan/reproduce/mirage/fix.py $LIN;; esac
}
cd /opt/mirage
step torch-baseline
python $DEMO --save-tokens /work/out/torch.json --max-new-tokens $NTOK --ignore-eos > /work/out/torch.log 2>&1; tail -2 /work/out/torch.log
for v in ${VARIANTS//,/ }; do
  apply $v
  for i in $(seq 1 $REPS); do
    step "$v run $i"
    python $DEMO --use-mirage --save-tokens /work/out/${v}_$i.json --max-new-tokens $NTOK --ignore-eos > /work/out/${v}_$i.log 2>&1; rc=$?; echo "exit $rc"; grep -iE "error|latency|Traceback" /work/out/${v}_$i.log | tail -3
    [ $rc != 0 ] && [ $i = 1 ] && { echo "first run failed; skipping $v"; break; }
  done
done
step compare
python - <<'PY'
import json, glob, os
def toks(p):
    j=json.load(open(p)); 
    for k in ("token_ids","tokens","output_ids","generated_token_ids"):
        if k in j: return j[k]
    return j
ref = toks("/work/out/torch.json") if os.path.exists("/work/out/torch.json") else None
groups = {}
for p in sorted(glob.glob("/work/out/*_[0-9]*.json")):
    groups.setdefault(os.path.basename(p).rsplit("_",1)[0], []).append(toks(p))
for v, runs in groups.items():
    distinct = {json.dumps(r) for r in runs}
    print(f"COMPARE {v}: {len(runs)} runs, {len(distinct)} distinct outputs; first run {'==' if ref is not None and runs[0]==ref else '!='} torch")
PY
if [ "$RUNSAN" = 1 ]; then
  for v in ${SANVARIANTS//,/ }; do
    apply $v; step "syncsan $v (Qwen3-1.7B, $SANTOK tokens, worker_kernel)"
    LD_PRELOAD=$SAN SYNCSAN_VERBOSE=1 SYNCSAN_KERNEL=worker_kernel python $DEMO --use-mirage --model Qwen/Qwen3-1.7B --max-new-tokens $SANTOK --ignore-eos > /work/out/san_$v.log 2>&1
    grep -E "distinct racing|unchecked|WARNING|Error" /work/out/san_$v.log | head -20
  done
fi
