# Lucebox Qwen3.5-0.8B decode megakernel. Args: MODES (comma list of native,syncsan,adversarial,control,stress) FIX (none|acquire|reset|both) NTOK REPS
MODES=${1:-native,stress}; FIX=${2:-none}; NTOK=${3:-16}; REPS=${4:-2000}
. /work/syncsan/reproduce/daytona/common.sh
step lucebox
git clone -q https://github.com/Luce-Org/lucebox.git /opt/luce && (cd /opt/luce && git checkout -q a70ab3c)
MK=/opt/luce/optimizations/megakernel
sed -i 's/"--use_fast_math",/"--use_fast_math", "-lineinfo",/' $MK/setup.py
[ "$FIX" != none ] && python /work/syncsan/reproduce/luce/patch_sync.py $MK/kernel.cu $FIX
$PIPI "transformers==5.9.0" accelerate safetensors huggingface_hub
SMARCH=sm_$ARCH
(cd $MK && CC=gcc CXX=g++ MEGAKERNEL_CUDA_ARCH=$SMARCH TORCH_CUDA_ARCH_LIST=${ARCH:0:-1}.${ARCH: -1} MAX_JOBS=8 pip install -e . --no-build-isolation --no-deps > /work/out/build.log 2>&1; grep -iE ' error|error:|fatal' /work/out/build.log | head; python -c 'import torch, qwen35_megakernel_bf16_C; print("built")')
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.5-0.8B')" >/dev/null 2>&1
D="python /work/syncsan/reproduce/luce/drive.py $NTOK"
T="LD_PRELOAD=$SAN SYNCSAN_VERBOSE=1 SYNCSAN_SHARED=0 SYNCSAN_KERNEL=decode_kernel SYNCSAN_SKIP_OPS==LDG.E.NA.128"
cd /tmp
for m in ${MODES//,/ }; do
  step "$m (fix=$FIX)"
  case $m in
    native) $D /tmp/ref.pt --hf 2>&1 | grep -E "DRIVE|Error"; $D /tmp/n2.pt /tmp/ref.pt 2>&1 | grep -E "DRIVE|Error";;
    syncsan) env $T $D /tmp/s.pt /tmp/ref.pt 2>&1 | grep -E "DRIVE|distinct racing|WARNING|Error";;
    adversarial) env $T SYNCSAN_ADVERSARIAL=1 $D /tmp/a.pt /tmp/ref.pt 2>&1 | grep -E "DRIVE|distinct racing|ADVERSARIAL|Error";;
    control) env $T SYNCSAN_ADVERSARIAL=control $D /tmp/c.pt /tmp/ref.pt 2>&1 | grep -E "DRIVE|distinct racing|ADVERSARIAL|Error";;
    stress) python /work/syncsan/reproduce/luce/stress.py $NTOK $REPS 2>&1 | grep -E "STRESS|Error";;
  esac
done
