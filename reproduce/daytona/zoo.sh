# megakernel-zoo Qwen3-8B offline-a (H100 only). Args: MODES FIX (none|acquire) NTOK REPS
MODES=${1:-native,stress}; FIX=${2:-none}; NTOK=${3:-16}; REPS=${4:-2000}
. /work/syncsan/reproduce/daytona/common.sh
step zoo
git clone -q https://github.com/dangxingyu/megakernel-zoo.git /opt/zoo && (cd /opt/zoo && git checkout -q 95e315c)
ENGINE=/opt/zoo/engines/qwen3-8b_h100/offline-a/engine
[ "$FIX" != none ] && python /work/syncsan/reproduce/zoo/patch_sync.py $ENGINE/mk.cu $FIX
$PIPI "transformers==4.57.1" accelerate safetensors huggingface_hub
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-8B', local_dir='/models/qwen3-8b', allow_patterns=['*.json','*.safetensors','*.txt','merges.txt','vocab.json'])" >/dev/null 2>&1
(cd $ENGINE && rm -f libmk.so && MK_DEFS=-lineinfo bash build.sh 2>&1 | grep -iE 'error' | head; ls -la libmk.so)
D="python /work/syncsan/reproduce/zoo/drive.py $ENGINE /models/qwen3-8b $NTOK"
T="LD_PRELOAD=$SAN SYNCSAN_VERBOSE=1 SYNCSAN_SHARED=0 SYNCSAN_KERNEL=k_step SYNCSAN_SKIP_OPS==LDG.E.LU.128"
cd /tmp
for m in ${MODES//,/ }; do
  step "$m (fix=$FIX)"
  case $m in
    native) env DRIVE_HF=1 $D /tmp/ref.pt 2>&1 | grep -E "DRIVE|Error"; $D /tmp/n2.pt /tmp/ref.pt 2>&1 | grep -E "DRIVE|Error";;
    syncsan) env $T $D /tmp/s.pt /tmp/ref.pt 2>&1 | grep -E "DRIVE|distinct racing|WARNING|Error";;
    adversarial) env $T SYNCSAN_ADVERSARIAL=1 $D /tmp/a.pt /tmp/ref.pt 2>&1 | grep -E "DRIVE|distinct racing|ADVERSARIAL|Error";;
    control) env $T SYNCSAN_ADVERSARIAL=control $D /tmp/c.pt /tmp/ref.pt 2>&1 | grep -E "DRIVE|distinct racing|ADVERSARIAL|Error";;
    stress) python /work/syncsan/reproduce/zoo/stress.py $ENGINE /models/qwen3-8b $NTOK $REPS 2>&1 | grep -E "STRESS|Error";;
  esac
done
