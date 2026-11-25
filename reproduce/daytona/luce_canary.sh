# Lucebox native canary witness: original and fixed (both) builds. Args: NTOK R
NTOK=${1:-16}; R=${2:-1000}
. /work/syncsan/reproduce/daytona/common.sh
step lucebox-orig
git clone -q https://github.com/Luce-Org/lucebox.git /opt/luce && (cd /opt/luce && git checkout -q a70ab3c)
MK=/opt/luce/optimizations/megakernel
$PIPI "transformers==5.9.0" accelerate safetensors huggingface_hub > /dev/null 2>&1
build() { (cd $MK && rm -rf build *.so && CC=gcc CXX=g++ MEGAKERNEL_CUDA_ARCH=sm_$ARCH TORCH_CUDA_ARCH_LIST=${ARCH:0:-1}.${ARCH: -1} MAX_JOBS=8 pip install -e . --no-build-isolation --no-deps > /work/out/build_$1.log 2>&1; python -c 'import torch, qwen35_megakernel_bf16_C; print("built")'); }
build orig
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.5-0.8B')" >/dev/null 2>&1
cd /tmp && python /work/syncsan/reproduce/luce/canary.py $NTOK $R 2>&1 | tee /work/out/canary_orig.log | grep -a CANARY
step lucebox-fixed
python /work/syncsan/reproduce/luce/patch_sync.py $MK/kernel.cu both && build fixed
cd /tmp && python /work/syncsan/reproduce/luce/canary.py $NTOK $R 2>&1 | tee /work/out/canary_fixed.log | grep -a CANARY
