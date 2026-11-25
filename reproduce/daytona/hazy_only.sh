# Hazy's Llama-1B megakernel on this syncsan version (shared-memory checking on), then the known-answer tests and
# cuHadron under syncsan (H100).
. /work/syncsan/reproduce/daytona/common.sh
SANT="LD_PRELOAD=$SAN SYNCSAN_VERBOSE=1"
keep() { grep -aE "distinct racing|ADVERSARIAL|WARNING|Output text|unchecked|untracked" | cut -c1-400 | head -12; }
step "consumer-side case, both models"
nvcc -O2 -arch=sm_$ARCH -lineinfo -o /tmp/rc /work/syncsan/tests/race_cases.cu -lcuda
for wo in 0 1; do echo "-- SYNCSAN_WARP_ORDER=$wo"; SYNCSAN_WARP_ORDER=$wo LD_PRELOAD=$SAN /tmp/rc mp_block_no_consumer_barrier 2>&1 | grep -A3 "syncsan: RACE" | cut -c1-220; done
step hazy-setup
$PIPI "torch==2.7.1" "transformers==4.48.3" "pydra-config>=0.0.13" accelerate tabulate tqdm einops pybind11 psutil art "huggingface_hub[cli]" > /dev/null 2>&1
python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"
git clone -q https://github.com/HazyResearch/Megakernels.git /opt/mk && cd /opt/mk && git checkout -q 7309cec && git submodule update --init --recursive -q && $PIPI -e . --no-deps 2>&1 | tail -2
huggingface-cli download unsloth/Llama-3.2-1B-Instruct --exclude '*.pth' 'original/*' > /dev/null 2>&1
PYLIB=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"); export LIBRARY_PATH=$PYLIB:${LIBRARY_PATH:-} LD_LIBRARY_PATH=$PYLIB:${LD_LIBRARY_PATH:-}; echo "libpython in $PYLIB"
PV=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
EXT=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
# without python3-config the Makefile names the module mk_llama with no extension suffix; give it the suffix
build() { cd /opt/mk/demos/$1 && THUNDERKITTENS_ROOT=/opt/mk/ThunderKittens MEGAKERNELS_ROOT=/opt/mk PYTHON_VERSION=$PV GPU=H100 make > /work/out/build_$1.log 2>&1; [ -f mk_llama ] && mv mk_llama mk_llama$EXT; ls /opt/mk/demos/$1/*.so 2>/dev/null | head -1; }
build low-latency-llama
cp -r /opt/mk/demos/low-latency-llama /opt/mk/demos/acqrel && rm -f /opt/mk/demos/acqrel/*.so && python /work/syncsan/reproduce/hazy/patch.py /opt/mk/demos/acqrel acqrel && build acqrel
H="python megakernels/scripts/generate.py mode=mk model=unsloth/Llama-3.2-1B-Instruct ntok=8 num_warmup=0 num_iters=1"
HT="$SANT SYNCSAN_KERNEL=megakernel::mk"
for m in native syncsan adversarial control fixed_syncsan fixed_adversarial; do
  step "hazy $m"
  case $m in native) E="";; syncsan) E="$HT";; adversarial) E="$HT SYNCSAN_ADVERSARIAL=1";; control) E="$HT SYNCSAN_ADVERSARIAL=control";;
    fixed_syncsan) E="$HT";; fixed_adversarial) E="$HT SYNCSAN_ADVERSARIAL=1";; esac
  D=""; case $m in fixed_*) D="mk_dir=/opt/mk/demos/acqrel";; esac
  (cd /opt/mk && timeout 1500 env $E $H $D 2>&1 | tee /work/out/hazy_$m.log | keep)
done

