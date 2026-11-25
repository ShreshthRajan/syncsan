# Rerun Hazy's Llama-1B megakernel and AutoMegaKernel on this syncsan version with shared-memory checking on (H100).
. /work/syncsan/reproduce/daytona/common.sh
SANT="LD_PRELOAD=$SAN SYNCSAN_VERBOSE=1"
keep() { grep -aE "distinct racing|ADVERSARIAL|WARNING|Output text|verif|Verif|VERIF|PASS|FAIL|match|unchecked|untracked" | cut -c1-400 | head -12; }

step "regression: the old behavior misses the block-wide cases"
cd /work/syncsan && nvcc -O2 -arch=sm_$ARCH -lineinfo -o /tmp/rc tests/race_cases.cu -lcuda
for c in mp_block_no_producer_barrier mp_block_no_consumer_barrier; do
  echo "$c old: $(SYNCSAN_WARP_ORDER=0 LD_PRELOAD=$SAN /tmp/rc $c 2>&1 | grep -c 'syncsan: RACE') new: $(LD_PRELOAD=$SAN /tmp/rc $c 2>&1 | grep -c 'syncsan: RACE')"
done

step amk-setup
git clone -q https://github.com/RightNow-AI/AutoMegaKernel.git /opt/amk && cd /opt/amk && git checkout -q 884534c && timeout 900 uv sync -q 2>&1 | tail -3
G=".venv/bin/python amk_cli.py generate toy --gpu h100 --prompt-ids 1,2,3 --max-tokens 16 --verify"
for m in native syncsan adversarial control; do
  step "amk $m"
  case $m in native) E="";; syncsan) E="$SANT";; adversarial) E="$SANT SYNCSAN_ADVERSARIAL=1";; control) E="$SANT SYNCSAN_ADVERSARIAL=control";; esac
  (cd /opt/amk && export PATH=/opt/amk/.venv/bin:$PATH && timeout 1200 env $E $G 2>&1 | tee /work/out/amk_$m.log | keep)
done
step "amk acquire fix"
cd /opt/amk && python - <<'PY'
import pathlib; p = pathlib.Path('vm/sync.cuh'); s = p.read_text()
s2 = s.replace('    return atomicAdd(c, 0u);', '    uint32_t v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(c) : "memory"); return v;')
assert s2 != s; p.write_text(s2); print('patched')
PY
rm -rf ~/.cache/torch_extensions
for m in syncsan adversarial; do
  step "amk fixed $m"; E="$SANT"; [ $m = adversarial ] && E="$SANT SYNCSAN_ADVERSARIAL=1"
  (cd /opt/amk && export PATH=/opt/amk/.venv/bin:$PATH && timeout 1200 env $E $G 2>&1 | tee /work/out/amk_fixed_$m.log | keep)
done

step hazy-setup
$PIPI "torch==2.7.1" "transformers==4.48.3" "pydra-config>=0.0.13" accelerate tabulate tqdm einops pybind11 psutil art "huggingface_hub[cli]" > /dev/null 2>&1
python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"
git clone -q https://github.com/HazyResearch/Megakernels.git /opt/mk && cd /opt/mk && git checkout -q 7309cec && git submodule update --init --recursive -q && $PIPI -e . --no-deps 2>&1 | tail -2
huggingface-cli download unsloth/Llama-3.2-1B-Instruct --exclude '*.pth' 'original/*' > /dev/null 2>&1
PV=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
build() { cd /opt/mk/demos/$1 && THUNDERKITTENS_ROOT=/opt/mk/ThunderKittens MEGAKERNELS_ROOT=/opt/mk PYTHON_VERSION=$PV GPU=H100 make > /work/out/build_$1.log 2>&1; ls /opt/mk/demos/$1/*.so 2>/dev/null | head -1; }
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
