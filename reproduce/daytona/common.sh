# Sourced by the job scripts: tools, Python 3.12 via uv, NVBit 1.8.2, syncsan built for this GPU's arch.
set -o pipefail
export DEBIAN_FRONTEND=noninteractive
step() { echo "=== $1  [$(date +%H:%M:%S)]"; }
step setup
if [ ! -f /opt/.setup_done ]; then
(apt-get update -qq && apt-get install -y -qq git curl bzip2 build-essential >/dev/null) || (sudo apt-get update -qq && sudo apt-get install -y -qq git curl bzip2 build-essential >/dev/null)
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; export PATH=$HOME/.local/bin:$PATH
# Images from pytorch/pytorch ship torch in /opt/conda; use that Python (pypi.nvidia.com, where torch's CUDA wheels
# come from, is outside Daytona's Tier 1-2 allow list).
if [ -x /opt/conda/bin/python ]; then export PATH=/opt/conda/bin:$PATH; PIPI="pip install -q"
elif python3 -c "import torch" 2>/dev/null; then mkdir -p /work/bin && ln -sf "$(command -v python3)" /work/bin/python; export PATH=/work/bin:$PATH PIP_BREAK_SYSTEM_PACKAGES=1; PIPI="pip install -q"
else uv venv -q -p 3.12 /work/venv && . /work/venv/bin/activate; PIPI="uv pip install -q"; fi
python -c "import torch; print('torch', torch.__version__, torch.version.cuda)" 2>/dev/null || echo "no preinstalled torch"
$PIPI "cmake>=3.29" setuptools wheel ninja numpy
mkdir -p /opt/nvbit && curl -sL https://github.com/NVlabs/NVBit/releases/download/v1.8.2/nvbit-Linux-x86_64-1.8.2.tar.bz2 | tar xj -C /opt/nvbit
fi
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
export PATH=$HOME/.local/bin:$PATH
if [ -x /opt/conda/bin/python ]; then export PATH=/opt/conda/bin:$PATH; PIPI="pip install -q"
elif [ -x /work/bin/python ]; then export PATH=/work/bin:$PATH PIP_BREAK_SYSTEM_PACKAGES=1; PIPI="pip install -q"
else . /work/venv/bin/activate; PIPI="uv pip install -q"; fi
touch /opt/.setup_done
(cd /work/syncsan && rm -rf build && cmake -S . -B build -DNVBIT_ROOT=/opt/nvbit/nvbit_release_x86_64 -DNVBIT_TOOL_CUDA_ARCHITECTURES=$ARCH >/dev/null && cmake --build build -j8 2>&1 | tail -1)
SAN=/work/syncsan/build/syncsan.so
ls -la $SAN
