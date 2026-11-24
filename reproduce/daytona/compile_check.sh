# Compile syncsan and the test programs without a GPU (nvcc only), to catch build errors cheaply.
set -o pipefail
apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq curl bzip2 cmake >/dev/null 2>&1
curl -sL https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz | tar xz -C /opt && export PATH=/opt/cmake-3.31.6-linux-x86_64/bin:$PATH
mkdir -p /opt/nvbit && curl -sL https://github.com/NVlabs/NVBit/releases/download/v1.8.2/nvbit-Linux-x86_64-1.8.2.tar.bz2 | tar xj -C /opt/nvbit
cd /work/syncsan && cmake --version | head -1
cmake -S . -B build -DNVBIT_ROOT=/opt/nvbit/nvbit_release_x86_64 -DNVBIT_TOOL_CUDA_ARCHITECTURES=90 > /work/out/cmake.log 2>&1 && cmake --build build -j4 2>&1 | grep -E "error|warning|Built target" | head -40; tail -5 /work/out/cmake.log
for f in tests/race_cases.cu tests/concurrent_streams.cu tests/mirage_norm.cu; do nvcc -arch=sm_90 -lineinfo -o /tmp/t $f -lcuda > /tmp/c.log 2>&1 && echo "compiled $f" || { echo "FAILED $f"; head -20 /tmp/c.log; }; done
