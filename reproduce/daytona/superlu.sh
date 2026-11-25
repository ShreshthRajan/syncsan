# SuperLU_DIST's GPU triangular solve under syncsan (H100): builds by commit, each natively, under syncsan, in adversarial
# mode and under the control. Args: BUILDS (comma list of name=commit), default the official fix 587f38f and the
# maintainers' release fix 239280c.
BUILDS=${1:-fix=587f38f6dc14c29c1f6f7f14d68dde3080a081c0,master=239280c103}
. /work/syncsan/reproduce/daytona/common.sh
step superlu-deps
apt-get install -y -qq libopenmpi-dev openmpi-bin libopenblas-dev liblapack-dev > /dev/null 2>&1
git clone -q https://github.com/xiaoyeli/superlu_dist.git /opt/superlu_dist
L=/usr/lib/x86_64-linux-gnu
CM="cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DTPL_ENABLE_CUDALIB=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_CUDA_FLAGS=-lineinfo\ -I$L/openmpi/include\ -I$L/openmpi/include/openmpi -DTPL_ENABLE_PARMETISLIB=OFF -DTPL_ENABLE_INTERNAL_BLASLIB=OFF -DTPL_BLAS_LIBRARIES=$L/libopenblas.so -DTPL_ENABLE_LAPACKLIB=ON -DTPL_LAPACK_LIBRARIES=$L/liblapack.so -DXSDK_ENABLE_Fortran=OFF -Denable_python=OFF -Denable_tests=OFF -Denable_openmp=OFF"
ENV="OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 OMP_NUM_THREADS=1 SUPERLU_ACC_OFFLOAD=1 GPU3DVERSION=1 ANC25D=0 NEW3DSOLVE=1 NEW3DSOLVETREECOMM=1 SUPERLU_ACC_SOLVE=1 SUPERLU_BIND_MPI_GPU=1 SUPERLU_MAXSUP=16 SUPERLU_RELAX=4 SUPERLU_NUM_GPU_STREAMS=1"
cd /tmp && python3 /work/syncsan/reproduce/superlu/gen_mtx.py lap2d 64 lap64.mtx && python3 /work/syncsan/reproduce/superlu/gen_mtx.py lap2d 160 lap160.mtx
T="LD_PRELOAD=$SAN SYNCSAN_VERBOSE=1 SYNCSAN_SHARED=0 SYNCSAN_KERNEL=lsum_"
keep() { grep -aE "Xtrue|rror|distinct racing|ADVERSARIAL|Segmentation" | cut -c1-300 | head -6; }
for bc in ${BUILDS//,/ }; do
  b=${bc%%=*}; c=${bc#*=}
  step "build $b ($c)"
  cp -r /opt/superlu_dist /opt/slu_$b && cd /opt/slu_$b && git checkout -q $c && git log --oneline -1 && eval $CM > cmake.log 2>&1 && cmake --build build -j16 --target pddrive3d > build.log 2>&1 || tail -30 build.log
  cd /opt/slu_$b/build/EXAMPLE
  step "$b native (lap160)"; mpirun --allow-run-as-root -np 1 env $ENV ./pddrive3d -i 0 /tmp/lap160.mtx 2>&1 | keep
  for m in syncsan adversarial control; do
    step "$b $m (lap64)"; E="$T"; [ $m = adversarial ] && E="$T SYNCSAN_ADVERSARIAL=1"; [ $m = control ] && E="$T SYNCSAN_ADVERSARIAL=control"
    mpirun --allow-run-as-root -np 1 env $ENV $E ./pddrive3d -i 0 /tmp/lap64.mtx 2>&1 | tee /work/out/${b}_$m.log | keep
  done
done
