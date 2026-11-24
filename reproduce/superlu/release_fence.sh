#!/bin/bash
# Third SuperLU_DIST build ("fixrel"): 587f38f plus a release fence on the producer side. Every atomicSub on an
# fmod/bmod dependency counter in SRC/cuda/pdgstrs_lsum_cuda.cu becomes __threadfence(); atomicSub(...). 587f38f adds
# the fence only after the consumer's atomicAdd(x,0) poll; the producers still publish lsum (atomicAdd) and then
# decrement the counter with a relaxed atomicSub and no fence, which the PTX model does not order.
set -e
f=${1:-SRC/cuda/pdgstrs_lsum_cuda.cu}
perl -pi -e 's/atomicSub\(&([fb]mod)\[/slu_release_sub(&$1\[/g' "$f"
perl -0pi -e 's/(#include <stdio.h>\n)/$1__device__ __forceinline__ int slu_release_sub(int *p, int v) { __threadfence(); return atomicSub(p, v); }\n/' "$f"
grep -c slu_release_sub "$f"
