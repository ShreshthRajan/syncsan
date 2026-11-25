# An incomplete fix in SuperLU_DIST's GPU triangular solve

SuperLU_DIST's GPU triangular solve hands partial sums between thread blocks with a counter. Commit 4755a5e (the HIP
port) dropped the consumer's `__threadfence()` after its `atomicAdd(&fmod, 0)` poll, which gave wrong solutions on
hardware; commit 587f38f put the fence back. syncsan finds the original bug, and finds that the fix is incomplete under
the PTX memory model: the producer still publishes with a relaxed `atomicSub` and no fence. Results are for a full
SuperLU_DIST build (CUDA, one MPI rank, `pddrive3d -i 0`, GPU solve) on an H100, with small supernodes (`SUPERLU_RELAX=4
SUPERLU_MAXSUP=16`) so that the solve has many dependencies between blocks.

## The handoff

A producer block adds its contribution into `lsum` with `d_atomicAdd` (`pdgstrs_lsum_cuda.cu:2457`, `:3277`) and then
decrements the consumer's counter with `atomicSub`. The consumer spins with `atomicAdd(&fmod[..], 0)` (`:2320`, `:3132`)
and then reads `lsum` (`x[..] += lsum[..]` at `:2333`, `:3147`; `lsum[il+i] = temp1` at `:2349`, `:3167`). For the
consumer's reads to be ordered after the producer's additions, the producer needs a release before the `atomicSub` and
the consumer an acquire after the poll. 587f38f added the acquire side only.

## Confirmed and fixed upstream

The SuperLU_DIST developers confirmed the missing release in
[xiaoyeli/superlu_dist#261](https://github.com/xiaoyeli/superlu_dist/issues/261) and fixed it in 239280c (Nov 24, 2026):
a `__threadfence()` before each producer's `atomicSub` on `fmod` and `bmod`, in `pdgstrs_lsum_cuda.cu` and its single and
complex counterparts, and a warp synchronization before the fence where one lane decremented for the whole warp. On
239280c, syncsan 0.10 reports 0 racing pairs, the adversarial solve is correct (error 6.2e-15) and so is the control
(`results/superlu_239280c_20261124`, `reproduce/daytona/superlu.sh`). On 587f38f in the same run: 4 pairs, adversarial
error 0.74, control 5.0e-15.

## Results

| Build | Racing pairs | Native error | Adversarial error | Control error |
|---|---|---|---|---|
| 4755a5e (consumer fence missing) | 4 | 0.08 to 0.41 | 0.63 to 0.64 | correct |
| 587f38f (official fix) | 4 | correct (about 1e-14) | 0.68 to 0.71 | correct |
| 587f38f with `__threadfence()` before each producer `atomicSub` | 0 | correct | correct | correct |

Error is `||X - Xtrue|| / ||X||` as printed by the driver, from two separate runs of each case. The official fix is
correct natively in our H100 runs; the adversarial run shows that its result depends on an ordering PTX does not
promise. With the default supernode settings every build is correct and no handoff between blocks occurs in the test
matrices. [reproduce/modal_superlu.py](../reproduce/modal_superlu.py) reruns the table;
[reproduce/superlu/release_fence.sh](../reproduce/superlu/release_fence.sh) is the producer fence.
