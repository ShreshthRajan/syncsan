# Prior work

syncsan combines two older ideas, happens-before race detection and adversarial memory, and applies them to GPU binaries
under the PTX memory model with Hopper and Blackwell async copies included.

**Adversarial memory.** Jumble (Flanagan and Freund, PLDI 2010) pairs a happens-before race detector with a memory that
returns older values the Java memory model allows, to show which races change a program's result. tsan11 (Lidbury and
Donaldson, POPL 2017), C11Tester and GenMC explore weak behaviours of C++11 atomics; they inject only on atomics,
because a racy plain load is undefined behaviour in C++11. PTX does not require data-race freedom (Lustig et al., ASPLOS
2019), so syncsan can inject on plain loads. On GPUs, Bugaroo (ISSRE 2018) delays stores in a binary recompiled through
SASSI and can return values the model forbids. Sorensen and Donaldson (PLDI 2016) exposed weak behaviours in GPU
applications by stressing the memory system, which shows only what the hardware happens to do.

**GPU race detection.** Barracuda (PLDI 2017) did happens-before race detection on CUDA binaries with fences and
atomics, and CURD (PLDI 2018) made it fast. iGUARD (SOSP 2021) detects races in global memory with scoped
synchronization under NVBit; its public build does not run on sm_90. ScoRD (ISCA 2020) proposed hardware support for
scoped races, HiRace (SC 2024) instruments source, and GPUVerify, GKLEE and Faial check kernels statically. Alglave et
al. (ASPLOS 2015) observed weak-memory behaviors on NVIDIA GPUs, and Dartagnan checks programs against a formal PTX
model. Triton's GSan (2026) is a happens-before detector for global memory with TMA, mbarriers and scoped atomics, for
programs written in Triton. compute-sanitizer racecheck and Triton's ConSan check shared memory. SuperCollider (PLDI
2026) instruments through ptxas, delays reads and reports a race when a value changes during the delay; it does not use
synchronization operations, has no false positives, and detects all but 3 of the 27 cuHadron tests its authors wrote,
including the cross-kernel and unified-memory ones syncsan does not check. RESOLVE (arXiv 2610.05683) perturbs timing in
megakernels under NVBit to expose races, without a happens-before relation. CUTracer injects delays.

**Formal verification.** TensorLean (Columbia and CertiK, November 2026) proves data-race freedom, termination and
functional correctness of PTX kernels against a tensor-level specification written in Lean for each kernel. It checked
149 operator kernels from FlashAttention, DeepGEMM, Liger and others and reports 38 bugs, including a FlashAttention-4
data race that 17,760 test runs missed. SF Tensor's PTX model in Lean (arXiv 2611.04447) proves smaller kernels correct,
including acquire-release and async-proxy patterns. Both need a specification per kernel; syncsan needs none and runs
whole persistent megakernels, at the cost of checking only the executions it observes.

**What syncsan adds.** Binary happens-before checking with Hopper and Blackwell async copies modeled (the copy engine
for TMA, bulk copies and `cp.async`, in global and shared memory, with TMA coordinates decoded from tensor maps), and a
stale-value mode that turns the races it finds into outputs the PTX model allows, under the conditions in
[how-it-works.md](how-it-works.md).
