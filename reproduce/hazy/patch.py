"""Turn Hazy's low-latency Llama handoffs into release/acquire pairs, the fix syncsan's reports point to.

    python patch_fences.py DEMO_DIR [fences|acqrel]

fences: uncomment the authors' own `fence.acq_rel.gpu` before each counter increment (release side), moved after
        `store_async_wait()` where it sat before it (attention_partial: a fence before the TMA store's completion doesn't
        order that store), and add one after each spin loop on a counter (acquire side).
acqrel: no fences; each counter increment becomes `atom.add.release.gpu` and each counter poll `ld.acquire.gpu`."""
import pathlib
import re
import sys

FENCE = 'asm volatile("fence.acq_rel.gpu;" ::: "memory");   // syncsan: acquire after the counter wait'


def close_of(src: str, i: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    for j in range(i, len(src)):
        if src[j] == open_ch:
            depth += 1
        elif src[j] == close_ch:
            depth -= 1
            if depth == 0:
                return j
    raise ValueError("unbalanced")


HELPERS = """
#ifndef SYNCSAN_ACQREL_HELPERS   // every op's .cu is #included into llama.cu: define these once
#define SYNCSAN_ACQREL_HELPERS
// syncsan: release/acquire counter operations (hazy/patch_fences.py acqrel)
template <class T> static __device__ __forceinline__ int syncsan_acquire_load(T *p) {
    int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
template <class T> static __device__ __forceinline__ void syncsan_release_add(T *p, int v) {
    asm volatile("red.release.gpu.global.add.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
#endif
"""


def patch_acqrel(src: str) -> tuple[str, int, int]:
    src, released = re.subn(r"\batomicAdd\(\s*&g\.Bar", "syncsan_release_add(&g.Bar", src)
    out, pos, acquired = [], 0, 0
    for m in re.finditer(r"\*\s*\(volatile int \*\)\s*&\s*g", src):
        if m.start() < pos:
            continue
        b = src.index("[", m.end())
        end = close_of(src, b, "[", "]")
        out.append(src[pos:m.start()] + "syncsan_acquire_load(&g" + src[m.end():end + 1] + ")")
        pos, acquired = end + 1, acquired + 1
    out.append(src[pos:])
    src = "".join(out)
    if released or acquired:
        last = list(re.finditer(r"^#include.*$", src, re.M))[-1]
        src = src[:last.end()] + "\n" + HELPERS + src[last.end():]
    return src, released, acquired


def patch(src: str) -> tuple[str, int, int]:
    # attention_partial: the commented fence precedes store_async_wait(); a release must follow the TMA store's completion
    src = re.sub(r'// asm volatile\("fence\.acq_rel\.gpu;"\);(\s*)kittens::tma::store_async_wait\(\);',
                 r'kittens::tma::store_async_wait();\1asm volatile("fence.acq_rel.gpu;");   // syncsan: after the TMA store completes', src)
    src, released = re.subn(r'// (asm volatile\("fence\.acq_rel\.gpu;(?:\\n)?"\);)', r'\1', src)
    released += src.count("syncsan: after the TMA store completes")
    out, pos, acquired = [], 0, 0
    for m in re.finditer(r"while\s*\(", src):
        if m.start() < pos:
            continue
        cond_end = close_of(src, m.end() - 1, "(", ")")
        if "volatile int" not in src[m.start():cond_end]:
            continue
        k = cond_end + 1
        while src[k].isspace():
            k += 1
        end = close_of(src, k, "{", "}") if src[k] == "{" else src.index(";", k)
        out.append(src[pos:end + 1] + "\n" + FENCE + "\n")
        pos, acquired = end + 1, acquired + 1
    out.append(src[pos:])
    return "".join(out), released, acquired


if __name__ == "__main__":
    total = [0, 0]
    fn = patch_acqrel if len(sys.argv) > 2 and sys.argv[2] == "acqrel" else patch
    for f in sorted(pathlib.Path(sys.argv[1]).glob("*.cu")):
        new, r, a = fn(f.read_text())
        if r or a:
            f.write_text(new)
            print(f"{f.name}: {r} release fences uncommented, {a} acquire fences added")
            total[0] += r; total[1] += a
    print(f"total: {total[0]} release, {total[1]} acquire")
