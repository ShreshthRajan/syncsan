"""Test matrices for the SuperLU_DIST GPU trisolve runs (MatrixMarket, general real).

    python gen_mtx.py lap2d N out.mtx        5-point Laplacian on an N x N grid, made nonsymmetric by a convection term
    python gen_mtx.py blockdiag B S out.mtx  B independent diagonally dominant random sparse S x S blocks on the diagonal
                                              (the stacked block-diagonal shape of the 587f38f report)
"""
import random
import sys


def write(path, n, entries):
    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{n} {n} {len(entries)}\n")
        for i, j, v in entries:
            f.write(f"{i + 1} {j + 1} {v:.17g}\n")


def lap2d(N):
    e = []
    for x in range(N):
        for y in range(N):
            i = x * N + y
            e.append((i, i, 4.0))
            for dx, dy, c in ((1, 0, -1.2), (-1, 0, -0.8), (0, 1, -1.1), (0, -1, -0.9)):
                if 0 <= x + dx < N and 0 <= y + dy < N:
                    e.append((i, (x + dx) * N + y + dy, c))
    return N * N, e


def blockdiag(B, S, seed=1):
    rnd, e = random.Random(seed), []
    for b in range(B):
        o = b * S
        for i in range(S):
            cols = {i} | {rnd.randrange(S) for _ in range(4)} | ({i + 1} if i + 1 < S else set()) | ({i - 1} if i else set())
            for j in cols:
                e.append((o + i, o + j, 8.0 + rnd.random() if i == j else rnd.uniform(-1, 1)))
    return B * S, e


if __name__ == "__main__":
    kind = sys.argv[1]
    n, e = lap2d(int(sys.argv[2])) if kind == "lap2d" else blockdiag(int(sys.argv[2]), int(sys.argv[3]))
    write(sys.argv[-1], n, e)
    print(f"{sys.argv[-1]}: n={n} nnz={len(e)}")
