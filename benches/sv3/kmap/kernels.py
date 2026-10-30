"""SV3 k-mapping. 256 threads.

Thread: each thread owns (rows * col) / 256 cells and walks the whole K of each cell.
Warp: 8 warps. Each warp walks its cells. The 32 lanes split K, then one warp sum.
When K is 8 or 16, only the first K lanes load, and the other lanes add 0.
SIMD: one 64-wide instruction. col must be a multiple of 64.
"""

import tilelang.ascend.language as T

THREADS = 256
WARP = 32


def _check(rows: int, cols: int, K: int) -> None:
    if rows < 1 or rows & (rows - 1):
        raise ValueError(f"rows={rows} must be a power of two and at least 1")
    if cols < 32 or cols & (cols - 1):
        raise ValueError(f"cols={cols} must be a power of two and at least 32")
    if K < 8 or K & (K - 1):
        raise ValueError(f"K={K} must be a power of two and at least 8")


def build_thread(rows: int, cols: int, K: int, threads: int | None = None):
    _check(rows, cols, K)
    cells = rows * cols
    # Fewer cells than the launch: use one thread per cell, a whole number of warps.
    if threads is None:
        threads = THREADS if cells >= THREADS else cells
    if threads < WARP or threads % WARP != 0 or threads > 2048:
        raise ValueError(f"thread launch {threads} must be a multiple of {WARP} and at most 2048")
    if cells < threads or cells % threads != 0:
        raise ValueError(f"rows*cols={cells} is not divisible by {threads} threads")

    @T.prim_func
    def main(
        A: T.Tensor((K, rows, cols), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((rows, cols), "float32"),
    ):
        with T.Kernel(1):
            out_ub = T.alloc_shared((rows, cols), "float32")
            with T.SimtVF(threads):
                acc = T.alloc_fragment((rows, cols), "float32")
                for row, col in T.Parallel(rows, cols):
                    acc[row, col] = T.float32(0)
                for k in T.serial(K):
                    xk = T.alloc_var("float32")
                    xk = T.Cast("float32", X[k])
                    for row, col in T.Parallel(rows, cols):
                        acc[row, col] = acc[row, col] + T.Cast("float32", A[k, row, col]) * xk
                for row, col in T.Parallel(rows, cols):
                    out_ub[row, col] = acc[row, col]
            T.copy(out_ub, Out)

    return main


def build_warp(rows: int, cols: int, K: int):
    _check(rows, cols, K)
    nwarps = THREADS // WARP
    cells = rows * cols
    if cells % nwarps != 0:
        raise ValueError(f"rows*cols={cells} is not divisible by {nwarps} warps")
    per = cells // nwarps
    # K below one warp: the first K lanes own one k each. The rest add 0.
    narrow = K < WARP
    steps = 1 if narrow else K // WARP

    @T.prim_func
    def main(
        A: T.Tensor((K, rows, cols), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((rows, cols), "float32"),
    ):
        with T.Kernel(1):
            out_ub = T.alloc_shared((rows, cols), "float32")
            with T.SimtVF(THREADS):
                tid = T.get_thread_binding()
                warp = tid // WARP
                lane = tid % WARP
                acc = T.alloc_var("float32")
                for c in T.serial(per):
                    cell = warp * per + c
                    row = cell // cols
                    col = cell % cols
                    acc = T.float32(0)
                    if narrow:
                        if lane < K:
                            acc = acc + T.Cast("float32", A[lane, row, col]) * T.Cast("float32", X[lane])
                    else:
                        for t in T.serial(steps):
                            k = lane + t * WARP
                            acc = acc + T.Cast("float32", A[k, row, col]) * T.Cast("float32", X[k])
                    total = T.alloc_var("float32")
                    total = T.warp_reduce_sum(acc)
                    if lane == 0:
                        out_ub[row, col] = total
            T.copy(out_ub, Out)

    return main


def build_simd(rows: int, cols: int, K: int):
    _check(rows, cols, K)
    if cols % 64 != 0:
        raise ValueError(f"cols={cols} must be a multiple of 64 for SIMD")

    @T.prim_func
    def main(
        A: T.Tensor((K, rows, cols), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((rows, cols), "float32"),
    ):
        with T.Kernel(1):
            a_row = T.alloc_shared((K, cols), "float16")
            x_ub = T.alloc_shared((K,), "float16")
            out_row = T.alloc_shared((cols,), "float32")
            T.copy(X, x_ub)
            for row in T.serial(rows):
                for k in T.serial(K):
                    T.copy(A[k, row, :], a_row[k, :])
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    acc = T.simd.alloc_local((1,), "float32")
                    for chunk in T.serial(cols // 64):
                        acc[0] = T.simd.vdup(T.float32(0), "float32")
                        for k in T.serial(K):
                            loaded = T.simd.vld(a_row[k, chunk * 64], dist="UNPK_B16")
                            values = T.simd.vcvt(loaded, "float32", part=0)
                            scale16 = T.simd.vld(x_ub[k], dist="BRC_B16")
                            scale = T.simd.vcvt(scale16, "float32", part=0)
                            T.simd.vmula(acc[0], values, scale, mask)
                        T.simd.vsts(out_row[chunk * 64], acc[0])
                T.copy(out_row, Out[row, :])

    return main
