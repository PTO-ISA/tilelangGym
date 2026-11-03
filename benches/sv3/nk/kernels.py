"""SV3 with K contiguous. Out[n] = sum_k A[n, k] * x[k]. N is the number of outputs.

Thread: each thread owns N / threads outputs and walks that output's K.
Warp: 256 threads, 8 warps. The lanes split K, then one warp sum.
SIMD: one 64-wide load along K. K at most 64 covers 64/K outputs. K = 128 is two loads of one output.
"""

import tilelang.ascend.language as T

THREADS = 256
WARP = 32
UB_BYTES = 248 * 1024


def _check(N: int, K: int) -> None:
    if N < 32 or N & (N - 1):
        raise ValueError(f"N={N} must be a power of two and at least 32")
    if K < 8 or K & (K - 1):
        raise ValueError(f"K={K} must be a power of two and at least 8")


def _threads(N: int, threads: int | None) -> int:
    if threads is None:
        threads = THREADS if N >= THREADS else N
    if threads < WARP or threads % WARP != 0 or threads > 2048:
        raise ValueError(f"thread launch {threads} must be a multiple of {WARP} and at most 2048")
    if N < threads or N % threads != 0:
        raise ValueError(f"N={N} is not divisible by {threads} threads")
    return threads


def build_thread(N: int, K: int, threads: int | None = None):
    _check(N, K)
    threads = _threads(N, threads)

    @T.prim_func
    def main(
        A: T.Tensor((N, K), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            out_ub = T.alloc_shared((N,), "float32")
            with T.SimtVF(threads):
                acc = T.alloc_fragment((N,), "float32")
                for n in T.Parallel(N):
                    acc[n] = T.float32(0)
                for k in T.serial(K):
                    xk = T.alloc_var("float32")
                    xk = T.Cast("float32", X[k])
                    for n in T.Parallel(N):
                        acc[n] = acc[n] + T.Cast("float32", A[n, k]) * xk
                for n in T.Parallel(N):
                    out_ub[n] = acc[n]
            T.copy(out_ub, Out)

    return main


def _slab(N: int, K: int, threads: int) -> int:
    x_bytes = K * 2
    out_bytes = N * 4
    full = N * K * 2 + x_bytes + out_bytes
    if full <= UB_BYTES:
        return N
    budget = UB_BYTES - x_bytes - out_bytes
    one = K * 2
    if budget < one * threads:
        raise ValueError(f"a slab of {threads} outputs does not fit in {UB_BYTES} bytes")
    slab = threads
    while slab * 2 <= N and (slab * 2) * one <= budget:
        slab *= 2
    return slab


def build_thread_ub(N: int, K: int, threads: int | None = None):
    _check(N, K)
    threads = _threads(N, threads)
    slab = _slab(N, K, threads)

    @T.prim_func
    def main(
        A: T.Tensor((N, K), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((slab, K), "float16")
            x_ub = T.alloc_shared((K,), "float16")
            out_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            for s in T.serial(N // slab):
                T.copy(A[s * slab : (s + 1) * slab, :], a_ub)
                with T.SimtVF(threads):
                    acc = T.alloc_fragment((slab,), "float32")
                    for n in T.Parallel(slab):
                        acc[n] = T.float32(0)
                    for k in T.serial(K):
                        xk = T.alloc_var("float32")
                        xk = T.Cast("float32", x_ub[k])
                        for n in T.Parallel(slab):
                            acc[n] = acc[n] + T.Cast("float32", a_ub[n, k]) * xk
                    for n in T.Parallel(slab):
                        out_ub[s * slab + n] = acc[n]
            T.copy(out_ub, Out)

    return main


def build_warp(N: int, K: int):
    _check(N, K)
    nwarps = THREADS // WARP
    if N % nwarps != 0:
        raise ValueError(f"N={N} is not divisible by {nwarps} warps")
    per = N // nwarps
    narrow = K < WARP
    steps = 1 if narrow else K // WARP

    @T.prim_func
    def main(
        A: T.Tensor((N, K), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            out_ub = T.alloc_shared((N,), "float32")
            with T.SimtVF(THREADS):
                tid = T.get_thread_binding()
                warp = tid // WARP
                lane = tid % WARP
                acc = T.alloc_var("float32")
                for c in T.serial(per):
                    n = warp * per + c
                    acc = T.float32(0)
                    if narrow:
                        if lane < K:
                            acc = acc + T.Cast("float32", A[n, lane]) * T.Cast("float32", X[lane])
                    else:
                        for t in T.serial(steps):
                            k = lane + t * WARP
                            acc = acc + T.Cast("float32", A[n, k]) * T.Cast("float32", X[k])
                    total = T.alloc_var("float32")
                    total = T.warp_reduce_sum(acc)
                    if lane == 0:
                        out_ub[n] = total
            T.copy(out_ub, Out)

    return main


def build_simd(N: int, K: int):
    _check(N, K)
    if K <= 64:
        if 64 % K != 0:
            raise ValueError(f"K={K} must divide 64")
        width = 64
        per = 64 // K
        if N % per != 0:
            raise ValueError(f"N={N} is not divisible by {per}")
        chunks = 1
    else:
        if K % 64 != 0:
            raise ValueError(f"K={K} must be a multiple of 64")
        width = 64
        per = 1
        chunks = K // 64

    rounds = 0
    w = width
    while w > per:
        w //= 2
        rounds += 1
    x_bytes = K * 2
    out_bytes = N * 4
    xs_bytes = width * 2
    full = N * K * 2 + x_bytes + out_bytes + xs_bytes
    if full <= UB_BYTES:
        slab = N
    else:
        budget = UB_BYTES - x_bytes - out_bytes - xs_bytes
        slab = per
        while slab * 2 <= N and (slab * 2) * K * 2 <= budget:
            slab *= 2

    @T.prim_func
    def main(
        A: T.Tensor((N, K), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((slab, K), "float16")
            x_ub = T.alloc_shared((K,), "float16")
            xs = T.alloc_shared((width,), "float16")
            out_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            if chunks == 1:
                for i in T.serial(width):
                    xs[i] = x_ub[i % K]
            for s in T.serial(N // slab):
                T.copy(A[s * slab : (s + 1) * slab, :], a_ub)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    store_mask = T.simd.pset(32, f"PAT_VL{per}")
                    scale = T.simd.vdup(T.float32(0), "float32")
                    if chunks == 1:
                        scale16 = T.simd.vld(xs[0], dist="UNPK_B16")
                        scale = T.simd.vcvt(scale16, "float32", part=0)
                    for g in T.serial(slab // per):
                        prod = T.simd.alloc_local((1,), "float32")
                        prod[0] = T.simd.vdup(T.float32(0), "float32")
                        for c in T.serial(chunks):
                            loaded = T.simd.vld(a_ub[g * per, c * width], dist="UNPK_B16")
                            values = T.simd.vcvt(loaded, "float32", part=0)
                            if chunks != 1:
                                scale16 = T.simd.vld(x_ub[c * width], dist="UNPK_B16")
                                scale = T.simd.vcvt(scale16, "float32", part=0)
                            part = T.simd.alloc_local((1,), "float32")
                            part[0] = T.simd.vmul(values, scale, mask)
                            for _step in T.unroll(rounds):
                                part[0] = T.simd.vcpadd(part[0], mask)
                            prod[0] = T.simd.vadd(prod[0], part[0], mask)
                        T.simd.vsts(out_ub[s * slab + g * per], prod[0], store_mask)
            T.copy(out_ub, Out)

    return main
