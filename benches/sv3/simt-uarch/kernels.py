"""Isolated instruction streams for the SV3 microbenchmarks.

A contiguous SIMT load gives each warp 32 adjacent elements.
fp32 is 128 bytes per LDS. fp16 is 64 bytes per LDS.
The stride-64 stream spaces adjacent threads 64 elements apart, which is the GEMV address pattern.
"""

import tilelang.ascend.language as T

N_CONTIG = 32768
REPS = 4
STRIDE = 64
FFMA_ITERS = 128
SIMD_VEC = 256

SPECS = {
    "lds-f32-t32": ("lds", "float32", 32, 1),
    "lds-f32-t256": ("lds", "float32", 256, 1),
    "lds-f32-t1024": ("lds", "float32", 1024, 1),
    "lds-f16-t256": ("lds", "float16", 256, 1),
    "lds-f16-stride-t256": ("lds", "float16", 256, STRIDE),
    "sts-f32-t256": ("sts", "float32", 256, 1),
    "ffma-t256": ("ffma", "float32", 256, 1),
    "simd-f32": ("simd", "float32", 0, 1),
    "simd-f16": ("simd", "float16", 0, 1),
    "lds-ffma-t32": ("lds-ffma", "float32", 32, 1),
    "lds-ffma-sts-t32": ("lds-ffma-sts", "float32", 32, 1),
    "lds-f32-pair-t32": ("lds-pair", "float32", 32, 1),
    "lds-f16-cut-t1024": ("lds-cut", "float16", 1024, 1),
}

WARP = 32
LADDER_STEPS = 128
CUT_THREADS = 1024
CUT_STEPS = 64
CUT_X = 0


def _length(dtype, threads, stride):
    if stride == 1:
        return N_CONTIG
    return threads * stride


def build(name: str):
    op, dtype, threads, stride = SPECS[name]
    if op == "lds":
        return _lds(dtype, threads, stride)
    if op == "sts":
        return _sts(threads)
    if op == "ffma":
        return _ffma(threads)
    if op == "simd":
        return _simd(dtype)
    if op == "lds-ffma":
        return _lds_ffma()
    if op == "lds-ffma-sts":
        return _lds_ffma_sts()
    if op == "lds-pair":
        return _lds_pair()
    if op == "lds-cut":
        return _lds_cut()
    raise KeyError(name)


def _lds(dtype: str, threads: int, stride: int):
    n = _length(dtype, threads, stride)
    steps = n // threads if stride == 1 else stride

    if stride == 1:

        @T.prim_func
        def main(Src: T.Tensor((n,), dtype), Dst: T.Tensor((threads,), "float32")):
            with T.Kernel(1):
                buf = T.alloc_shared((n,), dtype)
                out = T.alloc_shared((threads,), "float32")
                T.copy(Src, buf)
                with T.SimtVF(threads):
                    acc = T.alloc_fragment((threads,), "float32")
                    for t in T.Parallel(threads):
                        acc[t] = T.float32(0)
                    for _rep in T.serial(REPS):
                        for step in T.serial(steps):
                            for t in T.Parallel(threads):
                                acc[t] = acc[t] + T.Cast("float32", buf[step * threads + t])
                    for t in T.Parallel(threads):
                        out[t] = acc[t]
                T.copy(out, Dst)

        return main

    @T.prim_func
    def main(Src: T.Tensor((n,), dtype), Dst: T.Tensor((threads,), "float32")):
        with T.Kernel(1):
            buf = T.alloc_shared((n,), dtype)
            out = T.alloc_shared((threads,), "float32")
            T.copy(Src, buf)
            with T.SimtVF(threads):
                acc = T.alloc_fragment((threads,), "float32")
                for t in T.Parallel(threads):
                    acc[t] = T.float32(0)
                for _rep in T.serial(REPS):
                    for step in T.serial(steps):
                        for t in T.Parallel(threads):
                            acc[t] = acc[t] + T.Cast("float32", buf[t * stride + step])
                for t in T.Parallel(threads):
                    out[t] = acc[t]
            T.copy(out, Dst)

    return main


def _sts(threads: int):
    n = N_CONTIG

    @T.prim_func
    def main(Src: T.Tensor((n,), "float32"), Dst: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            out = T.alloc_shared((n,), "float32")
            T.copy(Src, out)
            with T.SimtVF(threads):
                for step in T.serial(n // threads):
                    for t in T.Parallel(threads):
                        out[step * threads + t] = T.float32(1)
            T.copy(out, Dst)

    return main


def _ffma(threads: int):
    @T.prim_func
    def main(Src: T.Tensor((threads,), "float32"), Dst: T.Tensor((threads,), "float32")):
        with T.Kernel(1):
            x_ub = T.alloc_shared((threads,), "float32")
            out = T.alloc_shared((threads,), "float32")
            T.copy(Src, x_ub)
            with T.SimtVF(threads):
                xv = T.alloc_fragment((threads,), "float32")
                acc = T.alloc_fragment((threads,), "float32")
                for t in T.Parallel(threads):
                    xv[t] = x_ub[t]
                    acc[t] = T.float32(0)
                for _step in T.serial(FFMA_ITERS):
                    for t in T.Parallel(threads):
                        acc[t] = acc[t] + xv[t] * xv[t]
                for t in T.Parallel(threads):
                    out[t] = acc[t]
            T.copy(out, Dst)

    return main


def _lds_ffma():
    threads = WARP
    steps = LADDER_STEPS
    n = threads * steps

    @T.prim_func
    def main(Src: T.Tensor((n,), "float32"), Dst: T.Tensor((threads,), "float32")):
        with T.Kernel(1):
            buf = T.alloc_shared((n,), "float32")
            out = T.alloc_shared((threads,), "float32")
            T.copy(Src, buf)
            with T.SimtVF(threads):
                loaded = T.alloc_fragment((threads,), "float32")
                acc = T.alloc_fragment((threads,), "float32")
                for t in T.Parallel(threads):
                    acc[t] = T.float32(0)
                for step in T.serial(steps):
                    for t in T.Parallel(threads):
                        loaded[t] = buf[step * threads + t]
                    for t in T.Parallel(threads):
                        acc[t] = acc[t] + loaded[t] * loaded[t]
                for t in T.Parallel(threads):
                    out[t] = acc[t]
            T.copy(out, Dst)

    return main


def _lds_ffma_sts():
    threads = WARP
    steps = LADDER_STEPS
    n = threads * steps

    @T.prim_func
    def main(Src: T.Tensor((n,), "float32"), Dst: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            buf = T.alloc_shared((n,), "float32")
            out = T.alloc_shared((n,), "float32")
            T.copy(Src, buf)
            with T.SimtVF(threads):
                loaded = T.alloc_fragment((threads,), "float32")
                acc = T.alloc_fragment((threads,), "float32")
                for t in T.Parallel(threads):
                    acc[t] = T.float32(0)
                for step in T.serial(steps):
                    for t in T.Parallel(threads):
                        loaded[t] = buf[step * threads + t]
                    for t in T.Parallel(threads):
                        acc[t] = acc[t] + loaded[t] * loaded[t]
                    for t in T.Parallel(threads):
                        out[step * threads + t] = acc[t]
            T.copy(out, Dst)

    return main


def _lds_pair():
    threads = WARP
    steps = LADDER_STEPS // 2
    n = threads * steps * 2

    @T.prim_func
    def main(Src: T.Tensor((n,), "float32"), Dst: T.Tensor((threads,), "float32")):
        with T.Kernel(1):
            buf = T.alloc_shared((n,), "float32")
            out = T.alloc_shared((threads,), "float32")
            T.copy(Src, buf)
            with T.SimtVF(threads):
                acc = T.alloc_fragment((threads,), "float32")
                for t in T.Parallel(threads):
                    acc[t] = T.float32(0)
                for step in T.serial(steps):
                    for t in T.Parallel(threads):
                        base = (step * threads + t) * 2
                        acc[t] = acc[t] + buf[base]
                        acc[t] = acc[t] + buf[base + 1]
                for t in T.Parallel(threads):
                    out[t] = acc[t]
            T.copy(out, Dst)

    return main


def _lds_cut():
    """Copy a cut of the N=4096, K=32 fp16 working set into UB, then read it with LDS.

    Each of the 1024 threads reads 64 contiguous halves from the buffer.
    Adjacent lanes read adjacent halves.
    """
    threads = CUT_THREADS
    steps = CUT_STEPS
    n = threads * steps

    @T.prim_func
    def main(Src: T.Tensor((n,), "float16"), Dst: T.Tensor((threads,), "float32")):
        with T.Kernel(1):
            buf = T.alloc_shared((n,), "float16")
            out = T.alloc_shared((threads,), "float32")
            T.copy(Src, buf)
            with T.SimtVF(threads):
                acc = T.alloc_fragment((threads,), "float32")
                for t in T.Parallel(threads):
                    acc[t] = T.float32(0)
                for step in T.serial(steps):
                    for t in T.Parallel(threads):
                        acc[t] = acc[t] + T.Cast("float32", buf[step * threads + t])
                for t in T.Parallel(threads):
                    out[t] = acc[t]
            T.copy(out, Dst)

    return main


def _simd(dtype: str):
    lanes = 2048 // (32 if dtype == "float32" else 16)
    n = lanes * SIMD_VEC

    @T.prim_func
    def main(Src: T.Tensor((n,), dtype), Dst: T.Tensor((lanes,), dtype)):
        with T.Kernel(1):
            buf = T.alloc_shared((n,), dtype)
            out = T.alloc_shared((lanes,), dtype)
            T.copy(Src, buf)
            with T.SimdVF():
                mask = T.simd.pset(32 if dtype == "float32" else 16)
                acc = T.simd.alloc_local((1,), dtype)
                acc[0] = T.simd.vdup(T.float32(0) if dtype == "float32" else T.float16(0), dtype)
                for i in T.serial(SIMD_VEC):
                    acc[0] = T.simd.vadd(acc[0], T.simd.vld(buf[i * lanes]), mask)
                T.simd.vsts(out[0], acc[0], mask)
            T.copy(out, Dst)

    return main
