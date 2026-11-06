"""SV8 kernels. G is 32. Storage is float32.

reload and keep: one thread owns one group.
load-4: one warp owns one group, four groups loaded before the max.
The pieces are load, max, and store on that same schedule.
"""

import tilelang.ascend.language as T


def make_thread_f32(R, C, threads, keep):
    """Float32. One thread owns one group of 32.

    keep: the 32 values stay in registers for the multiply.
    Otherwise the max drops them and the multiply loads X again.
    More threads than groups leaves the extra threads with no group.
    """
    G = 32
    if C % G != 0:
        raise ValueError(f"C={C} must be divisible by {G}")
    groups = R * (C // G)
    if threads < groups and groups % threads != 0:
        raise ValueError(f"groups={groups} must be a positive multiple of threads={threads}")

    @T.prim_func
    def main(
        X: T.Tensor((R * C,), "float32"),
        Out: T.Tensor((R * C,), "float32"),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((R * C,), "float32")
            out_ub = T.alloc_shared((R * C,), "float32")
            T.copy(X, x_ub)
            with T.SimtVF(threads):
                lane = T.get_thread_binding()
                vals = T.alloc_local((G,), "float32")
                m = T.alloc_var("float32", init=0.0)
                if threads > groups:
                    if lane < groups:
                        base = lane * G
                        for t in T.unroll(G):
                            v = x_ub[base + t]
                            if keep:
                                vals[t] = v
                            m = T.max(m, T.max(v, -v))
                        inv = T.float32(1.0) / T.max(m, T.float32(1e-6))
                        for t in T.unroll(G):
                            if keep:
                                x = vals[t]
                            else:
                                x = x_ub[base + t]
                            out_ub[base + t] = x * inv
                else:
                    for step in T.serial(groups // threads):
                        gid = step * threads + lane
                        base = gid * G
                        for t in T.unroll(G):
                            v = x_ub[base + t]
                            if keep:
                                vals[t] = v
                            m = T.max(m, T.max(v, -v))
                        inv = T.float32(1.0) / T.max(m, T.float32(1e-6))
                        for t in T.unroll(G):
                            if keep:
                                x = vals[t]
                            else:
                                x = x_ub[base + t]
                            out_ub[base + t] = x * inv
            T.copy(out_ub, Out)

    return main



def make_warp_f16_unroll(R, C, threads, factor, dtype="float16"):
    """Warp, `factor` groups loaded before their reductions.

    The next loads do not depend on the current warp max. `factor` is 2 or 4.
    `dtype` is the stored type. The max and the multiply stay in float32.
    """
    G = 32
    if dtype not in ("float16", "float32"):
        raise ValueError(f"dtype={dtype}")
    if factor not in (1, 2, 4):
        raise ValueError(f"factor={factor} must be 1, 2, or 4")
    if C % G != 0:
        raise ValueError(f"C={C} must be a multiple of {G}")
    if threads < 32 or threads % 32 != 0 or threads > 2048:
        raise ValueError(f"threads={threads} must be a multiple of 32 and at most 2048")
    groups = R * (C // G)
    warps = threads // 32
    if groups < warps or groups % warps != 0:
        raise ValueError(f"groups={groups} must be a positive multiple of warps={warps}")
    nsteps = groups // warps
    if nsteps % factor != 0:
        raise ValueError(f"steps={nsteps} must be divisible by {factor}")

    @T.prim_func
    def main(
        X: T.Tensor((R * C,), dtype),
        Out: T.Tensor((R * C,), dtype),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((R * C,), dtype)
            out_ub = T.alloc_shared((R * C,), dtype)
            T.copy(X, x_ub)
            with T.SimtVF(threads):
                tid = T.get_thread_binding()
                lane = tid % 32
                warp = tid // 32
                held = T.alloc_local((factor,), "float32")
                inv = T.alloc_local((factor,), "float32")
                for step in T.serial(nsteps // factor):
                    for k in T.unroll(factor):
                        addr = ((step * factor + k) * warps + warp) * G + lane
                        if dtype == "float16":
                            held[k] = T.Cast("float32", x_ub[addr])
                        else:
                            held[k] = x_ub[addr]
                    for k in T.unroll(factor):
                        x = held[k]
                        m = T.warp_reduce_max(T.max(x, -x))
                        inv[k] = T.float32(1.0) / T.max(m, T.float32(1e-6))
                    for k in T.unroll(factor):
                        addr = ((step * factor + k) * warps + warp) * G + lane
                        if dtype == "float16":
                            out_ub[addr] = T.Cast("float16", held[k] * inv[k])
                        else:
                            out_ub[addr] = held[k] * inv[k]
            T.copy(out_ub, Out)

    return main


def make_load4_piece(R, C, threads, piece):
    """One piece of load-4, float32. piece is load, max, or store."""
    G = 32
    factor = 4
    if piece not in ("load", "max", "store"):
        raise ValueError(piece)
    if C % G != 0:
        raise ValueError(f"C={C} must be a multiple of {G}")
    if threads < 32 or threads % 32 != 0 or threads > 2048:
        raise ValueError(f"threads={threads}")
    groups = R * (C // G)
    warps = threads // 32
    if groups < warps or groups % warps != 0:
        raise ValueError(f"groups={groups} warps={warps}")
    nsteps = groups // warps
    if nsteps % factor != 0:
        raise ValueError(f"steps={nsteps}")

    @T.prim_func
    def main(
        X: T.Tensor((R * C,), "float32"),
        Out: T.Tensor((R * C,), "float32"),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((R * C,), "float32")
            out_ub = T.alloc_shared((R * C,), "float32")
            T.copy(X, x_ub)
            with T.SimtVF(threads):
                tid = T.get_thread_binding()
                lane = tid % 32
                warp = tid // 32
                held = T.alloc_local((factor,), "float32")
                for step in T.serial(nsteps // factor):
                    for k in T.unroll(factor):
                        addr = ((step * factor + k) * warps + warp) * G + lane
                        if piece == "store":
                            out_ub[addr] = T.float32(1.0)
                        else:
                            held[k] = x_ub[addr]
                    if piece == "load":
                        for k in T.unroll(factor):
                            addr = ((step * factor + k) * warps + warp) * G + lane
                            out_ub[addr] = held[k]
                    elif piece == "max":
                        for k in T.unroll(factor):
                            addr = ((step * factor + k) * warps + warp) * G + lane
                            x = held[k]
                            m = T.warp_reduce_max(T.max(x, -x))
                            out_ub[addr] = m
            T.copy(out_ub, Out)

    return main

