"""SV2 block quant, SIMT. One thread per column.

    m = max over rows of abs(X[row, col])
    scale = max(eps, max over columns of m)
    Out[row, col] = X[row, col] / scale

C is the thread count and must be a multiple of 32, so a warp covers 32
adjacent columns of one row.
"""

import tilelang.ascend.language as T

EPS = 1e-6


def build(R: int, C: int):
    if C % 32 != 0:
        raise ValueError(f"C={C} must be a multiple of 32")

    @T.prim_func
    def main(X: T.Tensor((R, C), "float32"), Out: T.Tensor((R, C), "float32")):
        with T.Kernel(1):
            x_ub = T.alloc_shared((R, C), "float32")
            out_ub = T.alloc_shared((R, C), "float32")
            T.copy(X, x_ub)
            with T.SimtVF(threads=C):
                acc = T.alloc_reducer((1,), "float32", op="max")
                T.reducer_init(acc)
                for col in T.Parallel(C):
                    m = T.alloc_var("float32")
                    m = T.float32(0)
                    for row in T.serial(R):
                        vv = x_ub[row, col]
                        m = T.max(m, T.max(vv, -vv))
                    T.reducer_update(acc[0], m)
                result = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(acc, result)
                for col in T.Parallel(C):
                    scale = T.max(result[0], T.float32(EPS))
                    for row in T.serial(R):
                        out_ub[row, col] = x_ub[row, col] / scale
            T.copy(out_ub, Out)

    return main
