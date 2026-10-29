"""SV2.A block quant, SIMT. One thread per column.

    scale = max(eps, max over rows of abs(X[row, col]))
    Out[row, col] = X[row, col] * (1 / scale)

The scale stays in the thread. There is no cross-column redux and no broadcast.
C is the thread count and must be a multiple of 32.
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
                for col in T.Parallel(C):
                    m = T.alloc_var("float32")
                    m = T.float32(0)
                    for row in T.serial(R):
                        vv = x_ub[row, col]
                        m = T.max(m, T.max(vv, -vv))
                    inv = T.alloc_var("float32")
                    inv = T.float32(1.0) / T.max(m, T.float32(EPS))
                    for row in T.serial(R):
                        out_ub[row, col] = x_ub[row, col] * inv
            T.copy(out_ub, Out)

    return main
