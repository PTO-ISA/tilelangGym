"""SV2.A SIMT block quant. One thread per column, one local per row.

    scale = max(eps, max over rows of abs(X[row, col]))
    Out[row, col] = held[row] * (1 / scale)

The column never leaves the thread, so there is no cross-column redux.
"""

import shutil
import sys
from pathlib import Path

import torch

torch.utils.rename_privateuse1_backend("npu")

import tilelang.ascend.language as T

EPS = 1e-6
HERE = Path(__file__).resolve().parent


def build(R: int, C: int):
    if C % 32 != 0:
        raise ValueError(f"C={C} must be a multiple of 32")
    rows = tuple(range(R))

    @T.prim_func
    def main(X: T.Tensor((R, C), "float32"), Out: T.Tensor((R, C), "float32")):
        with T.Kernel(1):
            x_ub = T.alloc_shared((R, C), "float32")
            out_ub = T.alloc_shared((R, C), "float32")
            T.copy(X, x_ub)
            with T.SimtVF(threads=C):
                held = [T.alloc_var("float32") for _ in range(R)]
                for col in T.Parallel(C):
                    m = T.alloc_var("float32")
                    m = T.float32(0)
                    for row in rows:
                        held[row][0] = x_ub[row, col]
                        vv = held[row][0]
                        m = T.max(m, T.max(vv, -vv))
                    inv = T.alloc_var("float32")
                    inv = T.float32(1.0) / T.max(m, T.float32(EPS))
                    for row in rows:
                        out_ub[row, col] = held[row][0] * inv
            T.copy(out_ub, Out)

    return main


def build_gm(R: int, C: int):
    if C % 32 != 0:
        raise ValueError(f"C={C} must be a multiple of 32")
    rows = tuple(range(R))

    @T.prim_func
    def main(X: T.Tensor((R, C), "float32"), Out: T.Tensor((R, C), "float32")):
        with T.Kernel(1):
            with T.SimtVF(threads=C):
                held = [T.alloc_var("float32") for _ in range(R)]
                for col in T.Parallel(C):
                    m = T.alloc_var("float32")
                    m = T.float32(0)
                    for row in rows:
                        held[row][0] = X[row, col]
                        vv = held[row][0]
                        m = T.max(m, T.max(vv, -vv))
                    inv = T.alloc_var("float32")
                    inv = T.float32(1.0) / T.max(m, T.float32(EPS))
                    for row in rows:
                        Out[row, col] = held[row][0] * inv

    return main


def compile_one(rows: int, cols: int, kind: str = "ub") -> None:
    import tilelang

    name = f"keep-r{rows}c{cols}" if kind == "ub" else f"keep-gm-r{rows}c{cols}"
    out = HERE / name
    out.mkdir(parents=True, exist_ok=True)
    compiled = tilelang.compile(
        build(rows, cols) if kind == "ub" else build_gm(rows, cols),
        target="ascend",
        out_idx=-1,
        execution_backend="cython",
        compile_flags=[],
        pass_configs={"tl.disable_data_race_check": True},
    )
    (out / "kernel.cpp").write_text(compiled.get_kernel_source())
    shutil.copy(compiled.adapter.libpath, out / "kernel.so")
    print(f"OK {name}", flush=True)


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    kind = "ub"
    if args and args[0] == "gm":
        kind = "gm"
        args = args[1:]
    if len(args) != 2:
        print("usage: keep.py [gm] R C", flush=True)
        return 2
    compile_one(int(args[0]), int(args[1]), kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
