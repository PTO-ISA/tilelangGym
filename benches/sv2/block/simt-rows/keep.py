"""SIMT block quant that keeps one column in per-thread locals.

The row loop is unrolled into one local float per row, so pass2 can
divide those locals instead of reading X from the buffer again.
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
                acc = T.alloc_reducer((1,), "float32", op="max")
                T.reducer_init(acc)
                for col in T.Parallel(C):
                    m = T.alloc_var("float32")
                    m = T.float32(0)
                    for row in rows:
                        held[row][0] = x_ub[row, col]
                        vv = held[row][0]
                        m = T.max(m, T.max(vv, -vv))
                    T.reducer_update(acc[0], m)
                result = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(acc, result)
                for col in T.Parallel(C):
                    scale = T.max(result[0], T.float32(EPS))
                    for row in rows:
                        out_ub[row, col] = held[row][0] / scale
            T.copy(out_ub, Out)

    return main


def compile_one(rows: int, cols: int) -> None:
    import tilelang

    out = HERE / f"keep-r{rows}c{cols}"
    out.mkdir(parents=True, exist_ok=True)
    compiled = tilelang.compile(
        build(rows, cols),
        target="ascend",
        out_idx=-1,
        execution_backend="cython",
        compile_flags=[],
        pass_configs={"tl.disable_data_race_check": True},
    )
    (out / "kernel.cpp").write_text(compiled.get_kernel_source())
    shutil.copy(compiled.adapter.libpath, out / "kernel.so")
    print(f"OK keep-r{rows}c{cols}", flush=True)


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2:
        print("usage: keep.py R C", flush=True)
        return 2
    compile_one(int(args[0]), int(args[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
