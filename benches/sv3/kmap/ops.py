"""SV3 SIMT operators as separate kernels.

Same case as the head-to-head IPC charts: 32×128, K=32, 256 threads. N = 4096.

`load` sums A along K so the loads stay. `ffma` is the full gemv.
`sts` stores the constant 1 and does not load.
"""

import shutil
import sys
from pathlib import Path

import torch

torch.utils.rename_privateuse1_backend("npu")

import tilelang
import tilelang.ascend.language as T

HERE = Path(__file__).resolve().parent
OUT = HERE / "ops"
ROWS, COLS, K, THREADS = 32, 128, 32, 256
NAMES = ("load", "ffma", "sts")


def _check() -> None:
    cells = ROWS * COLS
    if cells % THREADS != 0:
        raise ValueError(f"N={cells} is not divisible by {THREADS}")


def build_load():
    _check()

    @T.prim_func
    def main(
        A: T.Tensor((K, ROWS, COLS), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((ROWS, COLS), "float32"),
    ):
        with T.Kernel(1):
            out_ub = T.alloc_shared((ROWS, COLS), "float32")
            with T.SimtVF(THREADS):
                acc = T.alloc_fragment((ROWS, COLS), "float32")
                for row, col in T.Parallel(ROWS, COLS):
                    acc[row, col] = T.float32(0)
                for k in T.serial(K):
                    for row, col in T.Parallel(ROWS, COLS):
                        acc[row, col] = acc[row, col] + T.Cast("float32", A[k, row, col])
                for row, col in T.Parallel(ROWS, COLS):
                    out_ub[row, col] = acc[row, col]
            T.copy(out_ub, Out)

    return main


def build_ffma():
    _check()

    @T.prim_func
    def main(
        A: T.Tensor((K, ROWS, COLS), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((ROWS, COLS), "float32"),
    ):
        with T.Kernel(1):
            out_ub = T.alloc_shared((ROWS, COLS), "float32")
            with T.SimtVF(THREADS):
                acc = T.alloc_fragment((ROWS, COLS), "float32")
                for row, col in T.Parallel(ROWS, COLS):
                    acc[row, col] = T.float32(0)
                for k in T.serial(K):
                    xk = T.alloc_var("float32")
                    xk = T.Cast("float32", X[k])
                    for row, col in T.Parallel(ROWS, COLS):
                        acc[row, col] = acc[row, col] + T.Cast("float32", A[k, row, col]) * xk
                for row, col in T.Parallel(ROWS, COLS):
                    out_ub[row, col] = acc[row, col]
            T.copy(out_ub, Out)

    return main


def build_sts():
    _check()

    @T.prim_func
    def main(
        A: T.Tensor((K, ROWS, COLS), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((ROWS, COLS), "float32"),
    ):
        with T.Kernel(1):
            out_ub = T.alloc_shared((ROWS, COLS), "float32")
            with T.SimtVF(THREADS):
                for row, col in T.Parallel(ROWS, COLS):
                    out_ub[row, col] = T.float32(1)
            T.copy(out_ub, Out)

    return main


BUILDERS = {"load": build_load, "ffma": build_ffma, "sts": build_sts}


def compile_one(name: str) -> Path:
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)
    compiled = tilelang.compile(
        BUILDERS[name](),
        target="ascend",
        out_idx=-1,
        execution_backend="cython",
        compile_flags=[],
        pass_configs={"tl.disable_data_race_check": True},
    )
    (dest / "kernel.cpp").write_text(compiled.get_kernel_source())
    shutil.copy(compiled.adapter.libpath, dest / "kernel.so")
    print(f"OK ops/{name}", flush=True)
    return dest


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args != ["compile"]:
        print("usage: ops.py compile", flush=True)
        return 2
    for name in NAMES:
        compile_one(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
