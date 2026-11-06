"""SV3 SIMT operators on A[N, K]. N = 4096, K = 32, 256 threads.

load sums A along K. ffma is the gemv. sts stores the constant 1.
"""

import shutil
import sys
from pathlib import Path

import torch

torch.utils.rename_privateuse1_backend("npu")

import tilelang
import tilelang.ascend.language as T

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs" / "ops"
N, K, THREADS = 4096, 32, 256
FIX = str(HERE.parents[2] / "bisheng_libstdcxx_clang_fix.h")
NAMES = ("load", "ffma", "sts")


def build_load():
    @T.prim_func
    def main(
        A: T.Tensor((N, K), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            out_ub = T.alloc_shared((N,), "float32")
            with T.SimtVF(THREADS):
                acc = T.alloc_fragment((N,), "float32")
                for n in T.Parallel(N):
                    acc[n] = T.float32(0)
                for k in T.serial(K):
                    for n in T.Parallel(N):
                        acc[n] = acc[n] + T.Cast("float32", A[n, k])
                for n in T.Parallel(N):
                    out_ub[n] = acc[n]
            T.copy(out_ub, Out)

    return main


def build_ffma():
    @T.prim_func
    def main(
        A: T.Tensor((N, K), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            out_ub = T.alloc_shared((N,), "float32")
            with T.SimtVF(THREADS):
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


def build_sts():
    @T.prim_func
    def main(
        A: T.Tensor((N, K), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            out_ub = T.alloc_shared((N,), "float32")
            with T.SimtVF(THREADS):
                for n in T.Parallel(N):
                    out_ub[n] = T.float32(1)
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
        compile_flags=["-include", FIX, "-isystem", "/usr/include/c++/12"],
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
