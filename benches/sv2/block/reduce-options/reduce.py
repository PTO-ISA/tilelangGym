"""SV2 reduce sweep. Absmax of N fp32, one scale out.

cross: threads = N, one element per thread, cross-thread then cross-warp.
loop:  threads = 128, each thread loops over N/128 elements, then the same 4-warp redux.
"""

import shutil
import sys
from pathlib import Path

import torch

torch.utils.rename_privateuse1_backend("npu")

import tilelang.ascend.language as T

EPS = 1e-6
LOOP_THREADS = 128
SIZES = (128, 256, 512, 1024, 2048)
HERE = Path(__file__).resolve().parent


def build(n: int, threads: int):
    if n % threads != 0:
        raise ValueError(f"threads={threads} must divide N={n}")
    reps = n // threads

    @T.prim_func
    def main(X: T.Tensor((n,), "float32"), Scale: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            x_ub = T.alloc_shared((n,), "float32")
            scale_ub = T.alloc_shared((1,), "float32")
            T.copy(X, x_ub)
            with T.SimtVF(threads=threads):
                acc = T.alloc_reducer((1,), "float32", op="max")
                T.reducer_init(acc)
                for lane in T.Parallel(threads):
                    m = T.alloc_var("float32")
                    m = T.float32(0)
                    for k in T.serial(reps):
                        vv = x_ub[lane + k * threads]
                        m = T.max(m, T.max(vv, -vv))
                    T.reducer_update(acc[0], m)
                result = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(acc, result)
                if T.get_thread_binding() == 0:
                    scale_ub[0] = T.max(result[0], T.float32(EPS))
            T.copy(scale_ub, Scale)

    return main


def compile_one(name: str, n: int, threads: int) -> None:
    import tilelang

    out = HERE / name
    out.mkdir(parents=True, exist_ok=True)
    compiled = tilelang.compile(
        build(n, threads),
        target="ascend",
        out_idx=-1,
        execution_backend="cython",
        compile_flags=[],
        pass_configs={"tl.disable_data_race_check": True},
    )
    (out / "kernel.cpp").write_text(compiled.get_kernel_source())
    shutil.copy(compiled.adapter.libpath, out / "kernel.so")
    print(f"compiled {name} N={n} threads={threads} -> {out / 'kernel.so'}", flush=True)


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args != ["compile"]:
        print("usage: reduce.py compile", flush=True)
        return 2
    for n in SIZES:
        compile_one(f"cross-n{n}", n, n)
        compile_one(f"loop-n{n}", n, LOOP_THREADS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
