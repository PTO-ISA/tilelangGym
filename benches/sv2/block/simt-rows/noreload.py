"""Block quant with no reload of X or the scale. Threads = C.

One thread owns one column. Each row of that column is one local.
pass2 divides the local. The scale is max(result[0], eps).

gm: no data UB. The locals are loaded from X and stored to Out.
one: one UB buffer. pass1 loads it into the locals, pass2 overwrites it.
"""

import shutil
import sys
from pathlib import Path

import torch

torch.utils.rename_privateuse1_backend("npu")

import tilelang.ascend.language as T

EPS = 1e-6
HERE = Path(__file__).resolve().parent


def build_gm(R: int, C: int):
    rows = tuple(range(R))

    @T.prim_func
    def main(X: T.Tensor((R, C), "float32"), Out: T.Tensor((R, C), "float32")):
        with T.Kernel(1):
            with T.SimtVF(threads=C):
                acc = T.alloc_reducer((1,), "float32", op="max")
                T.reducer_init(acc)
                held = [T.alloc_var("float32") for _ in rows]
                for col in T.Parallel(C):
                    m = T.alloc_var("float32")
                    m = T.float32(0)
                    for row in rows:
                        held[row][0] = X[row, col]
                        vv = held[row][0]
                        m = T.max(m, T.max(vv, -vv))
                    T.reducer_update(acc[0], m)
                result = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(acc, result)
                for col in T.Parallel(C):
                    scale = T.max(result[0], T.float32(EPS))
                    for row in rows:
                        Out[row, col] = held[row][0] / scale

    return main


def build_one(R: int, C: int):
    rows = tuple(range(R))

    @T.prim_func
    def main(X: T.Tensor((R, C), "float32"), Out: T.Tensor((R, C), "float32")):
        with T.Kernel(1):
            buf = T.alloc_shared((R, C), "float32")
            T.copy(X, buf)
            with T.SimtVF(threads=C):
                acc = T.alloc_reducer((1,), "float32", op="max")
                T.reducer_init(acc)
                held = [T.alloc_var("float32") for _ in rows]
                for col in T.Parallel(C):
                    m = T.alloc_var("float32")
                    m = T.float32(0)
                    for row in rows:
                        held[row][0] = buf[row, col]
                        vv = held[row][0]
                        m = T.max(m, T.max(vv, -vv))
                    T.reducer_update(acc[0], m)
                result = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(acc, result)
                for col in T.Parallel(C):
                    scale = T.max(result[0], T.float32(EPS))
                    for row in rows:
                        buf[row, col] = held[row][0] / scale
            T.copy(buf, Out)

    return main


def compile_one(rows: int, cols: int, kind: str) -> None:
    import tilelang

    out = HERE / f"noreload-{kind}-r{rows}c{cols}"
    out.mkdir(parents=True, exist_ok=True)
    build = build_gm if kind == "gm" else build_one
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
    print(f"OK noreload-{kind}-r{rows}c{cols}", flush=True)


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 3:
        print("usage: noreload.py gm|one R C", flush=True)
        return 2
    if args[0] not in ("gm", "one"):
        print("kind must be gm or one", flush=True)
        return 2
    compile_one(int(args[1]), int(args[2]), args[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
