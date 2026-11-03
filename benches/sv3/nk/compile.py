"""Compile one contiguous-K kernel. Usage: compile.py thread|ub|warp|simd N K [threads]"""

import shutil
import sys
from pathlib import Path

import torch

torch.utils.rename_privateuse1_backend("npu")

import tilelang

from kernels import build_simd, build_thread, build_thread_ub, build_warp

HERE = Path(__file__).resolve().parent
BUILDERS = {"thread": build_thread, "ub": build_thread_ub, "warp": build_warp, "simd": build_simd}
FIX = str(HERE.parents[2] / "bisheng_libstdcxx_clang_fix.h")


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) not in (3, 4):
        print("usage: compile.py thread|ub|warp|simd N K [threads]", flush=True)
        return 2
    kind, N, K = args[0], int(args[1]), int(args[2])
    threads = int(args[3]) if len(args) == 4 else None
    if threads is not None and kind not in ("thread", "ub"):
        print("threads is only valid for thread and ub", flush=True)
        return 2
    name = f"{kind}-n{N}-k{K}"
    if threads is not None:
        name = f"{name}-t{threads}"
    out = HERE / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    kernel = BUILDERS[kind](N, K, threads) if threads is not None else BUILDERS[kind](N, K)
    compiled = tilelang.compile(
        kernel,
        target="ascend",
        out_idx=-1,
        execution_backend="cython",
        compile_flags=["-include", FIX, "-isystem", "/usr/include/c++/12"],
        pass_configs={"tl.disable_data_race_check": True},
    )
    (out / "kernel.cpp").write_text(compiled.get_kernel_source())
    shutil.copy(compiled.adapter.libpath, out / "kernel.so")
    print(f"OK {name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
