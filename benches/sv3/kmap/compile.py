"""Compile one k-mapping kernel. Usage: compile.py thread|warp|simd rows cols K [threads]"""

import shutil
import sys
from pathlib import Path

import torch

torch.utils.rename_privateuse1_backend("npu")

import tilelang

from kernels import build_simd, build_thread, build_warp

HERE = Path(__file__).resolve().parent
BUILDERS = {"thread": build_thread, "warp": build_warp, "simd": build_simd}


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) not in (4, 5):
        print("usage: compile.py thread|warp|simd rows cols K [threads]", flush=True)
        return 2
    kind, rows, cols, K = args[0], int(args[1]), int(args[2]), int(args[3])
    threads = int(args[4]) if len(args) == 5 else None
    if threads is not None and kind != "thread":
        print("threads is only valid for thread", flush=True)
        return 2
    name = f"{kind}-r{rows}c{cols}-k{K}"
    if threads is not None:
        name = f"{name}-t{threads}"
    out = HERE / name
    out.mkdir(parents=True, exist_ok=True)
    kernel = BUILDERS[kind](rows, cols, K, threads) if threads is not None else BUILDERS[kind](rows, cols, K)
    compiled = tilelang.compile(
        kernel,
        target="ascend",
        out_idx=-1,
        execution_backend="cython",
        compile_flags=[],
        pass_configs={"tl.disable_data_race_check": True},
    )
    (out / "kernel.cpp").write_text(compiled.get_kernel_source())
    shutil.copy(compiled.adapter.libpath, out / "kernel.so")
    print(f"OK {name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
