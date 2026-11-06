"""Compile one SV8 kernel.

reload32 | keep32 | warp32u2 | warp32u4 | piece-load | piece-max | piece-store
rows cols threads
"""

import shutil
import sys
from pathlib import Path

import torch

torch.utils.rename_privateuse1_backend("npu")

import tilelang

from kernel import make_load4_piece, make_thread_f32, make_warp_f16_unroll

HERE = Path(__file__).resolve().parent


def main():
    kind = sys.argv[1]
    if kind in ("reload32", "keep32"):
        rows, cols, threads = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        name = f"{kind}-r{rows}c{cols}-g32-t{threads}"
        kernel = make_thread_f32(rows, cols, threads, keep=(kind == "keep32"))
        target = "ascend"
    elif kind in ("piece-load", "piece-max", "piece-store"):
        rows, cols, threads = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        piece = kind.split("-", 1)[1]
        name = f"{kind}-r{rows}c{cols}-g32-t{threads}"
        kernel = make_load4_piece(rows, cols, threads, piece)
        target = "ascend"
    elif kind in ("warp32u1", "warp32u2", "warp32u4"):
        rows, cols, threads = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        factor = {"warp32u1": 1, "warp32u2": 2, "warp32u4": 4}[kind]
        name = f"{kind}-r{rows}c{cols}-g32-t{threads}"
        kernel = make_warp_f16_unroll(rows, cols, threads, factor, "float32")
        target = "ascend"
    else:
        print(
            "usage: compile.py reload32|keep32|warp32u2|warp32u4|piece-load|piece-max|piece-store rows cols threads",
            flush=True,
        )
        return 2
    if kind.startswith("piece"):
        root = HERE / "runs" / "ops"
    elif kind.startswith("reload"):
        root = HERE / "runs" / "simt" / "reload"
    elif kind.startswith("keep"):
        root = HERE / "runs" / "simt" / "keep"
    elif kind.startswith("warp32"):
        root = HERE / "runs" / "simt" / "load4"
    else:
        root = HERE / "runs" / "simt"
    root.mkdir(parents=True, exist_ok=True)
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    compiled = tilelang.compile(
        kernel,
        target=target,
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
