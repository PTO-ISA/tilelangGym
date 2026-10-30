"""Compile the SIMT row ladder. Columns = threads, 128 then 256.

Shapes are the rows in ANALYSIS.md. block-quant/simt.py is the kernel.
"""

import importlib.util
import shutil
import sys
import traceback
from pathlib import Path

import torch

torch.utils.rename_privateuse1_backend("npu")

import tilelang

HERE = Path(__file__).resolve().parent
SIMT_PY = HERE.parent / "block-quant" / "simt.py"
SHAPES = (
    (32, 128),
    (64, 128),
    (128, 128),
    (192, 128),
    (208, 128),
    (215, 128),
    (216, 128),
    (32, 256),
    (64, 256),
    (96, 256),
    (104, 256),
    (107, 256),
    (108, 256),
)


def load_build():
    spec = importlib.util.spec_from_file_location("sv2_simt", SIMT_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build


def compile_one(build, rows: int, cols: int) -> str:
    name = f"r{rows}c{cols}"
    out = HERE / name
    out.mkdir(parents=True, exist_ok=True)
    try:
        compiled = tilelang.compile(
            build(rows, cols),
            target="ascend",
            out_idx=-1,
            execution_backend="cython",
            compile_flags=[],
            pass_configs={"tl.disable_data_race_check": True},
        )
    except Exception:
        (out / "compile.log").write_text(traceback.format_exc())
        line = f"FAIL {name} compile"
        print(line, flush=True)
        return line
    (out / "kernel.cpp").write_text(compiled.get_kernel_source())
    shutil.copy(compiled.adapter.libpath, out / "kernel.so")
    line = f"OK {name}"
    print(line, flush=True)
    return line


def main() -> int:
    build = load_build()
    failed = False
    for rows, cols in SHAPES:
        if not compile_one(build, rows, cols).startswith("OK"):
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
