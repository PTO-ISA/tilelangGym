"""Compile one microbenchmark. Usage: compile.py <case>"""

import shutil
import sys
from pathlib import Path

import torch

torch.utils.rename_privateuse1_backend("npu")

import tilelang

from kernels import SPECS, build

HERE = Path(__file__).resolve().parent
FIX = str(HERE.parents[3] / "bisheng_libstdcxx_clang_fix.h")


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1 or args[0] not in SPECS:
        print("usage: compile.py " + "|".join(SPECS), flush=True)
        return 2
    name = args[0]
    out = HERE / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    compiled = tilelang.compile(
        build(name),
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
