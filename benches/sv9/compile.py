"""compile.py keep|remat_scores|remat_idx E K threads"""

import shutil
import sys
from pathlib import Path

import torch

torch.utils.rename_privateuse1_backend("npu")

import tilelang

from kernel import make_kernel

HERE = Path(__file__).resolve().parent


def main():
    arm, E, K, threads = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    name = f"{arm}-e{E}-k{K}-t{threads}"
    out = HERE / "runs" / "simt" / name
    out.mkdir(parents=True, exist_ok=True)
    compiled = tilelang.compile(
        make_kernel(E, K, threads, arm),
        target="ascend",
        out_idx=-1,
        execution_backend="cython",
        compile_flags=[
            "-include",
            str(HERE.parents[1] / "bisheng_libstdcxx_clang_fix.h"),
            "-isystem",
            "/usr/include/c++/12",
        ],
        pass_configs={"tl.disable_data_race_check": True},
    )
    (out / "kernel.cpp").write_text(compiled.get_kernel_source())
    shutil.copy(compiled.adapter.libpath, out / "kernel.so")
    print(f"OK {name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
