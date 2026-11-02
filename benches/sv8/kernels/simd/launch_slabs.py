"""Compile one SIMD slab and call it n times on a contiguous tile.

Usage: launch_slabs.py pack_ub_vl256|keeplive_vbrc_vl256 slab_R slab_C n_calls
"""

import ctypes
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sv8d_pack_ub_scale_case3 as pack


def main():
    arm, slab_r, slab_c, n_calls = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    slab = slab_r * slab_c
    n = slab * n_calls
    rng = np.random.RandomState(2026)
    x = rng.uniform(-1, 1, (n,)).astype(np.float32)
    grouped = x.reshape(-1, 32)
    scale = 1.0 / np.maximum(np.max(np.abs(grouped), axis=1), 1e-6)
    gold = (grouped * scale[:, None]).reshape(n)
    nbytes = x.nbytes
    slab_bytes = slab * 4

    kernel = pack._kernel(slab_r, slab_c, 32, arm)
    lib = pack._load_acl()
    pack._chk(lib.aclInit(None), "init")
    pack._chk(lib.aclrtSetDevice(0), "dev")
    st, xd, od = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    pack._chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")
    pack._chk(lib.aclrtMalloc(ctypes.byref(xd), nbytes, pack._ACL_MEM_MALLOC_HUGE_FIRST), "mx")
    pack._chk(lib.aclrtMalloc(ctypes.byref(od), nbytes, pack._ACL_MEM_MALLOC_HUGE_FIRST), "mo")
    pack._chk(
        lib.aclrtMemcpy(
            xd, nbytes, np.ascontiguousarray(x).ctypes.data_as(ctypes.c_void_p), nbytes,
            pack._ACL_MEMCPY_HOST_TO_DEVICE,
        ),
        "h2d",
    )
    comp = kernel.compile()
    for i in range(n_calls):
        comp[1, st](int(od.value + i * slab_bytes), int(xd.value + i * slab_bytes))
    pack._chk(lib.aclrtSynchronizeStream(st), "sync")
    got = np.empty(n, np.float32)
    pack._chk(
        lib.aclrtMemcpy(got.ctypes.data_as(ctypes.c_void_p), nbytes, od, nbytes, pack._ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    err = float(np.max(np.abs(got - gold)))
    print(f"PASS maxabs={err:.3e} slabs={n_calls}" if err <= 1e-4 else f"FAIL maxabs={err:.3e}", flush=True)
    return 0 if err <= 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
