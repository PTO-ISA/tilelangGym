"""Call a slab kernel n times on one contiguous tile.

Usage: launch_slabs.py kernel.so slab_elems n_calls
The tile is n_calls slabs laid out one after another. G is 32.
"""

import ctypes
import sys

import numpy as np

from launch import _ACL_MEMCPY_DEVICE_TO_HOST, _ACL_MEMCPY_HOST_TO_DEVICE, _ACL_MEM_MALLOC_HUGE_FIRST, _chk, _load_acl


def main():
    so, slab, n_calls = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    n = slab * n_calls
    rng = np.random.RandomState(2026)
    x = rng.uniform(-1, 1, (n,)).astype(np.float32)
    grouped = x.reshape(-1, 32)
    scale = 1.0 / np.maximum(np.max(np.abs(grouped), axis=1), 1e-6)
    gold = (grouped * scale[:, None]).reshape(n)
    nbytes = x.nbytes
    slab_bytes = slab * 4

    lib = _load_acl()
    kern = ctypes.CDLL(so)
    kern.init.restype = ctypes.c_int
    kern.call.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    kern.call.restype = ctypes.c_int
    kern.get_last_error.restype = ctypes.c_char_p
    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    _chk(kern.init(), "kern.init")
    st, xd, od = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")
    _chk(lib.aclrtMalloc(ctypes.byref(xd), nbytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mx")
    _chk(lib.aclrtMalloc(ctypes.byref(od), nbytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mo")
    _chk(
        lib.aclrtMemcpy(xd, nbytes, np.ascontiguousarray(x).ctypes.data_as(ctypes.c_void_p), nbytes, _ACL_MEMCPY_HOST_TO_DEVICE),
        "h2d",
    )
    for i in range(n_calls):
        xs = ctypes.c_void_p(xd.value + i * slab_bytes)
        os_ = ctypes.c_void_p(od.value + i * slab_bytes)
        rc = kern.call(xs, os_, st)
        if rc != 0:
            print(f"FAIL call={rc} slab={i} {kern.get_last_error()}", flush=True)
            return 1
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    got = np.empty(n, np.float32)
    _chk(
        lib.aclrtMemcpy(got.ctypes.data_as(ctypes.c_void_p), nbytes, od, nbytes, _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    err = float(np.max(np.abs(got - gold)))
    print(f"PASS maxabs={err:.3e} slabs={n_calls}" if err <= 1e-4 else f"FAIL maxabs={err:.3e}", flush=True)
    return 0 if err <= 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
