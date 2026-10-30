#!/usr/bin/env python3
"""Launch a compiled SV2.A SIMT kernel.so under camodel.

Usage: launch_simt.py kernel.so R C

Gold is a per-column scale, then a multiply by 1/scale.
"""

import ctypes
import os
import sys

import numpy as np

EPS = 1e-6

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


def _load_acl():
    sim = os.environ["TL_CAMODEL_LIB"]
    ctypes.CDLL(os.path.join(sim, "libruntime_camodel.so"), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL("libascendcl.so")
    for name, arg in [
        ("aclInit", ctypes.c_char_p),
        ("aclFinalize", None),
        ("aclrtSetDevice", ctypes.c_int),
        ("aclrtResetDevice", ctypes.c_int),
    ]:
        getattr(lib, name).argtypes = [arg] if arg else []
        getattr(lib, name).restype = ctypes.c_int
    lib.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtCreateStream.restype = ctypes.c_int
    lib.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
    lib.aclrtDestroyStream.restype = ctypes.c_int
    lib.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
    lib.aclrtMalloc.restype = ctypes.c_int
    lib.aclrtFree.argtypes = [ctypes.c_void_p]
    lib.aclrtFree.restype = ctypes.c_int
    lib.aclrtMemcpy.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    ]
    lib.aclrtMemcpy.restype = ctypes.c_int
    lib.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
    lib.aclrtSynchronizeStream.restype = ctypes.c_int
    return lib


def _chk(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed {rc}")


def main():
    so, R, C = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    nbytes = R * C * 4
    rng = np.random.RandomState(2026)
    X = rng.uniform(-3, 3, (R, C)).astype(np.float32)
    scale = np.maximum(np.float32(EPS), np.max(np.abs(X), axis=0)).astype(np.float32)
    ref = (X * (np.float32(1.0) / scale)).astype(np.float32)

    lib = _load_acl()
    kern = ctypes.CDLL(so)
    kern.init.restype = ctypes.c_int
    kern.call.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    kern.call.restype = ctypes.c_int
    kern.get_last_error.restype = ctypes.c_char_p

    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    _chk(kern.init(), "kern.init")
    st = ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")
    xd = ctypes.c_void_p()
    od = ctypes.c_void_p()
    _chk(lib.aclrtMalloc(ctypes.byref(xd), nbytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mx")
    _chk(lib.aclrtMalloc(ctypes.byref(od), nbytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mo")
    _chk(
        lib.aclrtMemcpy(
            xd, nbytes, np.ascontiguousarray(X).ctypes.data_as(ctypes.c_void_p), nbytes, _ACL_MEMCPY_HOST_TO_DEVICE
        ),
        "h2d",
    )
    rc = kern.call(xd, od, st)
    if rc != 0:
        err = kern.get_last_error()
        print(f"FAIL r{R}c{C} call={rc} {err}", flush=True)
        return 1
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    oh = np.zeros((R, C), dtype=np.float32)
    _chk(
        lib.aclrtMemcpy(oh.ctypes.data_as(ctypes.c_void_p), nbytes, od, nbytes, _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    lib.aclrtFree(xd)
    lib.aclrtFree(od)
    lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0)
    lib.aclFinalize()
    err = float(np.max(np.abs(oh - ref)))
    if not np.allclose(oh, ref, atol=1e-4, rtol=1e-4):
        print(f"FAIL r{R}c{C} maxabs={err}", flush=True)
        return 1
    print(f"PASS r{R}c{C} maxabs={err:.3e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
