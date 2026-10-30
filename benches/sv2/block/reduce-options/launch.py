#!/usr/bin/env python3
"""Launch one reduce-options kernel.so. Usage: launch.py kernel.so N"""

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
    so = sys.argv[1]
    n = int(sys.argv[2])
    nbytes = n * 4
    rng = np.random.RandomState(2026)
    X = rng.uniform(-3, 3, (n,)).astype(np.float32)
    scale = np.float32(max(EPS, float(np.max(np.abs(X)))))

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
    _chk(lib.aclrtMalloc(ctypes.byref(od), 4, _ACL_MEM_MALLOC_HUGE_FIRST), "mo")
    _chk(
        lib.aclrtMemcpy(
            xd, nbytes, np.ascontiguousarray(X).ctypes.data_as(ctypes.c_void_p), nbytes, _ACL_MEMCPY_HOST_TO_DEVICE
        ),
        "h2d",
    )
    rc = kern.call(xd, od, st)
    if rc != 0:
        err = kern.get_last_error()
        print(f"FAIL call={rc} {err}", flush=True)
        return 1
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    oh = np.zeros((1,), dtype=np.float32)
    _chk(
        lib.aclrtMemcpy(oh.ctypes.data_as(ctypes.c_void_p), 4, od, 4, _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    lib.aclrtFree(xd)
    lib.aclrtFree(od)
    lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0)
    lib.aclFinalize()
    err = float(abs(oh[0] - scale))
    if err > 1e-5:
        print(f"FAIL maxabs={err} got={float(oh[0])} scale={float(scale)}", flush=True)
        return 1
    print(f"PASS maxabs={err:.3e} scale={float(scale):.6g}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
