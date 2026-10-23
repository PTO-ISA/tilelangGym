#!/usr/bin/env python3
"""Launch one SV2.A operator kernel and check the values that kernel writes.

Usage: ops_launch.py kernel.so load|max|fdiv|fmul|sts
"""

import ctypes
import os
import sys

import numpy as np

EPS = 1e-6
R = 16
C = 1024

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


def _gold(name, x):
    if name == "load":
        return x, np.ones(x.shape, dtype=bool)
    if name == "max":
        ref = np.zeros_like(x)
        ref[0] = np.max(np.abs(x), axis=0)
        mask = np.zeros(x.shape, dtype=bool)
        mask[0] = True
        return ref, mask
    if name == "fdiv":
        ref = np.zeros_like(x)
        ref[0] = np.float32(1.0) / np.maximum(np.float32(EPS), np.abs(x[0]))
        mask = np.zeros(x.shape, dtype=bool)
        mask[0] = True
        return ref, mask
    if name == "fmul":
        return x * x[0:1], np.ones(x.shape, dtype=bool)
    if name == "sts":
        return np.ones_like(x), np.ones(x.shape, dtype=bool)
    raise SystemExit(f"unknown kernel {name}")


def main():
    so, name = sys.argv[1], sys.argv[2]
    nbytes = R * C * 4
    rng = np.random.RandomState(2026)
    x = rng.uniform(-3, 3, (R, C)).astype(np.float32)
    ref, mask = _gold(name, x)

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
            xd, nbytes, np.ascontiguousarray(x).ctypes.data_as(ctypes.c_void_p), nbytes, _ACL_MEMCPY_HOST_TO_DEVICE
        ),
        "h2d",
    )
    rc = kern.call(xd, od, st)
    if rc != 0:
        err = kern.get_last_error()
        print(f"FAIL {name} call={rc} {err}", flush=True)
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
    got = oh[mask]
    exp = ref[mask]
    err = float(np.max(np.abs(got - exp)))
    if not np.allclose(got, exp, atol=1e-4, rtol=1e-4):
        print(f"FAIL {name} maxabs={err}", flush=True)
        return 1
    print(f"PASS {name} maxabs={err:.3e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
