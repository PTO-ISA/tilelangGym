"""Launch an SV8 kernel. Usage: launch.py kernel.so rows cols [G] [float16]"""

import ctypes
import os
import sys

import numpy as np

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2
G = 16


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
    so, rows, cols = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    group = int(sys.argv[4]) if len(sys.argv) > 4 else G
    dtype = np.float16 if len(sys.argv) > 5 and sys.argv[5] == "float16" else np.float32
    rng = np.random.RandomState(2026)
    x = rng.uniform(-1, 1, (rows * cols,)).astype(dtype)
    grouped = x.astype(np.float32).reshape(-1, group)
    scale = 1.0 / np.maximum(np.max(np.abs(grouped), axis=1), 1e-6)
    gold = (grouped * scale[:, None]).astype(dtype).reshape(rows, cols)
    nbytes = x.nbytes

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
    xd, od = ctypes.c_void_p(), ctypes.c_void_p()
    _chk(lib.aclrtMalloc(ctypes.byref(xd), nbytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mx")
    _chk(lib.aclrtMalloc(ctypes.byref(od), nbytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mo")
    _chk(
        lib.aclrtMemcpy(xd, nbytes, np.ascontiguousarray(x).ctypes.data_as(ctypes.c_void_p), nbytes, _ACL_MEMCPY_HOST_TO_DEVICE),
        "h2d",
    )
    rc = kern.call(xd, od, st)
    if rc != 0:
        err = kern.get_last_error()
        print(f"FAIL call={rc} {err}", flush=True)
        return 1
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    got = np.zeros((rows, cols), dtype=dtype)
    _chk(
        lib.aclrtMemcpy(got.ctypes.data_as(ctypes.c_void_p), nbytes, od, nbytes, _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    lib.aclrtFree(xd)
    lib.aclrtFree(od)
    lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0)
    lib.aclFinalize()
    tol = 2e-3 if dtype == np.float16 else 1e-5
    err = float(np.max(np.abs(got.astype(np.float32) - gold.astype(np.float32))))
    if err > tol:
        print(f"FAIL maxabs={err}", flush=True)
        return 1
    print(f"PASS maxabs={err:.3e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
