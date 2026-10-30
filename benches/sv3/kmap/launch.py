"""Launch one k-mapping kernel.so. Usage: launch.py kernel.so rows cols K"""

import ctypes
import os
import sys

import numpy as np

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
    so, rows, cols, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    rng = np.random.RandomState(2026)
    A = rng.uniform(-1, 1, (K, rows, cols)).astype(np.float16)
    X = rng.uniform(-1, 1, (K,)).astype(np.float16)
    gold = (A.astype(np.float32) * X.astype(np.float32)[:, None, None]).sum(axis=0)
    a_bytes = A.nbytes
    x_bytes = X.nbytes
    o_bytes = rows * cols * 4

    lib = _load_acl()
    kern = ctypes.CDLL(so)
    kern.init.restype = ctypes.c_int
    kern.call.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    kern.call.restype = ctypes.c_int
    kern.get_last_error.restype = ctypes.c_char_p

    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    _chk(kern.init(), "kern.init")
    st = ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")
    ad, xd, od = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    _chk(lib.aclrtMalloc(ctypes.byref(ad), a_bytes, _ACL_MEM_MALLOC_HUGE_FIRST), "ma")
    _chk(lib.aclrtMalloc(ctypes.byref(xd), x_bytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mx")
    _chk(lib.aclrtMalloc(ctypes.byref(od), o_bytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mo")
    _chk(
        lib.aclrtMemcpy(ad, a_bytes, np.ascontiguousarray(A).ctypes.data_as(ctypes.c_void_p), a_bytes, _ACL_MEMCPY_HOST_TO_DEVICE),
        "h2d-a",
    )
    _chk(
        lib.aclrtMemcpy(xd, x_bytes, np.ascontiguousarray(X).ctypes.data_as(ctypes.c_void_p), x_bytes, _ACL_MEMCPY_HOST_TO_DEVICE),
        "h2d-x",
    )
    rc = kern.call(ad, xd, od, st)
    if rc != 0:
        err = kern.get_last_error()
        print(f"FAIL call={rc} {err}", flush=True)
        return 1
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    got = np.zeros((rows, cols), dtype=np.float32)
    _chk(
        lib.aclrtMemcpy(got.ctypes.data_as(ctypes.c_void_p), o_bytes, od, o_bytes, _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    lib.aclrtFree(ad)
    lib.aclrtFree(xd)
    lib.aclrtFree(od)
    lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0)
    lib.aclFinalize()
    err = float(np.max(np.abs(got - gold)))
    if err > 1e-2:
        print(f"FAIL maxabs={err}", flush=True)
        return 1
    print(f"PASS maxabs={err:.3e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
