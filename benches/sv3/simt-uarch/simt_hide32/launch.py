"""Launch one 32-float-per-thread kernel. Usage: launch.py <so> <op> <threads>"""

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
    lib.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
    lib.aclrtMalloc.restype = ctypes.c_int
    lib.aclrtMemcpy.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    ]
    lib.aclrtMemcpy.restype = ctypes.c_int
    lib.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
    lib.aclrtSynchronizeStream.restype = ctypes.c_int
    return lib


def main():
    so, op, threads = sys.argv[1], sys.argv[2], int(sys.argv[3])
    n = threads * 32
    rng = np.random.RandomState(2026)
    a = rng.uniform(-1, 1, (n,)).astype(np.float32)
    b = rng.uniform(-1, 1, (n,)).astype(np.float32)
    if op == "sts":
        gold = np.ones((n,), np.float32)
    elif op == "fadd":
        gold = a + b
    elif op == "ffma":
        gold = a * b + a
    else:
        gold = a
    lib = _load_acl()
    kern = ctypes.CDLL(so)
    kern.init.restype = ctypes.c_int
    kern.call.restype = ctypes.c_int
    if op in ("fadd", "ffma"):
        kern.call.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    else:
        kern.call.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    if lib.aclInit(None) or lib.aclrtSetDevice(0) or kern.init():
        print("FAIL init", flush=True)
        return 1
    st = ctypes.c_void_p()
    lib.aclrtCreateStream(ctypes.byref(st))
    ptrs = []

    def alloc(arr):
        p = ctypes.c_void_p()
        lib.aclrtMalloc(ctypes.byref(p), arr.nbytes, _ACL_MEM_MALLOC_HUGE_FIRST)
        lib.aclrtMemcpy(p, arr.nbytes, np.ascontiguousarray(arr).ctypes.data_as(ctypes.c_void_p), arr.nbytes, _ACL_MEMCPY_HOST_TO_DEVICE)
        ptrs.append(p)
        return p

    dst = ctypes.c_void_p()
    lib.aclrtMalloc(ctypes.byref(dst), gold.nbytes, _ACL_MEM_MALLOC_HUGE_FIRST)
    if op in ("fadd", "ffma"):
        rc = kern.call(dst, alloc(a), alloc(b), st)
    else:
        rc = kern.call(dst, alloc(a), st)
    if rc != 0:
        print(f"FAIL call={rc}", flush=True)
        return 1
    lib.aclrtSynchronizeStream(st)
    got = np.zeros_like(gold)
    lib.aclrtMemcpy(got.ctypes.data_as(ctypes.c_void_p), gold.nbytes, dst, gold.nbytes, _ACL_MEMCPY_DEVICE_TO_HOST)
    err = float(np.max(np.abs(got - gold)))
    print(f"PASS maxabs={err:.3e}" if err <= 1e-2 else f"FAIL maxabs={err:.3e} got0={got[:4]} gold0={gold[:4]}", flush=True)
    return 0 if err <= 1e-2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
