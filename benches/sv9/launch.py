"""launch.py kernel.so E K. Stable top-K, smaller index wins a tie."""

import ctypes
import os
import sys

import numpy as np

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2
NEG = np.float32(-3.402823e38)


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


def gold(scores, k):
    work = scores.astype(np.float32).copy()
    out = np.empty(k, np.int32)
    for i in range(k):
        m = np.max(work)
        idx = int(np.flatnonzero(work == m)[0])
        out[i] = idx
        work[idx] = NEG
    return out


def main():
    so, e, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rng = np.random.RandomState(2026)
    scores = rng.uniform(-1, 1, (e,)).astype(np.float32)
    # One exact tie, so the smaller index must win.
    scores[3] = scores[1]
    expect = gold(scores, k)
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
    sb, ib = scores.nbytes, k * 4
    _chk(lib.aclrtMalloc(ctypes.byref(xd), sb, _ACL_MEM_MALLOC_HUGE_FIRST), "mx")
    _chk(lib.aclrtMalloc(ctypes.byref(od), ib, _ACL_MEM_MALLOC_HUGE_FIRST), "mo")
    _chk(
        lib.aclrtMemcpy(xd, sb, np.ascontiguousarray(scores).ctypes.data_as(ctypes.c_void_p), sb, _ACL_MEMCPY_HOST_TO_DEVICE),
        "h2d",
    )
    rc = kern.call(xd, od, st)
    if rc != 0:
        print(f"FAIL call={rc} {kern.get_last_error()}", flush=True)
        return 1
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    got = np.empty(k, np.int32)
    _chk(
        lib.aclrtMemcpy(got.ctypes.data_as(ctypes.c_void_p), ib, od, ib, _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    ok = bool(np.array_equal(got, expect))
    print(f"PASS idx={got.tolist()}" if ok else f"FAIL got={got.tolist()} gold={expect.tolist()}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
