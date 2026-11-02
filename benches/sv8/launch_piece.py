"""Launch one load-4 piece. Usage: launch_piece.py load|max|store kernel.so rows cols"""

import ctypes
import os
import sys

import numpy as np

from launch import _chk, _load_acl

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


def main():
    piece, so, rows, cols = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    rng = np.random.RandomState(2026)
    x = rng.uniform(-1, 1, (rows * cols,)).astype(np.float32)
    if piece == "load":
        gold = x.reshape(rows, cols)
    elif piece == "max":
        grouped = x.reshape(-1, 32)
        gold = np.repeat(np.max(np.abs(grouped), axis=1), 32).reshape(rows, cols)
    elif piece == "store":
        gold = np.ones((rows, cols), np.float32)
    else:
        raise SystemExit(piece)
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
    st, xd, od = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")
    _chk(lib.aclrtMalloc(ctypes.byref(xd), nbytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mx")
    _chk(lib.aclrtMalloc(ctypes.byref(od), nbytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mo")
    _chk(
        lib.aclrtMemcpy(xd, nbytes, np.ascontiguousarray(x).ctypes.data_as(ctypes.c_void_p), nbytes, _ACL_MEMCPY_HOST_TO_DEVICE),
        "h2d",
    )
    rc = kern.call(xd, od, st)
    if rc != 0:
        print(f"FAIL call={rc} {kern.get_last_error()}", flush=True)
        return 1
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    got = np.zeros((rows, cols), np.float32)
    _chk(
        lib.aclrtMemcpy(got.ctypes.data_as(ctypes.c_void_p), nbytes, od, nbytes, _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    err = float(np.max(np.abs(got - gold)))
    print(f"PASS maxabs={err:.3e}" if err <= 1e-4 else f"FAIL maxabs={err:.3e}", flush=True)
    return 0 if err <= 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
