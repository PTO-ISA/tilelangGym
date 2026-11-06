import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nk"))
import launch

so = sys.argv[1]
n, k = 4096, 32
rng = np.random.RandomState(2026)
A = rng.uniform(-1, 1, (n, k)).astype(np.float16)
X = rng.uniform(-1, 1, (k,)).astype(np.float16)
gold = (A.astype(np.float32) * X.astype(np.float32)[None, :]).sum(axis=1)
lib = launch._load_acl()
import ctypes
kern = ctypes.CDLL(so)
kern.init.restype = ctypes.c_int
kern.call.argtypes = [ctypes.c_void_p] * 4
kern.call.restype = ctypes.c_int
launch._chk(lib.aclInit(None), "init")
launch._chk(lib.aclrtSetDevice(0), "dev")
launch._chk(kern.init(), "initk")
st = ctypes.c_void_p()
launch._chk(lib.aclrtCreateStream(ctypes.byref(st)), "st")
ad, xd, od = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
launch._chk(lib.aclrtMalloc(ctypes.byref(ad), A.nbytes, 0), "ma")
launch._chk(lib.aclrtMalloc(ctypes.byref(xd), X.nbytes, 0), "mx")
launch._chk(lib.aclrtMalloc(ctypes.byref(od), n * 4, 0), "mo")
launch._chk(lib.aclrtMemcpy(ad, A.nbytes, np.ascontiguousarray(A).ctypes.data_as(ctypes.c_void_p), A.nbytes, 1), "ha")
launch._chk(lib.aclrtMemcpy(xd, X.nbytes, np.ascontiguousarray(X).ctypes.data_as(ctypes.c_void_p), X.nbytes, 1), "hx")
print("call", kern.call(ad, xd, od, st), flush=True)
launch._chk(lib.aclrtSynchronizeStream(st), "sync")
got = np.zeros(n, np.float32)
launch._chk(lib.aclrtMemcpy(got.ctypes.data_as(ctypes.c_void_p), n * 4, od, n * 4, 2), "d2h")
err = np.abs(got - gold)
print("max", float(err.max()), "at", int(err.argmax()), flush=True)
for b in range(0, n, 256):
    print(f"block {b:4d} max {err[b:b+256].max():.3e}", flush=True)
