"""Launch one microbenchmark. Usage: launch.py <case>"""

import ctypes
import os
import sys

import numpy as np

from kernels import CUT_STEPS, CUT_THREADS, CUT_X, FFMA_ITERS, LADDER_STEPS, N_CONTIG, REPS, SIMD_VEC, SPECS, STRIDE, WARP

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


def _arrays(name: str):
    op, dtype, threads, stride = SPECS[name]
    rng = np.random.RandomState(2026)
    np_dtype = np.float32 if dtype == "float32" else np.float16
    if op == "lds":
        n = N_CONTIG if stride == 1 else threads * STRIDE
        src = rng.uniform(-1, 1, (n,)).astype(np_dtype)
        values = src.astype(np.float32)
        if stride == 1:
            gold = values.reshape(-1, threads).sum(axis=0) * REPS
        else:
            gold = values.reshape(threads, STRIDE).sum(axis=1) * REPS
        return src, gold
    if op == "sts":
        src = np.zeros((N_CONTIG,), np.float32)
        return src, np.ones((N_CONTIG,), np.float32)
    if op == "ffma":
        src = rng.uniform(0.5, 1.5, (threads,)).astype(np.float32)
        return src, FFMA_ITERS * src * src
    if op in ("lds-ffma", "lds-ffma-sts"):
        src = rng.uniform(-1, 1, (LADDER_STEPS * WARP,)).astype(np.float32)
        squares = src.reshape(LADDER_STEPS, WARP) ** 2
        running = np.cumsum(squares, axis=0)
        if op == "lds-ffma":
            return src, running[-1]
        return src, running.reshape(-1)
    if op == "lds-cut":
        n = CUT_THREADS * CUT_STEPS
        src = rng.uniform(-1, 1, (n + CUT_X,)).astype(np.float16)
        values = src.astype(np.float32)
        gold = values[:n].reshape(CUT_STEPS, CUT_THREADS).sum(axis=0)
        return src, gold
    if op == "lds-pair":
        src = rng.uniform(-1, 1, (LADDER_STEPS * WARP,)).astype(np.float32)
        return src, src.reshape(LADDER_STEPS // 2, WARP, 2).sum(axis=(0, 2))
    lanes = 64 if dtype == "float32" else 128
    src = rng.uniform(-1, 1, (lanes * SIMD_VEC,)).astype(np_dtype)
    if dtype == "float16":
        acc = np.zeros((lanes,), np.float16)
        for row in src.reshape(SIMD_VEC, lanes):
            acc = acc + row
        return src, acc
    return src, src.reshape(SIMD_VEC, lanes).sum(axis=0)


def main():
    name = sys.argv[1]
    src, gold = _arrays(name)
    so = os.path.join(os.path.dirname(__file__), "runs", name, "kernel.so")
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
    sd, dd = ctypes.c_void_p(), ctypes.c_void_p()
    src_b, dst_b = src.nbytes, gold.nbytes
    _chk(lib.aclrtMalloc(ctypes.byref(sd), src_b, _ACL_MEM_MALLOC_HUGE_FIRST), "ms")
    _chk(lib.aclrtMalloc(ctypes.byref(dd), dst_b, _ACL_MEM_MALLOC_HUGE_FIRST), "md")
    _chk(lib.aclrtMemcpy(sd, src_b, np.ascontiguousarray(src).ctypes.data_as(ctypes.c_void_p), src_b, _ACL_MEMCPY_HOST_TO_DEVICE), "h2d")
    rc = kern.call(sd, dd, st)
    if rc != 0:
        print(f"FAIL call={rc} {kern.get_last_error()}", flush=True)
        return 1
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    got = np.zeros_like(gold)
    _chk(lib.aclrtMemcpy(got.ctypes.data_as(ctypes.c_void_p), dst_b, dd, dst_b, _ACL_MEMCPY_DEVICE_TO_HOST), "d2h")
    err = float(np.max(np.abs(got.astype(np.float32) - gold.astype(np.float32))))
    print(f"PASS maxabs={err:.3e}" if err <= 1e-2 else f"FAIL maxabs={err:.3e} got0={got[:4]} gold0={gold[:4]}", flush=True)
    return 0 if err <= 1e-2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
