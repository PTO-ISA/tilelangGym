#!/usr/bin/env python3
"""SV2.A block quant, SIMD PTO-DSL.

One scale per column. Nothing is broadcast across columns.

    scale[col] = max(eps, max over rows |X[row, col]|)
    Out[row, col] = X[row, col] * (1 / scale[col])

C is a multiple of VL. One instruction covers 64 columns, and each lane keeps that column's own scale.
"""

import ctypes
import os
import sys
import time

import numpy as np
from ptodsl import pto

VL = 64
EPS = 1e-6

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


def reference(X):
    scale = np.maximum(np.float32(EPS), np.max(np.abs(X), axis=0)).astype(np.float32)
    inv = (np.float32(1.0) / scale).astype(np.float32)
    return (X * inv).astype(np.float32), scale


def build(R: int, C: int):
    if C % VL != 0:
        raise ValueError(f"C={C} must be a multiple of VL={VL}")
    x_bytes = R * C * 4
    off_out = x_bytes

    @pto.jit(
        name=f"SV2A_col_r{R}_c{C}",
        target="a5",
        backend="vpto",
        mode="explicit",
        kernel_kind="vector",
        insert_sync=False,
        ast_rewrite=False,
    )
    def kernel(out_gm: pto.ptr(pto.f32, "gm"), x_gm: pto.ptr(pto.f32, "gm")):
        x_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        out_ub = pto.castptr(pto.const(off_out, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

        pto.mte_load(x_gm, x_ub, 0, x_bytes, nburst=(1, x_bytes, x_bytes))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        mask = pto.vmi.create_mask(VL, size=VL)
        neg1 = pto.vmi.vbrc(pto.f32(-1.0), size=VL)
        ones = pto.vmi.vbrc(pto.f32(1.0), size=VL)
        eps = pto.vmi.vbrc(pto.f32(EPS), size=VL)

        for g in range(C // VL):
            base = g * VL
            c0 = pto.vmi.vload(x_ub, base, size=VL)
            acc0 = pto.vmi.vmax(c0, pto.vmi.vmul(c0, neg1, mask), mask)
            if R == 1:
                acc = acc0
            else:
                fold = pto.for_(1, R, step=1).carry(acc=acc0)
                with fold:
                    c = pto.vmi.vload(x_ub, fold.iv * C + base, size=VL)
                    absv = pto.vmi.vmax(c, pto.vmi.vmul(c, neg1, mask), mask)
                    fold.update(acc=pto.vmi.vmax(fold.acc, absv, mask))
                acc = fold.final("acc")
            scale = pto.vmi.vmax(acc, eps, mask)
            inv = pto.vmi.vdiv(ones, scale, mask)
            with pto.for_(0, R, step=1) as row:
                c = pto.vmi.vload(x_ub, row * C + base, size=VL)
                pto.vmi.vstore(pto.vmi.vmul(c, inv, mask), out_ub, row * C + base, mask)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, x_bytes, nburst=(1, x_bytes, x_bytes))

    return kernel


def _load_acl():
    sim = os.environ.get("TL_CAMODEL_LIB")
    if sim:
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


def run(R: int, C: int) -> int:
    kernel = build(R, C)
    nbytes = R * C * 4
    rng = np.random.RandomState(2026)
    X = rng.uniform(-3, 3, (R, C)).astype(np.float32)
    ref, scale = reference(X)
    lib = _load_acl()
    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
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
    t0 = time.perf_counter()
    comp = kernel.compile()
    cs = time.perf_counter() - t0
    t0 = time.perf_counter()
    comp[1, st](int(od.value), int(xd.value))
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    ls = time.perf_counter() - t0
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
    print(f"PASS r{R}c{C} maxabs={err:.3e} compile={cs:.3f}s launch={ls:.3f}s", flush=True)
    return 0


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2:
        print("usage: simd.py R C", flush=True)
        return 2
    return run(int(args[0]), int(args[1]))


if __name__ == "__main__":
    raise SystemExit(main())
