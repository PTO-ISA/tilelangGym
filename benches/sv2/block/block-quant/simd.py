#!/usr/bin/env python3
"""SV2 block quant, SIMD PTO-DSL.

One scale for the whole R*C block, not per lane and not per row:

    scale = max(eps, max |X|)
    Out = X / scale

VL is only the load width. vcmax collapses every lane into that one scale.
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
    scale = np.float32(max(EPS, float(np.max(np.abs(X)))))
    return (X / scale).astype(np.float32), scale


def build(R: int, C: int):
    if (R * C) % VL != 0:
        raise ValueError(f"R*C={R * C} must be a multiple of VL={VL}")
    nch = (R * C) // VL
    x_bytes = R * C * 4
    off_out = x_bytes

    @pto.jit(
        name=f"SV2_block_r{R}_c{C}",
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
        one = pto.vmi.create_mask(1, size=1)
        neg1 = pto.vmi.vbrc(pto.f32(-1.0), size=VL)
        zero = pto.vmi.vbrc(pto.f32(0.0), size=VL)
        eps1 = pto.vmi.vbrc(pto.f32(EPS), size=1)

        fold = pto.for_(0, nch, step=1).carry(acc=zero)
        with fold:
            c = pto.vmi.vload(x_ub, fold.iv * VL, size=VL)
            absv = pto.vmi.vmax(c, pto.vmi.vmul(c, neg1, mask), mask)
            fold.update(acc=pto.vmi.vmax(fold.acc, absv, mask))
        acc = fold.final("acc")
        amax = pto.vmi.vcmax(acc, mask)
        amax = pto.vmi.vmax(amax, eps1, one)
        scale = pto.vmi.vbrc(amax, size=VL)

        with pto.for_(0, nch, step=1) as ch:
            c = pto.vmi.vload(x_ub, ch * VL, size=VL)
            pto.vmi.vstore(pto.vmi.vdiv(c, scale, mask), out_ub, ch * VL, mask)

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
        print(f"FAIL r{R}c{C} maxabs={err} scale={float(scale)}", flush=True)
        return 1
    print(f"PASS r{R}c{C} maxabs={err:.3e} scale={float(scale):.6g} compile={cs:.3f}s launch={ls:.3f}s", flush=True)
    return 0


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2:
        print("usage: simd.py R C", flush=True)
        return 2
    return run(int(args[0]), int(args[1]))


if __name__ == "__main__":
    raise SystemExit(main())
