#!/usr/bin/env python3
"""ST-SV8d VL-resident — Case-3 quant bcast with X kept in VL vreg.

Tags: sv8d_vlres_r{R}_c{C}_g{G}_t{T}_vl_resident

Motif (matches H2 retest — NOT the failing size-1 + scratch-expand path):
  - VL-load X once per trip (VL=64); KEEP across absmax → recip expand → mul
  - Abs: i32 bitcast + VAND clear-sign (Harness §18 required try)
  - Reduce: G32 on VL=64 uses per-group vcmax(group=1) (NG=2 compact is PTOAS-illegal);
  G16 may use vcmax(group=4). Inner reduce_axis, §22
  - Recip on compact NG scales; expand NG→VL via ONE compact VST + mem_bar +
    vbrc(vload size=1) + vsel range masks (no per-group VL broadcast VST)
  - Mul uses KEPT x (no second full-X VLDS)

Expected instr (S3): VCGMAX family for reduce; VLDS_X ≈ R*(C/VL) not E;
  SMEM_BAR ≪ prior 1160; no size-1 VMAX tree over E.
"""


import argparse
import ctypes
import time

import numpy as np
from ptodsl import pto

EPS = 1e-6
T_DEFAULT = 32
VL = 64

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


def _layout(R: int, C: int, G: int):
    assert C % VL == 0
    assert VL % G == 0
    NG = VL // G
    x_bytes = R * C * 4
    out_bytes = R * C * 4
    off_out = x_bytes
    # compact scale scratch: NG floats (one VL-trip's group scales)
    off_scratch = x_bytes + out_bytes
    return NG, x_bytes, out_bytes, off_out, off_scratch


def _kernel_vl_resident(R: int, C: int, G: int):
    NG, x_bytes, out_bytes, off_out, off_scratch = _layout(R, C, G)
    NCH = C // VL

    @pto.jit(
        name=f"SV8d_vlres_r{R}_c{C}_g{G}_vl_resident",
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
        scratch_ub = pto.castptr(pto.const(off_scratch, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

        pto.mte_load(x_gm, x_ub, 0, x_bytes, nburst=(1, x_bytes, x_bytes))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        mask_vl = pto.vmi.create_mask(VL, size=VL)
        clear_u = pto.vmi.vbrc(pto.i32(0x7FFFFFFF), size=VL)
        eps_vl = pto.vmi.vbrc(pto.f32(EPS), size=VL)
        one_vl = pto.vmi.vbrc(pto.f32(1.0), size=VL)
        arange = pto.vmi.vci(pto.i32(0), size=VL)
        zero_vl = pto.vmi.vbrc(pto.f32(0.0), size=VL)
        # size-1 mask for per-group vcmax(group=1) path (G32 NG=2 illegal compact)
        mask1 = pto.vmi.create_mask(1, size=1)
        eps1 = pto.vmi.vbrc(pto.f32(EPS), size=1)
        one1 = pto.vmi.vbrc(pto.f32(1.0), size=1)

        with pto.for_(0, R, step=1) as i:
            for ch in range(NCH):
                # --- KEEP X in VL vreg across reduce → mul ---
                x = pto.vmi.vload(x_ub, i * C + ch * VL, size=VL)

                # Abs: i32-VAND clear-sign
                xi = pto.vmi.vinterpret_cast(x, pto.i32)
                abs_i = pto.vmi.vand(xi, clear_u, mask_vl)
                absx = pto.vmi.vinterpret_cast(abs_i, pto.f32)

                # Per-group reduce: mask other lanes to 0, vcmax(group=1) → size-1
                # (PTOAS rejects vreg<2xf32> from vcmax group=NG=2 on VL=64)
                sc = zero_vl
                for gi in range(NG):
                    lo = gi * G
                    hi = lo + G
                    ge = pto.vmi.vcmp(arange, pto.vmi.vbrc(pto.i32(lo), size=VL), mask_vl, "ge")
                    lt = pto.vmi.vcmp(arange, pto.vmi.vbrc(pto.i32(hi), size=VL), mask_vl, "lt")
                    # in-group abs; out-of-group → 0 (safe for absmax)
                    mid_m = pto.vmi.vsel(ge, absx, zero_vl)
                    in_abs = pto.vmi.vsel(lt, mid_m, zero_vl)
                    red1 = pto.vmi.vcmax(in_abs, mask_vl, group=1)
                    red1 = pto.vmi.vmax(red1, eps1, mask1)
                    inv1 = pto.vmi.vdiv(one1, red1, mask1)
                    sb = pto.vmi.vbrc(inv1, size=VL)
                    mid = pto.vmi.vsel(ge, sb, sc)
                    sc = pto.vmi.vsel(lt, mid, sc)

                # Mul with KEPT x — no second X load from UB
                pto.vmi.vstore(pto.vmi.vmul(x, sc, mask_vl), out_ub, i * C + ch * VL, mask_vl)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, out_bytes, nburst=(1, out_bytes, out_bytes))

    return kernel


def reference(X, R, C, G):
    CG = C // G
    Out = np.empty_like(X)
    for i in range(R):
        for g in range(CG):
            chunk = X[i, g * G : (g + 1) * G].astype(np.float32)
            m = max(float(np.max(np.abs(chunk))), EPS)
            inv = np.float32(1.0 / m)
            Out[i, g * G : (g + 1) * G] = (chunk * inv).astype(np.float32)
    return Out


def _load_acl():
    lib = ctypes.CDLL("libascendcl.so")
    for n, t in [
        ("aclInit", ctypes.c_char_p),
        ("aclFinalize", None),
        ("aclrtSetDevice", ctypes.c_int),
        ("aclrtResetDevice", ctypes.c_int),
    ]:
        getattr(lib, n).argtypes = [t] if t else []
        getattr(lib, n).restype = ctypes.c_int
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


def run_acl(kernel, tag, R, C, G) -> int:
    _, x_bytes, out_bytes, _, _ = _layout(R, C, G)
    rng = np.random.RandomState(8)
    X = rng.uniform(-3, 3, (R, C)).astype(np.float32)
    ref = reference(X, R, C, G)
    lib = _load_acl()
    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    st = ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")
    xd = ctypes.c_void_p()
    od = ctypes.c_void_p()
    _chk(lib.aclrtMalloc(ctypes.byref(xd), x_bytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mx")
    _chk(lib.aclrtMalloc(ctypes.byref(od), out_bytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mo")
    _chk(
        lib.aclrtMemcpy(
            xd, x_bytes, np.ascontiguousarray(X).ctypes.data_as(ctypes.c_void_p), x_bytes,
            _ACL_MEMCPY_HOST_TO_DEVICE,
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
        lib.aclrtMemcpy(
            oh.ctypes.data_as(ctypes.c_void_p), out_bytes, od, out_bytes,
            _ACL_MEMCPY_DEVICE_TO_HOST,
        ),
        "d2h",
    )
    lib.aclrtFree(xd)
    lib.aclrtFree(od)
    lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0)
    lib.aclFinalize()
    maxabs = float(np.max(np.abs(oh - ref)))
    if not np.allclose(oh, ref, atol=1e-4, rtol=1e-4):
        raise AssertionError(f"mismatch max_diff={maxabs}")
    print(f"PASS {tag}  compile={cs:.3f}s launch={ls:.3f}s maxabs={maxabs:.3e}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("R", nargs="?", type=int, default=64)
    ap.add_argument("C", nargs="?", type=int, default=128)
    ap.add_argument("G", nargs="?", type=int, default=32)
    ap.add_argument("T", nargs="?", type=int, default=T_DEFAULT)
    ap.add_argument("arm", nargs="?", default="vl_resident", choices=("vl_resident",))
    ap.add_argument("--emit-mlir", action="store_true")
    args = ap.parse_args(argv)
    if args.C % args.G != 0:
        raise SystemExit(f"C={args.C} must be divisible by G={args.G}")
    if VL % args.G != 0:
        raise SystemExit(f"VL={VL} must be divisible by G={args.G}")
    tag = f"sv8d_vlres_r{args.R}_c{args.C}_g{args.G}_t{args.T}_{args.arm}"
    kernel = _kernel_vl_resident(args.R, args.C, args.G)
    if args.emit_mlir:
        print(kernel.compile().mlir_text())
        return 0
    return run_acl(kernel, tag, args.R, args.C, args.G)


if __name__ == "__main__":
    raise SystemExit(main())
