#!/usr/bin/env python3
"""SV5 SIMD 8K absmax — VMI vcmax(group=4) → PTOAS vcgmax (Fallback A).

R=64 C=128 G=16 E=8192. VL=64 × 2 trips/row. no vdiv.
Store uses trip spacing NG*SLOT so group_store packs; gold gathers
contiguous-within-trip at raw[i, tr*NG*SLOT + k].
"""
import argparse, ctypes, time
import numpy as np
from ptodsl import pto

PHYS_VL = 64
SLOT = 8  # PTOAS group_store layout footprint per trip's NG groups
_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2

def reference(X, R, C, G):
    CG = C // G
    Y = np.empty((R, CG), dtype=np.float32)
    for i in range(R):
        for g in range(CG):
            Y[i, g] = float(np.max(np.abs(X[i, g * G : (g + 1) * G].astype(np.float32))))
    return Y

def _load_acl():
    lib = ctypes.CDLL("libascendcl.so")
    for n, t in [("aclInit", ctypes.c_char_p), ("aclFinalize", None),
                 ("aclrtSetDevice", ctypes.c_int), ("aclrtResetDevice", ctypes.c_int)]:
        getattr(lib, n).argtypes = [t] if t else []; getattr(lib, n).restype = ctypes.c_int
    lib.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]; lib.aclrtCreateStream.restype = ctypes.c_int
    lib.aclrtDestroyStream.argtypes = [ctypes.c_void_p]; lib.aclrtDestroyStream.restype = ctypes.c_int
    lib.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]; lib.aclrtMalloc.restype = ctypes.c_int
    lib.aclrtFree.argtypes = [ctypes.c_void_p]; lib.aclrtFree.restype = ctypes.c_int
    lib.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]; lib.aclrtMemcpy.restype = ctypes.c_int
    lib.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]; lib.aclrtSynchronizeStream.restype = ctypes.c_int
    return lib

def _chk(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed {rc}")

def build(R, C, G):
    CG = C // G
    x_bytes = R * C * 4
    NG = PHYS_VL // G
    n_trips = C // PHYS_VL
    y_stride = n_trips * NG * SLOT  # 64
    y_bytes = R * y_stride * 4

    @pto.jit(name=f"SV5_simd8k_absmax_vcg_r{R}_c{C}_g{G}_vl64x2",
             target="a5", backend="vpto", mode="explicit",
             kernel_kind="vector", insert_sync=False, ast_rewrite=False)
    def kernel(y_gm: pto.ptr(pto.f32, "gm"), x_gm: pto.ptr(pto.f32, "gm")):
        x_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        y_ub = pto.castptr(pto.const(x_bytes, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        pto.mte_load(x_gm, x_ub, 0, x_bytes, nburst=(1, x_bytes, x_bytes))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        VL = PHYS_VL
        mask = pto.vmi.create_mask(PHYS_VL, size=VL)
        mask_ng = pto.vmi.create_mask(NG, size=NG)
        # SoftLib-style abs: VAND clear-sign via i32 bitcast
        # (pto.vmi.vand requires integer-like element type; f32 vand fails verify)
        clear_u = pto.vmi.vbrc(pto.i32(0x7FFFFFFF), size=VL)
        with pto.for_(0, R, step=1) as i:
            for tr in range(n_trips):
                x = pto.vmi.vload(x_ub, i * C + tr * VL, size=VL)
                xi = pto.vmi.vinterpret_cast(x, pto.i32)
                abs_i = pto.vmi.vand(xi, clear_u, mask)
                absx = pto.vmi.vinterpret_cast(abs_i, pto.f32)
                red = pto.vmi.vcmax(absx, mask, group=NG)  # → vcgmax
                pto.vmi.vstore(red, y_ub, i * y_stride + tr * NG * SLOT, mask_ng)
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(y_ub, y_gm, y_bytes, nburst=(1, y_bytes, y_bytes))
    return kernel, y_bytes, y_stride, NG, n_trips

def run(R=64, C=128, G=16):
    CG = C // G
    kernel, y_bytes, y_stride, NG, n_trips = build(R, C, G)
    x_bytes = R * C * 4
    rng = np.random.RandomState(5)
    X = rng.uniform(-3, 3, (R, C)).astype(np.float32)
    ref = reference(X, R, C, G)
    lib = _load_acl(); _chk(lib.aclInit(None), "init"); _chk(lib.aclrtSetDevice(0), "dev")
    st = ctypes.c_void_p(); _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")
    xd = ctypes.c_void_p(); yd = ctypes.c_void_p()
    _chk(lib.aclrtMalloc(ctypes.byref(xd), x_bytes, 0), "mx")
    _chk(lib.aclrtMalloc(ctypes.byref(yd), y_bytes, 0), "my")
    _chk(lib.aclrtMemcpy(xd, x_bytes, np.ascontiguousarray(X).ctypes.data_as(ctypes.c_void_p), x_bytes, 1), "h2d")
    t0 = time.perf_counter(); comp = kernel.compile(); cs = time.perf_counter() - t0
    t0 = time.perf_counter(); comp[1, st](int(yd.value), int(xd.value))
    _chk(lib.aclrtSynchronizeStream(st), "sync"); ls = time.perf_counter() - t0
    raw = np.zeros((R, y_stride), dtype=np.float32)
    _chk(lib.aclrtMemcpy(raw.ctypes.data_as(ctypes.c_void_p), y_bytes, yd, y_bytes, 2), "d2h")
    yh = np.empty((R, CG), dtype=np.float32)
    for i in range(R):
        for tr in range(n_trips):
            base = tr * NG * SLOT
            for k in range(NG):
                yh[i, tr * NG + k] = raw[i, base + k]
    lib.aclrtFree(xd); lib.aclrtFree(yd); lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0); lib.aclFinalize()
    maxabs = float(np.max(np.abs(yh - ref)))
    print(f"raw0[:8]={raw[0,:8].tolist()} raw0[32:40]={raw[0,32:40].tolist()}", flush=True)
    print(f"yh[:1]={yh[:1].tolist()} ref[:1]={ref[:1].tolist()} maxabs={maxabs:.3e}", flush=True)
    if not np.allclose(yh, ref, atol=1e-4, rtol=1e-4):
        print(
            f"FAIL_GOLD sv5d_absmax_vcg_r{R}_c{C}_g{G}_vl64x2  compile={cs:.3f}s launch={ls:.3f}s "
            f"maxabs={maxabs:.3e} E={R*C} VL=64 group={NG} trips={n_trips}",
            flush=True,
        )
        # exit 0 so msprof flushes instr dumps / instr_exe.csv
        return 0
    print(
        f"PASS sv5d_absmax_vcg_r{R}_c{C}_g{G}_vl64x2  compile={cs:.3f}s launch={ls:.3f}s "
        f"maxabs={maxabs:.3e} E={R*C} VL=64 group={NG} trips={n_trips}",
        flush=True,
    )
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-mlir", action="store_true")
    ap.add_argument("R", nargs="?", type=int, default=64)
    ap.add_argument("C", nargs="?", type=int, default=128)
    ap.add_argument("G", nargs="?", type=int, default=16)
    args, _ = ap.parse_known_args()
    if args.emit_mlir:
        k, *_ = build(args.R, args.C, args.G)
        print(k.compile().mlir_text()); return 0
    return run(args.R, args.C, args.G)

if __name__ == "__main__":
    raise SystemExit(main())
