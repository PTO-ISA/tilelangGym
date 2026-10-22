#!/usr/bin/env python3
"""SV5 SIMD 8K absmax — VL=128 / group=8 (one trip per row).

Store modes (CLI --store):
  B  group-mode: vstore(red, y_ub, i*CG, group=8, stride=1)  NO mask  [preferred]
  A  mask compact: vstore(red, y_ub, i*CG, mask_ng)

Gold: numpy absmax per G=16; gather compact raw[:, :CG].
"""
import argparse, ctypes, os, time
import numpy as np
from ptodsl import pto

PHYS_VL = 64
VL = 128
SLOT = 8

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

def build(R, C, G, store_mode):
    CG = C // G
    NG = VL // G  # 8
    assert VL == C and NG == CG
    x_bytes = R * C * 4
    y_stride = CG  # compact 8
    y_bytes = R * y_stride * 4
    tag = f"SV5_simd8k_absmax_vl128_g{NG}_store{store_mode}_r{R}"

    @pto.jit(name=tag, target="a5", backend="vpto", mode="explicit",
             kernel_kind="vector", insert_sync=False, ast_rewrite=False)
    def kernel(y_gm: pto.ptr(pto.f32, "gm"), x_gm: pto.ptr(pto.f32, "gm")):
        x_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        y_ub = pto.castptr(pto.const(x_bytes, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        pto.mte_load(x_gm, x_ub, 0, x_bytes, nburst=(1, x_bytes, x_bytes))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        mask = pto.vmi.create_mask(VL, size=VL)  # fix: full 128 lanes (was PHYS_VL=64)
        mask_ng = pto.vmi.create_mask(NG, size=NG)
        # SoftLib-style abs: VAND clear-sign via i32 bitcast
        # (pto.vmi.vand requires integer-like element type; f32 vand fails verify)
        clear_u = pto.vmi.vbrc(pto.i32(0x7FFFFFFF), size=VL)
        with pto.for_(0, R, step=1) as i:
            x = pto.vmi.vload(x_ub, i * C, size=VL)
            xi = pto.vmi.vinterpret_cast(x, pto.i32)
            abs_i = pto.vmi.vand(xi, clear_u, mask)
            absx = pto.vmi.vinterpret_cast(abs_i, pto.f32)
            red = pto.vmi.vcmax(absx, mask, group=NG)  # group=8 → vcgmax
            if store_mode == "B":
                # group-mode pack: no mask
                pto.vmi.vstore(red, y_ub, i * CG, group=NG, stride=1)
            else:
                # A: mask compact (Fallback-A style mask form)
                pto.vmi.vstore(red, y_ub, i * CG, mask_ng)
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(y_ub, y_gm, y_bytes, nburst=(1, y_bytes, y_bytes))
    return kernel, y_bytes, y_stride, CG, NG, tag

def run(R=64, C=128, G=16, store_mode="B", out_dir=None):
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    CG = C // G
    kernel, y_bytes, y_stride, CG, NG, tag = build(R, C, G, store_mode)
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
    mlir = ""
    try:
        mlir = comp.mlir_text()
        if out_dir:
            open(os.path.join(out_dir, f"{tag}.mlir"), "w").write(mlir)
    except Exception as e:
        print(f"mlir_text failed: {e}", flush=True)
    t0 = time.perf_counter(); comp[1, st](int(yd.value), int(xd.value))
    _chk(lib.aclrtSynchronizeStream(st), "sync"); ls = time.perf_counter() - t0
    raw = np.zeros((R, y_stride), dtype=np.float32)
    _chk(lib.aclrtMemcpy(raw.ctypes.data_as(ctypes.c_void_p), y_bytes, yd, y_bytes, 2), "d2h")
    lib.aclrtFree(xd); lib.aclrtFree(yd); lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0); lib.aclFinalize()

    # gathers: compact + every-SLOT
    yh_c = raw[:, :CG].copy()
    yh_slot = np.zeros((R, CG), dtype=np.float32)
    for i in range(R):
        for k in range(CG):
            idx = k * SLOT
            if idx < y_stride:
                yh_slot[i, k] = raw[i, idx]
    mad_c = float(np.max(np.abs(yh_c - ref)))
    mad_s = float(np.max(np.abs(yh_slot - ref)))
    ok_c = np.allclose(yh_c, ref, atol=1e-4, rtol=1e-4)
    ok_s = np.allclose(yh_slot, ref, atol=1e-4, rtol=1e-4)
    nnz = int(np.count_nonzero(raw))
    nz0 = np.nonzero(raw[0])[0][:48].tolist()
    print(f"raw0[:8]={raw[0,:8].tolist()} nnz={nnz} nz0_idx={nz0}", flush=True)
    print(f"yh_c0={yh_c[0].tolist()} ref0={ref[0].tolist()} mad_c={mad_c:.3e} ok_c={ok_c}", flush=True)
    print(f"yh_slot0={yh_slot[0].tolist()} mad_s={mad_s:.3e} ok_s={ok_s}", flush=True)
    if out_dir:
        np.save(os.path.join(out_dir, f"raw_store{store_mode}.npy"), raw)
        open(os.path.join(out_dir, f"report_store{store_mode}.txt"), "w").write(
            f"tag={tag} store={store_mode} compile={cs:.3f}s launch={ls:.3f}s\n"
            f"nnz={nnz} nz0={nz0}\nraw0[:16]={raw[0,:16].tolist()}\n"
            f"hex={raw[0,:min(16,y_stride)].tobytes().hex()}\n"
            f"mad_c={mad_c:.3e} ok_c={ok_c}\nmad_s={mad_s:.3e} ok_s={ok_s}\n"
            f"ref0={ref[0].tolist()}\nyh_c0={yh_c[0].tolist()}\n"
        )
        if mlir:
            keys = [ln.strip() for ln in mlir.splitlines()
                    if any(k in ln for k in ("vcmax", "vstore", "layout", "slots", "group", "mask", "vreg"))]
            open(os.path.join(out_dir, f"mlir_types_store{store_mode}.txt"), "w").write("\n".join(keys[:80]) + "\n")
    if ok_c or ok_s:
        which = "compact" if ok_c else "every_slot"
        maxabs = mad_c if ok_c else mad_s
        print(
            f"PASS {tag} gather={which} compile={cs:.3f}s launch={ls:.3f}s "
            f"maxabs={maxabs:.3e} E={R*C} VL={VL} group={NG}",
            flush=True,
        )
        return 0
    print(
        f"FAIL_GOLD {tag} compile={cs:.3f}s launch={ls:.3f}s "
        f"mad_c={mad_c:.3e} mad_s={mad_s:.3e} nnz={nnz} E={R*C} VL={VL} group={NG}",
        flush=True,
    )
    # Do NOT raise / nonzero-exit: that aborts msprof before instr dumps /
    # instr_exe.csv flush (empty dumps + VF_cyc=0). Exit 0 so sim_dsl finishes
    # OPPROF; gold status remains FAIL_GOLD in the printed log line above.
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-mlir", action="store_true")
    ap.add_argument("--store", choices=["A", "B"], default="B")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("R", nargs="?", type=int, default=64)
    ap.add_argument("C", nargs="?", type=int, default=128)
    ap.add_argument("G", nargs="?", type=int, default=16)
    args, _ = ap.parse_known_args()
    if args.emit_mlir:
        k, *_ = build(args.R, args.C, args.G, args.store)
        print(k.compile().mlir_text()); return 0
    return run(args.R, args.C, args.G, store_mode=args.store, out_dir=args.out_dir)

if __name__ == "__main__":
    raise SystemExit(main())
