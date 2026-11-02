#!/usr/bin/env python3
"""ST-SV8d Case-3 — packed-UB scale path (t548u / Lok hypothesis).

Cut S_scale VSEL paint (baseline RV_VSEL:1024 + VDIV:256 + VDUP:256 on VL=64
KEEP-X recip_vmul SoT) by:

  1. vcmax(group=NG) → compact absmax
  2. continuous compact store to UB
  3. packed / whole-VL VDIV for recip (not size-1 × NG × trips)
  4. expand via vbrc(..., size=VL, group=NG)  [= e2b/VDUP family]
     — FORBID vsel range-paint for scale expand

Arms
----
  pack_ub_vl256
      VL=256 NG=8 gold G=32. Two-pass:
        P1: abs+vcmax(group=8) → compact store all absmax (N_SCALE=256)
        P1b: ONE VL load of packed absmax → vmax(eps) → VL VDIV → store inv
        P2: reload X + load compact NG inv + vbrc(group=NG) + VMUL
      Loses KEEP-X (reload X). Expected VDIV≪256 (ideally 1), VSEL≈0 for expand.

  keeplive_vbrc_vl256
      Same VL/NG. Single-pass KEEP-X:
        abs+vcmax(group=8) → compact vmax+vdiv (size=NG) → vbrc(group=NG) → VMUL
      Optional compact store (debug). Expected VDIV≈N_TRIPS (32), VSEL≈0.

  pack_ub_vl64
      VL=64 NG=2. Same pack idea; known cliff: PTOAS historically rejects
      vcmax(group=2) on VL=64 (vreg<2xf32>). If compile fails → record
      COMPILE_CLIFF; do not invent numbers.

era=case3-quant-bcast  shape=R×C G32 E=R*C (t573u tile-sens; pack allows N_SCALE<=VL)  reduce_axis=inne
Tags: sv8d_pack_r{R}_c{C}_g{G}_t{T}_{arm}
"""


import argparse
import ctypes
import time

import numpy as np
from ptodsl import pto

EPS = 1e-6
T_DEFAULT = 32
SEED = 8

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2

ARMS = ("pack_ub_vl256", "keeplive_vbrc_vl256", "pack_ub_vl64")


def _vl_ng(arm: str, G: int):
    if arm.endswith("vl256"):
        VL = 256
    elif arm.endswith("vl64"):
        VL = 64
    else:
        raise ValueError(arm)
    assert VL % G == 0
    NG = VL // G
    return VL, NG


def _layout(R: int, C: int, G: int, VL: int):
    assert C % G == 0
    assert (R * C) % VL == 0
    NG = VL // G
    N_TRIPS = (R * C) // VL
    N_SCALE = R * (C // G)  # == N_TRIPS * NG
    x_bytes = R * C * 4
    out_bytes = R * C * 4
    off_out = x_bytes
    off_scale = x_bytes + out_bytes  # compact absmax / inv, N_SCALE f32
    return NG, N_TRIPS, N_SCALE, x_bytes, out_bytes, off_out, off_scale


def _kernel_pack_ub_vl256(R: int, C: int, G: int):
    """Lok primary: compact absmax→UB → batch VL-DIV → e2b expand."""
    VL, NG = 256, 256 // G
    NG2, N_TRIPS, N_SCALE, x_bytes, out_bytes, off_out, off_scale = _layout(R, C, G, VL)
    assert NG == NG2
    # t573u tile-sens: allow N_SCALE <= VL (pad unused lanes). SoT E=8192 still N_SCALE==VL.
    assert N_SCALE <= VL, f"pack_ub_vl256 wants N_SCALE<=VL got {N_SCALE} vs {VL}"
    assert VL % N_SCALE == 0 or N_SCALE == VL or True  # pad path always OK for N_SCALE<VL

    @pto.jit(
        name=f"SV8d_pack_r{R}_c{C}_g{G}_pack_ub_vl256",
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
        scale_ub = pto.castptr(pto.const(off_scale, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

        pto.mte_load(x_gm, x_ub, 0, x_bytes, nburst=(1, x_bytes, x_bytes))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        # t573u: prefill scale scratch with EPS so N_SCALE<VL pad lanes are safe
        if N_SCALE < VL:
            _m = pto.vmi.create_mask(VL, size=VL)
            pto.vmi.vstore(pto.vmi.vbrc(pto.f32(EPS), size=VL), scale_ub, 0, mask=_m)
            pto.mem_bar("VST_VLD")

        # --- P1: reduce → compact absmax continuously to UB ---
        with pto.for_(0, N_TRIPS, step=1) as i:
            # masks inside loop (Q-VMI-01: no hoist — pset external users)
            mask_vl = pto.vmi.create_mask(VL, size=VL)
            clear_u = pto.vmi.vbrc(pto.i32(0x7FFFFFFF), size=VL)
            x = pto.vmi.vload(x_ub, i * VL, size=VL)
            xi = pto.vmi.vinterpret_cast(x, pto.i32)
            abs_i = pto.vmi.vand(xi, clear_u, mask_vl)
            a = pto.vmi.vinterpret_cast(abs_i, pto.f32)
            m = pto.vmi.vcmax(a, mask_vl, group=NG)
            pto.vmi.vstore(m, scale_ub, i * NG, stride=1, group=NG)

        pto.mem_bar("VST_VLD")

        # --- P1b: whole-VL packed scale compute (ONE VL VDIV; pad if N_SCALE<VL) ---
        mask_s = pto.vmi.create_mask(VL, size=VL)
        eps_s = pto.vmi.vbrc(pto.f32(EPS), size=VL)
        one_s = pto.vmi.vbrc(pto.f32(1.0), size=VL)
        packed = pto.vmi.vload(scale_ub, 0, size=VL)
        packed = pto.vmi.vmax(packed, eps_s, mask_s)
        inv_all = pto.vmi.vdiv(one_s, packed, mask_s)
        pto.vmi.vstore(inv_all, scale_ub, 0, mask_s)
        pto.mem_bar("VST_VLD")

        # --- P2: reload X + compact inv → vbrc(group=NG) expand (no VSEL) + VMUL ---
        with pto.for_(0, N_TRIPS, step=1) as i:
            mask_vl = pto.vmi.create_mask(VL, size=VL)
            x = pto.vmi.vload(x_ub, i * VL, size=VL)
            # compact reload of NG inv scales for this trip
            inv_c = pto.vmi.vload(scale_ub, i * NG, size=NG)
            # e2b / group-BRC: each of NG scalars → G=VL/NG lanes
            sc = pto.vmi.vbrc(inv_c, size=VL, group=NG)
            y = pto.vmi.vmul(x, sc, mask_vl)
            pto.vmi.vstore(y, out_ub, i * VL, mask_vl)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, out_bytes, nburst=(1, out_bytes, out_bytes))

    return kernel


def _kernel_keeplive_vbrc_vl256(R: int, C: int, G: int):
    """KEEP-X single-pass: compact DIV + vbrc(group=NG) — no VSEL paint."""
    VL, NG = 256, 256 // G
    _, N_TRIPS, _, x_bytes, out_bytes, off_out, off_scale = _layout(R, C, G, VL)

    @pto.jit(
        name=f"SV8d_pack_r{R}_c{C}_g{G}_keeplive_vbrc_vl256",
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
        scale_ub = pto.castptr(pto.const(off_scale, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

        pto.mte_load(x_gm, x_ub, 0, x_bytes, nburst=(1, x_bytes, x_bytes))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        with pto.for_(0, N_TRIPS, step=1) as i:
            mask_vl = pto.vmi.create_mask(VL, size=VL)
            gmask = pto.vmi.create_mask(NG, size=NG)
            clear_u = pto.vmi.vbrc(pto.i32(0x7FFFFFFF), size=VL)
            eps_g = pto.vmi.vbrc(pto.f32(EPS), size=NG)
            one_g = pto.vmi.vbrc(pto.f32(1.0), size=NG)

            x = pto.vmi.vload(x_ub, i * VL, size=VL)
            xi = pto.vmi.vinterpret_cast(x, pto.i32)
            abs_i = pto.vmi.vand(xi, clear_u, mask_vl)
            a = pto.vmi.vinterpret_cast(abs_i, pto.f32)
            m = pto.vmi.vcmax(a, mask_vl, group=NG)
            m = pto.vmi.vmax(m, eps_g, gmask)
            inv = pto.vmi.vdiv(one_g, m, gmask)
            # continuous compact store (Lok: save reduced → UB); SSA also keeps inv
            pto.vmi.vstore(inv, scale_ub, i * NG, stride=1, group=NG)
            # expand WITHOUT VSEL range paint
            sc = pto.vmi.vbrc(inv, size=VL, group=NG)
            y = pto.vmi.vmul(x, sc, mask_vl)
            pto.vmi.vstore(y, out_ub, i * VL, mask_vl)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, out_bytes, nburst=(1, out_bytes, out_bytes))

    return kernel


def _kernel_pack_ub_vl64(R: int, C: int, G: int):
    """Optional VL=64 NG=2 pack. May COMPILE_CLIFF on vcmax(group=2)."""
    VL, NG = 64, 64 // G
    _, N_TRIPS, N_SCALE, x_bytes, out_bytes, off_out, off_scale = _layout(R, C, G, VL)
    # N_SCALE=256, VL=64 → 4 VL-DIV batches
    N_BATCH = N_SCALE // VL
    assert N_SCALE % VL == 0

    @pto.jit(
        name=f"SV8d_pack_r{R}_c{C}_g{G}_pack_ub_vl64",
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
        scale_ub = pto.castptr(pto.const(off_scale, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

        pto.mte_load(x_gm, x_ub, 0, x_bytes, nburst=(1, x_bytes, x_bytes))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        with pto.for_(0, N_TRIPS, step=1) as i:
            mask_vl = pto.vmi.create_mask(VL, size=VL)
            clear_u = pto.vmi.vbrc(pto.i32(0x7FFFFFFF), size=VL)
            x = pto.vmi.vload(x_ub, i * VL, size=VL)
            xi = pto.vmi.vinterpret_cast(x, pto.i32)
            abs_i = pto.vmi.vand(xi, clear_u, mask_vl)
            a = pto.vmi.vinterpret_cast(abs_i, pto.f32)
            # known cliff if NG=2 on VL=64
            m = pto.vmi.vcmax(a, mask_vl, group=NG)
            pto.vmi.vstore(m, scale_ub, i * NG, stride=1, group=NG)

        pto.mem_bar("VST_VLD")

        with pto.for_(0, N_BATCH, step=1) as b:
            mask_s = pto.vmi.create_mask(VL, size=VL)
            eps_s = pto.vmi.vbrc(pto.f32(EPS), size=VL)
            one_s = pto.vmi.vbrc(pto.f32(1.0), size=VL)
            packed = pto.vmi.vload(scale_ub, b * VL, size=VL)
            packed = pto.vmi.vmax(packed, eps_s, mask_s)
            inv_all = pto.vmi.vdiv(one_s, packed, mask_s)
            pto.vmi.vstore(inv_all, scale_ub, b * VL, mask_s)

        pto.mem_bar("VST_VLD")

        with pto.for_(0, N_TRIPS, step=1) as i:
            mask_vl = pto.vmi.create_mask(VL, size=VL)
            x = pto.vmi.vload(x_ub, i * VL, size=VL)
            inv_c = pto.vmi.vload(scale_ub, i * NG, size=NG)
            sc = pto.vmi.vbrc(inv_c, size=VL, group=NG)
            y = pto.vmi.vmul(x, sc, mask_vl)
            pto.vmi.vstore(y, out_ub, i * VL, mask_vl)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, out_bytes, nburst=(1, out_bytes, out_bytes))

    return kernel


def _kernel(R: int, C: int, G: int, arm: str):
    if arm == "pack_ub_vl256":
        return _kernel_pack_ub_vl256(R, C, G)
    if arm == "keeplive_vbrc_vl256":
        return _kernel_keeplive_vbrc_vl256(R, C, G)
    if arm == "pack_ub_vl64":
        return _kernel_pack_ub_vl64(R, C, G)
    raise ValueError(arm)


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


def run_acl(kernel, tag, R, C, G, VL) -> int:
    _, _, _, x_bytes, out_bytes, _, _ = _layout(R, C, G, VL)
    rng = np.random.RandomState(SEED)
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
    ap.add_argument("arm", nargs="?", default="pack_ub_vl256", choices=ARMS)
    ap.add_argument("--emit-mlir", action="store_true")
    args = ap.parse_args(argv)
    if args.C % args.G != 0:
        raise SystemExit(f"C={args.C} must be divisible by G={args.G}")
    VL, NG = _vl_ng(args.arm, args.G)
    if VL % args.G != 0:
        raise SystemExit(f"VL={VL} must be divisible by G={args.G}")
    if (args.R * args.C) % VL != 0:
        raise SystemExit(f"R*C={args.R*args.C} must be divisible by VL={VL}")
    tag = f"sv8d_pack_r{args.R}_c{args.C}_g{args.G}_t{args.T}_{args.arm}"
    print(
        f"BUILD {tag} VL={VL} NG={NG} N_TRIPS={(args.R*args.C)//VL} "
        f"N_SCALE={args.R*(args.C//args.G)} era=case3-quant-bcast reduce_axis=inner",
        flush=True,
    )
    kernel = _kernel(args.R, args.C, args.G, args.arm)
    if args.emit_mlir:
        print(kernel.compile().mlir_text())
        return 0
    return run_acl(kernel, tag, args.R, args.C, args.G, VL)


if __name__ == "__main__":
    raise SystemExit(main())
