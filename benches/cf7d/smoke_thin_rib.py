"""Thin compile smoke for CF7d reduce-in-b B/C (no hoist)."""
from ptodsl import pto

VL = 64
B = 8
E = 256
NEG_INF_F32 = -3.40282347e38
SCALE = 1.000001
GATE_SUM_THR = 0.0
G = 2
SCORES_BYTES = E * 4
THR_BYTES = B * 4
GATE_BYTES = B * 4
OUT_BYTES = E * 4
OFF_THR = SCORES_BYTES
OFF_GATE = OFF_THR + ((THR_BYTES + 31) // 32) * 32
OFF_OUT = OFF_GATE + ((GATE_BYTES + 31) // 32) * 32
OFF_SCRATCH = OFF_OUT + ((OUT_BYTES + 31) // 32) * 32


@pto.jit(
    name="CF7d_rib_smoke_B_g2",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_B(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    scratch_ub = pto.castptr(pto.const(OFF_SCRATCH, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    mask_B = pto.vmi.create_mask(B, size=B)
    mask1 = pto.vmi.create_mask(1, size=1)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)
    thr_sum_v = pto.vmi.vbrc(pto.f32(GATE_SUM_THR), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(B):
        gate_v = pto.vmi.vload(gate_ub, 0, size=B)
        gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
        pto.vmi.vstore(gs, scratch_ub, 0, mask1)
        gs_s = pto.vmi.vload(scratch_ub, 0, size=1)
        p_outer = pto.vmi.vcmp(pto.vmi.vbrc(gs_s, size=VL), thr_sum_v, mask, "gt")
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(G):
            k0 = pto.vmi.vmul(s0, scale_v, mask)
            k1 = pto.vmi.vmul(s1, scale_v, mask)
            k2 = pto.vmi.vmul(s2, scale_v, mask)
            k3 = pto.vmi.vmul(s3, scale_v, mask)
            k0 = pto.vmi.vsel(pto.vmi.vand(pto.vmi.vcmp(k0, thr_b, mask, "gt"), p_outer), neg_inf, k0)
            k1 = pto.vmi.vsel(pto.vmi.vand(pto.vmi.vcmp(k1, thr_b, mask, "gt"), p_outer), neg_inf, k1)
            k2 = pto.vmi.vsel(pto.vmi.vand(pto.vmi.vcmp(k2, thr_b, mask, "gt"), p_outer), neg_inf, k2)
            k3 = pto.vmi.vsel(pto.vmi.vand(pto.vmi.vcmp(k3, thr_b, mask, "gt"), p_outer), neg_inf, k3)
            s0 = pto.vmi.vsel(p_outer, k0, s0)
            s1 = pto.vmi.vsel(p_outer, k1, s1)
            s2 = pto.vmi.vsel(p_outer, k2, s2)
            s3 = pto.vmi.vsel(p_outer, k3, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))


@pto.jit(
    name="CF7d_rib_smoke_C_g2",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    scratch_ub = pto.castptr(pto.const(OFF_SCRATCH, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    mask_B = pto.vmi.create_mask(B, size=B)
    mask1 = pto.vmi.create_mask(1, size=1)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(B):
        gate_v = pto.vmi.vload(gate_ub, 0, size=B)
        gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
        pto.vmi.vstore(gs, scratch_ub, 0, mask1)
        gs_sc = pto.load_scalar(scratch_ub, 0)
        gate_on = gs_sc > pto.f32(GATE_SUM_THR)
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        with pto.if_(gate_on) as br:
            with br.then_:
                t0, t1, t2, t3 = s0, s1, s2, s3
                for _t in range(G):
                    t0 = pto.vmi.vmul(t0, scale_v, mask)
                    t1 = pto.vmi.vmul(t1, scale_v, mask)
                    t2 = pto.vmi.vmul(t2, scale_v, mask)
                    t3 = pto.vmi.vmul(t3, scale_v, mask)
                    t0 = pto.vmi.vsel(pto.vmi.vcmp(t0, thr_b, mask, "gt"), neg_inf, t0)
                    t1 = pto.vmi.vsel(pto.vmi.vcmp(t1, thr_b, mask, "gt"), neg_inf, t1)
                    t2 = pto.vmi.vsel(pto.vmi.vcmp(t2, thr_b, mask, "gt"), neg_inf, t2)
                    t3 = pto.vmi.vsel(pto.vmi.vcmp(t3, thr_b, mask, "gt"), neg_inf, t3)
                br.assign(s0=t0, s1=t1, s2=t2, s3=t3)
            with br.else_:
                br.assign(s0=s0, s1=s1, s2=s2, s3=s3)
        s0 = br.s0
        s1 = br.s1
        s2 = br.s2
        s3 = br.s3

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))


if __name__ == "__main__":
    import sys
    arm = (sys.argv[1] if len(sys.argv) > 1 else "B").upper()
    k = kernel_B if arm == "B" else kernel_C
    print(f"[rib-smoke] compile arm={arm} ...", flush=True)
    k.compile()
    print(f"COMPILE_OK rib arm={arm}", flush=True)
