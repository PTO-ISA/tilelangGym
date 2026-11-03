#!/usr/bin/env python3
"""ST-CF7d sibling — reduce-in-b (no hoist) global gate × inner kill (arms B/C).

Case id: cf7d_reduce_in_b / rib
Tag: cf7d_rib_e{E}_b{B}_g{G}_vl{VL}_pgate{P}_arm{B|C}

Motivation (Lok): hoisted CF7d computes gate_on once then reuses across B trips,
so arm C can fake a win (reduce tax paid once). This sibling recomputes the SAME
sum(gate)>GATE_SUM_THR inside each outer `b` iteration so both arms pay reduce
dep every trip. Numerically identical gold to hoisted global-gate; schedule differs.

Keep benches/cf7d/cf7d_outer_inner_if.py (hoisted) unchanged.
"""

import argparse
import ctypes
import time

import numpy as np
from ptodsl import pto

E = 256
B = 8
VL = 64
N_CHUNKS = E // VL
GATE_SUM_THR = 0.0
SCALE = 1.000001
NEG_INF_F32 = -3.40282347e38

SCORES_BYTES = E * 4
THR_BYTES = B * 4
GATE_BYTES = B * 4
OUT_BYTES = E * 4
OFF_THR = SCORES_BYTES
OFF_GATE = OFF_THR + ((THR_BYTES + 31) // 32) * 32
OFF_OUT = OFF_GATE + ((GATE_BYTES + 31) // 32) * 32
OFF_SCRATCH = OFF_OUT + ((OUT_BYTES + 31) // 32) * 32

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


def gate_on_for(pgate: int) -> bool:
    """Map legacy pgate label → global on/off. on ≈ pgate>=50; off otherwise."""
    return int(pgate) >= 50



@pto.jit(
    name="CF7d_rib_armB_g0",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_B_g0(
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

    # NO hoist: same sum(gate)>THR recomputed inside each b trip
    for b in range(8):  # B fixed — stable overhead; reduce tax × B
        gate_v = pto.vmi.vload(gate_ub, 0, size=B)
        gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
        pto.vmi.vstore(gs, scratch_ub, 0, mask1)
        gs_s = pto.vmi.vload(scratch_ub, 0, size=1)
        p_outer = pto.vmi.vcmp(pto.vmi.vbrc(gs_s, size=VL), thr_sum_v, mask, "gt")
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        pass  # G=0 — per-b reduce tax only (no kill)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))


@pto.jit(
    name="CF7d_rib_armC_g0",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g0(
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

    # NO hoist: reduce + pto.if_ inside each b trip (C pays reduce dep × B)
    for b in range(8):  # B fixed
        gate_v = pto.vmi.vload(gate_ub, 0, size=B)
        gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
        pto.vmi.vstore(gs, scratch_ub, 0, mask1)
        gs_sc = pto.load_scalar(scratch_ub, 0)
        gate_on = gs_sc > pto.f32(GATE_SUM_THR)
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        with pto.if_(gate_on) as br:  # real VF scalar if after per-b reduce
            with br.then_:
                br.assign(s0=s0, s1=s1, s2=s2, s3=s3)  # G=0 — per-b reduce + if_ overhead only
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


@pto.jit(
    name="CF7d_rib_armB_g8",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_B_g8(
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

    # NO hoist: same sum(gate)>THR recomputed inside each b trip
    for b in range(8):  # B fixed — stable overhead; reduce tax × B
        gate_v = pto.vmi.vload(gate_ub, 0, size=B)
        gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
        pto.vmi.vstore(gs, scratch_ub, 0, mask1)
        gs_s = pto.vmi.vload(scratch_ub, 0, size=1)
        p_outer = pto.vmi.vcmp(pto.vmi.vbrc(gs_s, size=VL), thr_sum_v, mask, "gt")
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(8):  # G fat + kill; PAND with per-b p_outer
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
    name="CF7d_rib_armC_g8",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g8(
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

    # NO hoist: reduce + pto.if_ inside each b trip (C pays reduce dep × B)
    for b in range(8):  # B fixed
        gate_v = pto.vmi.vload(gate_ub, 0, size=B)
        gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
        pto.vmi.vstore(gs, scratch_ub, 0, mask1)
        gs_sc = pto.load_scalar(scratch_ub, 0)
        gate_on = gs_sc > pto.f32(GATE_SUM_THR)
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        with pto.if_(gate_on) as br:  # real VF scalar if after per-b reduce
            with br.then_:
                t0, t1, t2, t3 = s0, s1, s2, s3
                for _t in range(8):
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


@pto.jit(
    name="CF7d_rib_armB_g16",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_B_g16(
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

    # NO hoist: same sum(gate)>THR recomputed inside each b trip
    for b in range(8):  # B fixed — stable overhead; reduce tax × B
        gate_v = pto.vmi.vload(gate_ub, 0, size=B)
        gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
        pto.vmi.vstore(gs, scratch_ub, 0, mask1)
        gs_s = pto.vmi.vload(scratch_ub, 0, size=1)
        p_outer = pto.vmi.vcmp(pto.vmi.vbrc(gs_s, size=VL), thr_sum_v, mask, "gt")
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(16):  # G fat + kill; PAND with per-b p_outer
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
    name="CF7d_rib_armC_g16",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g16(
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

    # NO hoist: reduce + pto.if_ inside each b trip (C pays reduce dep × B)
    for b in range(8):  # B fixed
        gate_v = pto.vmi.vload(gate_ub, 0, size=B)
        gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
        pto.vmi.vstore(gs, scratch_ub, 0, mask1)
        gs_sc = pto.load_scalar(scratch_ub, 0)
        gate_on = gs_sc > pto.f32(GATE_SUM_THR)
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        with pto.if_(gate_on) as br:  # real VF scalar if after per-b reduce
            with br.then_:
                t0, t1, t2, t3 = s0, s1, s2, s3
                for _t in range(16):
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


@pto.jit(
    name="CF7d_rib_armB_g32",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_B_g32(
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

    # NO hoist: same sum(gate)>THR recomputed inside each b trip
    for b in range(8):  # B fixed — stable overhead; reduce tax × B
        gate_v = pto.vmi.vload(gate_ub, 0, size=B)
        gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
        pto.vmi.vstore(gs, scratch_ub, 0, mask1)
        gs_s = pto.vmi.vload(scratch_ub, 0, size=1)
        p_outer = pto.vmi.vcmp(pto.vmi.vbrc(gs_s, size=VL), thr_sum_v, mask, "gt")
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(32):  # G fat + kill; PAND with per-b p_outer
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
    name="CF7d_rib_armC_g32",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g32(
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

    # NO hoist: reduce + pto.if_ inside each b trip (C pays reduce dep × B)
    for b in range(8):  # B fixed
        gate_v = pto.vmi.vload(gate_ub, 0, size=B)
        gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
        pto.vmi.vstore(gs, scratch_ub, 0, mask1)
        gs_sc = pto.load_scalar(scratch_ub, 0)
        gate_on = gs_sc > pto.f32(GATE_SUM_THR)
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        with pto.if_(gate_on) as br:  # real VF scalar if after per-b reduce
            with br.then_:
                t0, t1, t2, t3 = s0, s1, s2, s3
                for _t in range(32):
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


@pto.jit(
    name="CF7d_rib_armB_g64",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_B_g64(
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

    # NO hoist: same sum(gate)>THR recomputed inside each b trip
    for b in range(8):  # B fixed — stable overhead; reduce tax × B
        gate_v = pto.vmi.vload(gate_ub, 0, size=B)
        gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
        pto.vmi.vstore(gs, scratch_ub, 0, mask1)
        gs_s = pto.vmi.vload(scratch_ub, 0, size=1)
        p_outer = pto.vmi.vcmp(pto.vmi.vbrc(gs_s, size=VL), thr_sum_v, mask, "gt")
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(64):  # G fat + kill; PAND with per-b p_outer
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
    name="CF7d_rib_armC_g64",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g64(
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

    # NO hoist: reduce + pto.if_ inside each b trip (C pays reduce dep × B)
    for b in range(8):  # B fixed
        gate_v = pto.vmi.vload(gate_ub, 0, size=B)
        gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
        pto.vmi.vstore(gs, scratch_ub, 0, mask1)
        gs_sc = pto.load_scalar(scratch_ub, 0)
        gate_on = gs_sc > pto.f32(GATE_SUM_THR)
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        with pto.if_(gate_on) as br:  # real VF scalar if after per-b reduce
            with br.then_:
                t0, t1, t2, t3 = s0, s1, s2, s3
                for _t in range(64):
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


KERNELS = {
    ("B", 0): kernel_B_g0,
    ("C", 0): kernel_C_g0,
    ("B", 8): kernel_B_g8,
    ("C", 8): kernel_C_g8,
    ("B", 16): kernel_B_g16,
    ("C", 16): kernel_C_g16,
    ("B", 32): kernel_B_g32,
    ("C", 32): kernel_C_g32,
    ("B", 64): kernel_B_g64,
    ("C", 64): kernel_C_g64
}


def build_kernel(arm, G):
    arm = arm.upper()
    if arm == "A":
        raise KeyError("arm A dropped; use B or C (reduce-in-b)")
    key = (arm, int(G))
    if key not in KERNELS:
        raise KeyError(f"no module kernel for {key}; G in {sorted({k[1] for k in KERNELS})}")
    return KERNELS[key]


def reference(scores, thr, gate, G, gate_sum_thr=GATE_SUM_THR):
    """Gold: SAME sum(gate)>thr as hoisted, but written as recompute-per-b.

    for b in range(B):
        gate_on_b = sum(gate) > thr   # identical each trip; must recompute in kernel
        if gate_on_b:
            for _ in range(G):
                scale; kill
    """
    out = scores.astype(np.float32).copy()
    thr = thr.astype(np.float32)
    gate = gate.astype(np.float32)
    scale = np.float32(SCALE)
    for b in range(len(thr)):
        gate_on_b = float(np.sum(gate)) > float(gate_sum_thr)
        if gate_on_b:
            for _ in range(G):
                out = (out * scale).astype(np.float32)
                out = np.where(out > thr[b], np.float32(NEG_INF_F32), out).astype(np.float32)
    return out


def make_inputs(pgate, seed=27):
    rng = np.random.default_rng(seed)
    scores = rng.uniform(-3, 3, E).astype(np.float32)
    thr = rng.uniform(-1, 1, B).astype(np.float32)
    on = gate_on_for(pgate)
    if on:
        gate = rng.uniform(0.1, 2.0, size=B).astype(np.float32)
    else:
        gate = rng.uniform(-2.0, -0.1, size=B).astype(np.float32)
    assert (float(np.sum(gate)) > GATE_SUM_THR) == on
    return scores, thr, gate, on


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
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
    ]
    lib.aclrtMemcpy.restype = ctypes.c_int
    lib.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
    lib.aclrtSynchronizeStream.restype = ctypes.c_int
    return lib


def _chk(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed {rc}")


def run_acl(tag, arm, G, pgate):
    scores, thr, gate, gate_on = make_inputs(pgate)
    ref = reference(scores, thr, gate, G)
    arm = arm.upper()
    kernel = build_kernel(arm, G)

    lib = _load_acl()
    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    st = ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")
    sd = ctypes.c_void_p(); td = ctypes.c_void_p(); gd = ctypes.c_void_p(); od = ctypes.c_void_p()
    _chk(lib.aclrtMalloc(ctypes.byref(sd), SCORES_BYTES, _ACL_MEM_MALLOC_HUGE_FIRST), "ms")
    _chk(lib.aclrtMalloc(ctypes.byref(td), THR_BYTES, _ACL_MEM_MALLOC_HUGE_FIRST), "mt")
    _chk(lib.aclrtMalloc(ctypes.byref(gd), GATE_BYTES, _ACL_MEM_MALLOC_HUGE_FIRST), "mg")
    _chk(lib.aclrtMalloc(ctypes.byref(od), OUT_BYTES, _ACL_MEM_MALLOC_HUGE_FIRST), "mo")
    _chk(lib.aclrtMemcpy(sd, SCORES_BYTES, np.ascontiguousarray(scores).ctypes.data_as(ctypes.c_void_p), SCORES_BYTES, _ACL_MEMCPY_HOST_TO_DEVICE), "h2d_s")
    _chk(lib.aclrtMemcpy(td, THR_BYTES, np.ascontiguousarray(thr).ctypes.data_as(ctypes.c_void_p), THR_BYTES, _ACL_MEMCPY_HOST_TO_DEVICE), "h2d_t")
    _chk(lib.aclrtMemcpy(gd, GATE_BYTES, np.ascontiguousarray(gate).ctypes.data_as(ctypes.c_void_p), GATE_BYTES, _ACL_MEMCPY_HOST_TO_DEVICE), "h2d_g")
    t0 = time.perf_counter()
    comp = kernel.compile()
    cs = time.perf_counter() - t0
    t0 = time.perf_counter()
    comp[1, st](int(od.value), int(sd.value), int(td.value), int(gd.value))
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    ls = time.perf_counter() - t0
    oh = np.zeros(E, dtype=np.float32)
    _chk(lib.aclrtMemcpy(oh.ctypes.data_as(ctypes.c_void_p), OUT_BYTES, od, OUT_BYTES, _ACL_MEMCPY_DEVICE_TO_HOST), "d2h")
    lib.aclrtFree(sd); lib.aclrtFree(td); lib.aclrtFree(gd); lib.aclrtFree(od)
    lib.aclrtDestroyStream(st); lib.aclrtResetDevice(0); lib.aclFinalize()
    if not np.allclose(oh, ref, atol=1e-4, rtol=1e-4, equal_nan=True):
        diff = np.abs(oh.astype(np.float64) - ref.astype(np.float64))
        idx = int(np.nanargmax(diff))
        raise AssertionError(f"mismatch max_diff={diff[idx]} at i={idx}")
    print(
        f"PASS {tag}  compile={cs:.3f}s launch={ls:.3f}s "
        f"maxabs={float(np.max(np.abs(oh - ref)))} gate_on={gate_on}"
    )
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("E", nargs="?", type=int, default=E)
    ap.add_argument("B", nargs="?", type=int, default=B)
    ap.add_argument("G", nargs="?", type=int, default=0)
    ap.add_argument("pgate", nargs="?", type=int, default=50)
    ap.add_argument("arm", nargs="?", type=str, default="B")
    ap.add_argument("--emit-mlir", action="store_true")
    args = ap.parse_args(argv)
    arm = args.arm.upper()
    tag = f"cf7d_rib_e{E}_b{B}_g{args.G}_vl{VL}_pgate{args.pgate}_arm{arm}"
    if args.emit_mlir:
        k = build_kernel(arm, args.G)
        print(k.compile().mlir_text())
        return 0
    return run_acl(tag, arm, args.G, args.pgate)


if __name__ == "__main__":
    raise SystemExit(main())
