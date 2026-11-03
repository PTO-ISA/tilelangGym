#!/usr/bin/env python3
"""ST-CF7d — PTO-DSL global outer gate × inner kill (arms B/C).

Design: reports/ST_CF7_OUTER_INNER_IF_DESIGN.md (+ GLOBAL_GATE rev 2026-11-03)
Module-level @pto.jit (no nested factory).
Tag: cf7d_e{E}_b{B}_g{G}_vl{VL}_pgate{P}_arm{B|C}

Rev (Lok 2026-11-03):
  - Drop arm A (pure_slack).
  - Outer = ONE global gate: gate_sum = sum(gate) > GATE_SUM_THR (not per-b).
  - Nest: for b in B (fixed large / stable overhead); for _ in G (sensitivity);
    kill inside G so B needs vand/PAND with bcast(gate_on).
  - Arm B: vcadd → vbrc → vcmp → vand with kill; always enter G; publish via vsel(p_outer).
  - Arm C: same reduce → scalar cond; pto.if_(gate_on) wraps G nest (real VF skip;
    NO host n_trips specialization).
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
    name="CF7d_armB_g0",
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
    mask_B = pto.vmi.create_mask(B, size=VL)
    mask1 = pto.vmi.create_mask(1, size=1)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    # global gate: reduce_sum(gate) > GATE_SUM_THR
    gate_v = pto.vmi.vload(gate_ub, 0, size=VL)
    gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
    pto.vmi.vstore(gs, scratch_ub, 0, mask1)
    thr_sum_v = pto.vmi.vbrc(pto.f32(GATE_SUM_THR), size=VL)
    gs_s = pto.vmi.vload(scratch_ub, 0, size=1)
    p_outer = pto.vmi.vcmp(pto.vmi.vbrc(gs_s, size=VL), thr_sum_v, mask, "gt")

    for b in range(8):  # B fixed — stable overhead
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        pass  # G=0 — gate reduce + B loop overhead only (no kill)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g0",
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
    mask_B = pto.vmi.create_mask(B, size=VL)
    mask1 = pto.vmi.create_mask(1, size=1)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    # global gate: reduce_sum(gate) > GATE_SUM_THR
    gate_v = pto.vmi.vload(gate_ub, 0, size=VL)
    gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
    pto.vmi.vstore(gs, scratch_ub, 0, mask1)
    gs_sc = pto.load_scalar(scratch_ub, 0)
    gate_on = gs_sc > pto.f32(GATE_SUM_THR)

    for b in range(8):  # B fixed — stable overhead
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        with pto.if_(gate_on) as br:  # real VF scalar if — NO host n_trips
            with br.then_:
                br.assign(s0=s0, s1=s1, s2=s2, s3=s3)  # G=0
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
    name="CF7d_armB_g8",
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
    mask_B = pto.vmi.create_mask(B, size=VL)
    mask1 = pto.vmi.create_mask(1, size=1)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    # global gate: reduce_sum(gate) > GATE_SUM_THR
    gate_v = pto.vmi.vload(gate_ub, 0, size=VL)
    gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
    pto.vmi.vstore(gs, scratch_ub, 0, mask1)
    thr_sum_v = pto.vmi.vbrc(pto.f32(GATE_SUM_THR), size=VL)
    gs_s = pto.vmi.vload(scratch_ub, 0, size=1)
    p_outer = pto.vmi.vcmp(pto.vmi.vbrc(gs_s, size=VL), thr_sum_v, mask, "gt")

    for b in range(8):  # B fixed — stable overhead
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(8):  # G fat + kill each step; PAND with p_outer
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
    name="CF7d_armC_g8",
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
    mask_B = pto.vmi.create_mask(B, size=VL)  # VL-resident; size=B hits ensure_layout (sv5)
    mask1 = pto.vmi.create_mask(1, size=1)

    # global gate: reduce_sum(gate) > GATE_SUM_THR (hoisted once)
    # Use vload size=1 + load_scalar after V visibility (load_scalar alone races V store)
    gate_v = pto.vmi.vload(gate_ub, 0, size=VL)
    gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
    pto.vmi.vstore(gs, scratch_ub, 0, mask1)
    pto.set_flag("V", "S", event_id=2)
    pto.wait_flag("V", "S", event_id=2)
    gs_sc = pto.load_scalar(scratch_ub, 0)
    gate_on = gs_sc > pto.f32(GATE_SUM_THR)

    # Identity seed via MTE (no vreg SSA that mem2reg can forward into scf.if)
    pto.mte_load(scores_gm, out_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.set_flag("MTE2", "V", event_id=1)
    pto.wait_flag("MTE2", "V", event_id=1)

    for b in range(8):  # B fixed — stable overhead; accumulate across b
        with pto.if_(gate_on) as br:  # side-effect-only: real VF skip, no vreg phi
            with br.then_:
                neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
                scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)
                thr_s = pto.vmi.vload(thr_ub, b, size=1)
                thr_b = pto.vmi.vbrc(thr_s, size=VL)
                t0 = pto.vmi.vload(out_ub, 0 * VL, size=VL)
                t1 = pto.vmi.vload(out_ub, 1 * VL, size=VL)
                t2 = pto.vmi.vload(out_ub, 2 * VL, size=VL)
                t3 = pto.vmi.vload(out_ub, 3 * VL, size=VL)
                for _t in range(8):
                    t0 = pto.vmi.vmul(t0, scale_v, mask)
                    t1 = pto.vmi.vmul(t1, scale_v, mask)
                    t2 = pto.vmi.vmul(t2, scale_v, mask)
                    t3 = pto.vmi.vmul(t3, scale_v, mask)
                    t0 = pto.vmi.vsel(pto.vmi.vcmp(t0, thr_b, mask, "gt"), neg_inf, t0)
                    t1 = pto.vmi.vsel(pto.vmi.vcmp(t1, thr_b, mask, "gt"), neg_inf, t1)
                    t2 = pto.vmi.vsel(pto.vmi.vcmp(t2, thr_b, mask, "gt"), neg_inf, t2)
                    t3 = pto.vmi.vsel(pto.vmi.vcmp(t3, thr_b, mask, "gt"), neg_inf, t3)
                pto.vmi.vstore(t0, out_ub, 0 * VL, mask)
                pto.vmi.vstore(t1, out_ub, 1 * VL, mask)
                pto.vmi.vstore(t2, out_ub, 2 * VL, mask)
                pto.vmi.vstore(t3, out_ub, 3 * VL, mask)

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))


@pto.jit(
    name="CF7d_armB_g16",
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
    mask_B = pto.vmi.create_mask(B, size=VL)
    mask1 = pto.vmi.create_mask(1, size=1)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    # global gate: reduce_sum(gate) > GATE_SUM_THR
    gate_v = pto.vmi.vload(gate_ub, 0, size=VL)
    gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
    pto.vmi.vstore(gs, scratch_ub, 0, mask1)
    thr_sum_v = pto.vmi.vbrc(pto.f32(GATE_SUM_THR), size=VL)
    gs_s = pto.vmi.vload(scratch_ub, 0, size=1)
    p_outer = pto.vmi.vcmp(pto.vmi.vbrc(gs_s, size=VL), thr_sum_v, mask, "gt")

    for b in range(8):  # B fixed — stable overhead
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(16):  # G fat + kill each step; PAND with p_outer
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
    name="CF7d_armC_g16",
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
    mask_B = pto.vmi.create_mask(B, size=VL)  # VL-resident; size=B hits ensure_layout (sv5)
    mask1 = pto.vmi.create_mask(1, size=1)

    # global gate: reduce_sum(gate) > GATE_SUM_THR (hoisted once)
    # Use vload size=1 + load_scalar after V visibility (load_scalar alone races V store)
    gate_v = pto.vmi.vload(gate_ub, 0, size=VL)
    gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
    pto.vmi.vstore(gs, scratch_ub, 0, mask1)
    pto.set_flag("V", "S", event_id=2)
    pto.wait_flag("V", "S", event_id=2)
    gs_sc = pto.load_scalar(scratch_ub, 0)
    gate_on = gs_sc > pto.f32(GATE_SUM_THR)

    # Identity seed via MTE (no vreg SSA that mem2reg can forward into scf.if)
    pto.mte_load(scores_gm, out_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.set_flag("MTE2", "V", event_id=1)
    pto.wait_flag("MTE2", "V", event_id=1)

    for b in range(8):  # B fixed — stable overhead; accumulate across b
        with pto.if_(gate_on) as br:  # side-effect-only: real VF skip, no vreg phi
            with br.then_:
                neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
                scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)
                thr_s = pto.vmi.vload(thr_ub, b, size=1)
                thr_b = pto.vmi.vbrc(thr_s, size=VL)
                t0 = pto.vmi.vload(out_ub, 0 * VL, size=VL)
                t1 = pto.vmi.vload(out_ub, 1 * VL, size=VL)
                t2 = pto.vmi.vload(out_ub, 2 * VL, size=VL)
                t3 = pto.vmi.vload(out_ub, 3 * VL, size=VL)
                for _t in range(16):
                    t0 = pto.vmi.vmul(t0, scale_v, mask)
                    t1 = pto.vmi.vmul(t1, scale_v, mask)
                    t2 = pto.vmi.vmul(t2, scale_v, mask)
                    t3 = pto.vmi.vmul(t3, scale_v, mask)
                    t0 = pto.vmi.vsel(pto.vmi.vcmp(t0, thr_b, mask, "gt"), neg_inf, t0)
                    t1 = pto.vmi.vsel(pto.vmi.vcmp(t1, thr_b, mask, "gt"), neg_inf, t1)
                    t2 = pto.vmi.vsel(pto.vmi.vcmp(t2, thr_b, mask, "gt"), neg_inf, t2)
                    t3 = pto.vmi.vsel(pto.vmi.vcmp(t3, thr_b, mask, "gt"), neg_inf, t3)
                pto.vmi.vstore(t0, out_ub, 0 * VL, mask)
                pto.vmi.vstore(t1, out_ub, 1 * VL, mask)
                pto.vmi.vstore(t2, out_ub, 2 * VL, mask)
                pto.vmi.vstore(t3, out_ub, 3 * VL, mask)

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))


@pto.jit(
    name="CF7d_armB_g32",
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
    mask_B = pto.vmi.create_mask(B, size=VL)
    mask1 = pto.vmi.create_mask(1, size=1)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    # global gate: reduce_sum(gate) > GATE_SUM_THR
    gate_v = pto.vmi.vload(gate_ub, 0, size=VL)
    gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
    pto.vmi.vstore(gs, scratch_ub, 0, mask1)
    thr_sum_v = pto.vmi.vbrc(pto.f32(GATE_SUM_THR), size=VL)
    gs_s = pto.vmi.vload(scratch_ub, 0, size=1)
    p_outer = pto.vmi.vcmp(pto.vmi.vbrc(gs_s, size=VL), thr_sum_v, mask, "gt")

    for b in range(8):  # B fixed — stable overhead
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(32):  # G fat + kill each step; PAND with p_outer
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
    name="CF7d_armC_g32",
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
    mask_B = pto.vmi.create_mask(B, size=VL)  # VL-resident; size=B hits ensure_layout (sv5)
    mask1 = pto.vmi.create_mask(1, size=1)

    # global gate: reduce_sum(gate) > GATE_SUM_THR (hoisted once)
    # Use vload size=1 + load_scalar after V visibility (load_scalar alone races V store)
    gate_v = pto.vmi.vload(gate_ub, 0, size=VL)
    gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
    pto.vmi.vstore(gs, scratch_ub, 0, mask1)
    pto.set_flag("V", "S", event_id=2)
    pto.wait_flag("V", "S", event_id=2)
    gs_sc = pto.load_scalar(scratch_ub, 0)
    gate_on = gs_sc > pto.f32(GATE_SUM_THR)

    # Identity seed via MTE (no vreg SSA that mem2reg can forward into scf.if)
    pto.mte_load(scores_gm, out_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.set_flag("MTE2", "V", event_id=1)
    pto.wait_flag("MTE2", "V", event_id=1)

    for b in range(8):  # B fixed — stable overhead; accumulate across b
        with pto.if_(gate_on) as br:  # side-effect-only: real VF skip, no vreg phi
            with br.then_:
                neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
                scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)
                thr_s = pto.vmi.vload(thr_ub, b, size=1)
                thr_b = pto.vmi.vbrc(thr_s, size=VL)
                t0 = pto.vmi.vload(out_ub, 0 * VL, size=VL)
                t1 = pto.vmi.vload(out_ub, 1 * VL, size=VL)
                t2 = pto.vmi.vload(out_ub, 2 * VL, size=VL)
                t3 = pto.vmi.vload(out_ub, 3 * VL, size=VL)
                for _t in range(32):
                    t0 = pto.vmi.vmul(t0, scale_v, mask)
                    t1 = pto.vmi.vmul(t1, scale_v, mask)
                    t2 = pto.vmi.vmul(t2, scale_v, mask)
                    t3 = pto.vmi.vmul(t3, scale_v, mask)
                    t0 = pto.vmi.vsel(pto.vmi.vcmp(t0, thr_b, mask, "gt"), neg_inf, t0)
                    t1 = pto.vmi.vsel(pto.vmi.vcmp(t1, thr_b, mask, "gt"), neg_inf, t1)
                    t2 = pto.vmi.vsel(pto.vmi.vcmp(t2, thr_b, mask, "gt"), neg_inf, t2)
                    t3 = pto.vmi.vsel(pto.vmi.vcmp(t3, thr_b, mask, "gt"), neg_inf, t3)
                pto.vmi.vstore(t0, out_ub, 0 * VL, mask)
                pto.vmi.vstore(t1, out_ub, 1 * VL, mask)
                pto.vmi.vstore(t2, out_ub, 2 * VL, mask)
                pto.vmi.vstore(t3, out_ub, 3 * VL, mask)

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))


@pto.jit(
    name="CF7d_armB_g64",
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
    mask_B = pto.vmi.create_mask(B, size=VL)
    mask1 = pto.vmi.create_mask(1, size=1)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    # global gate: reduce_sum(gate) > GATE_SUM_THR
    gate_v = pto.vmi.vload(gate_ub, 0, size=VL)
    gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
    pto.vmi.vstore(gs, scratch_ub, 0, mask1)
    thr_sum_v = pto.vmi.vbrc(pto.f32(GATE_SUM_THR), size=VL)
    gs_s = pto.vmi.vload(scratch_ub, 0, size=1)
    p_outer = pto.vmi.vcmp(pto.vmi.vbrc(gs_s, size=VL), thr_sum_v, mask, "gt")

    for b in range(8):  # B fixed — stable overhead
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(64):  # G fat + kill each step; PAND with p_outer
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
    name="CF7d_armC_g64",
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
    mask_B = pto.vmi.create_mask(B, size=VL)  # VL-resident; size=B hits ensure_layout (sv5)
    mask1 = pto.vmi.create_mask(1, size=1)

    # global gate: reduce_sum(gate) > GATE_SUM_THR (hoisted once)
    # Use vload size=1 + load_scalar after V visibility (load_scalar alone races V store)
    gate_v = pto.vmi.vload(gate_ub, 0, size=VL)
    gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
    pto.vmi.vstore(gs, scratch_ub, 0, mask1)
    pto.set_flag("V", "S", event_id=2)
    pto.wait_flag("V", "S", event_id=2)
    gs_sc = pto.load_scalar(scratch_ub, 0)
    gate_on = gs_sc > pto.f32(GATE_SUM_THR)

    # Identity seed via MTE (no vreg SSA that mem2reg can forward into scf.if)
    pto.mte_load(scores_gm, out_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.set_flag("MTE2", "V", event_id=1)
    pto.wait_flag("MTE2", "V", event_id=1)

    for b in range(8):  # B fixed — stable overhead; accumulate across b
        with pto.if_(gate_on) as br:  # side-effect-only: real VF skip, no vreg phi
            with br.then_:
                neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
                scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)
                thr_s = pto.vmi.vload(thr_ub, b, size=1)
                thr_b = pto.vmi.vbrc(thr_s, size=VL)
                t0 = pto.vmi.vload(out_ub, 0 * VL, size=VL)
                t1 = pto.vmi.vload(out_ub, 1 * VL, size=VL)
                t2 = pto.vmi.vload(out_ub, 2 * VL, size=VL)
                t3 = pto.vmi.vload(out_ub, 3 * VL, size=VL)
                for _t in range(64):
                    t0 = pto.vmi.vmul(t0, scale_v, mask)
                    t1 = pto.vmi.vmul(t1, scale_v, mask)
                    t2 = pto.vmi.vmul(t2, scale_v, mask)
                    t3 = pto.vmi.vmul(t3, scale_v, mask)
                    t0 = pto.vmi.vsel(pto.vmi.vcmp(t0, thr_b, mask, "gt"), neg_inf, t0)
                    t1 = pto.vmi.vsel(pto.vmi.vcmp(t1, thr_b, mask, "gt"), neg_inf, t1)
                    t2 = pto.vmi.vsel(pto.vmi.vcmp(t2, thr_b, mask, "gt"), neg_inf, t2)
                    t3 = pto.vmi.vsel(pto.vmi.vcmp(t3, thr_b, mask, "gt"), neg_inf, t3)
                pto.vmi.vstore(t0, out_ub, 0 * VL, mask)
                pto.vmi.vstore(t1, out_ub, 1 * VL, mask)
                pto.vmi.vstore(t2, out_ub, 2 * VL, mask)
                pto.vmi.vstore(t3, out_ub, 3 * VL, mask)

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
    ("C", 64): kernel_C_g64,
}


def build_kernel(arm, G):
    arm = arm.upper()
    if arm == "A":
        raise KeyError("arm A (pure_slack) dropped in global-gate rev; use B or C")
    key = (arm, int(G))
    if key not in KERNELS:
        raise KeyError(f"no module kernel for {key}; G in {sorted({k[1] for k in KERNELS})}")
    return KERNELS[key]


def reference(scores, thr, gate, G, gate_sum_thr=GATE_SUM_THR):
    """Gold: global gate_on; B outer; G inner; kill inside G only when gate_on."""
    out = scores.astype(np.float32).copy()
    thr = thr.astype(np.float32)
    gate = gate.astype(np.float32)
    scale = np.float32(SCALE)
    gate_on = float(np.sum(gate)) > float(gate_sum_thr)
    for b in range(len(thr)):
        if gate_on:
            for _ in range(G):
                out = (out * scale).astype(np.float32)
                out = np.where(out > thr[b], np.float32(NEG_INF_F32), out).astype(np.float32)
    return out


def make_inputs(pgate, seed=27):
    """Build gate[] so sum(gate) is above/below GATE_SUM_THR from pgate label."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(-3, 3, E).astype(np.float32)
    thr = rng.uniform(-1, 1, B).astype(np.float32)
    on = gate_on_for(pgate)
    if on:
        # sum > 0: all positive
        gate = rng.uniform(0.1, 2.0, size=B).astype(np.float32)
    else:
        # sum < 0: all negative
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
        n_neg_oh = int(np.sum(oh <= np.float32(NEG_INF_F32 / 2)))
        n_neg_ref = int(np.sum(ref <= np.float32(NEG_INF_F32 / 2)))
        print(
            f"MISMATCH tag={tag} max_diff={diff[idx]} at i={idx} "
            f"oh={oh[idx]} ref={ref[idx]} n_neg_oh={n_neg_oh} n_neg_ref={n_neg_ref} "
            f"oh[:4]={oh[:4].tolist()} gate={gate.tolist()} gate_on={gate_on}",
            flush=True,
        )
        raise AssertionError(f"mismatch max_diff={diff[idx]}")
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
    tag = f"cf7d_e{E}_b{B}_g{args.G}_vl{VL}_pgate{args.pgate}_arm{arm}"
    if args.emit_mlir:
        k = build_kernel(arm, args.G)
        print(k.compile().mlir_text())
        return 0
    return run_acl(tag, arm, args.G, args.pgate)


if __name__ == "__main__":
    raise SystemExit(main())
