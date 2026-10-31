#!/usr/bin/env python3
"""ST-CF7d — PTO-DSL outer-gate × inner kill crossover (arms A/B/C).

Design: reports/ST_CF7_OUTER_INNER_IF_DESIGN.md
Module-level @pto.jit (no nested factory).
Tag: cf7d_e{E}_b{B}_g{G}_vl{VL}_pgate{P}_arm{A|B|C}
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
GATE_THR = 0.0
SCALE = 1.000001
NEG_INF_F32 = -3.40282347e38

SCORES_BYTES = E * 4
THR_BYTES = B * 4
GATE_BYTES = B * 4
OUT_BYTES = E * 4
OFF_THR = SCORES_BYTES
OFF_GATE = OFF_THR + ((THR_BYTES + 31) // 32) * 32
OFF_OUT = OFF_GATE + ((GATE_BYTES + 31) // 32) * 32

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


def n_true_for(pgate, b=B):
    return max(0, int(round(b * pgate / 100.0)))


@pto.jit(
    name="CF7d_armA_g0_n8",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_A_g0_n8(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)
    gate_thr_v = pto.vmi.vbrc(pto.f32(GATE_THR), size=VL)

    for b in range(8):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        gate_s = pto.vmi.vload(gate_ub, b, size=1)
        gate_b = pto.vmi.vbrc(gate_s, size=VL)
        p_outer = pto.vmi.vcmp(gate_b, gate_thr_v, mask, "gt")
        k0 = s0
        k1 = s1
        k2 = s2
        k3 = s3
        k0 = pto.vmi.vsel(pto.vmi.vcmp(k0, thr_b, mask, "gt"), neg_inf, k0)
        k1 = pto.vmi.vsel(pto.vmi.vcmp(k1, thr_b, mask, "gt"), neg_inf, k1)
        k2 = pto.vmi.vsel(pto.vmi.vcmp(k2, thr_b, mask, "gt"), neg_inf, k2)
        k3 = pto.vmi.vsel(pto.vmi.vcmp(k3, thr_b, mask, "gt"), neg_inf, k3)
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
    name="CF7d_armB_g0_n8",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_B_g0_n8(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)
    gate_thr_v = pto.vmi.vbrc(pto.f32(GATE_THR), size=VL)

    for b in range(8):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        gate_s = pto.vmi.vload(gate_ub, b, size=1)
        gate_b = pto.vmi.vbrc(gate_s, size=VL)
        p_outer = pto.vmi.vcmp(gate_b, gate_thr_v, mask, "gt")
        k0 = s0
        k1 = s1
        k2 = s2
        k3 = s3
        kill0 = pto.vmi.vsel(pto.vmi.vcmp(k0, thr_b, mask, "gt"), neg_inf, k0)
        kill1 = pto.vmi.vsel(pto.vmi.vcmp(k1, thr_b, mask, "gt"), neg_inf, k1)
        kill2 = pto.vmi.vsel(pto.vmi.vcmp(k2, thr_b, mask, "gt"), neg_inf, k2)
        kill3 = pto.vmi.vsel(pto.vmi.vcmp(k3, thr_b, mask, "gt"), neg_inf, k3)
        s0 = pto.vmi.vsel(p_outer, kill0, s0)
        s1 = pto.vmi.vsel(p_outer, kill1, s1)
        s2 = pto.vmi.vsel(p_outer, kill2, s2)
        s3 = pto.vmi.vsel(p_outer, kill3, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g0_n0",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g0_n0(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(0):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g0_n2",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g0_n2(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(2):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g0_n4",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g0_n4(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(4):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g0_n6",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g0_n6(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(6):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g0_n8",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g0_n8(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(8):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armA_g8_n8",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_A_g8_n8(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)
    gate_thr_v = pto.vmi.vbrc(pto.f32(GATE_THR), size=VL)

    for b in range(8):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        gate_s = pto.vmi.vload(gate_ub, b, size=1)
        gate_b = pto.vmi.vbrc(gate_s, size=VL)
        p_outer = pto.vmi.vcmp(gate_b, gate_thr_v, mask, "gt")
        k0 = s0
        k1 = s1
        k2 = s2
        k3 = s3
        for _t in range(8):
            k0 = pto.vmi.vmul(k0, scale_v, mask)
            k1 = pto.vmi.vmul(k1, scale_v, mask)
            k2 = pto.vmi.vmul(k2, scale_v, mask)
            k3 = pto.vmi.vmul(k3, scale_v, mask)
        k0 = pto.vmi.vsel(pto.vmi.vcmp(k0, thr_b, mask, "gt"), neg_inf, k0)
        k1 = pto.vmi.vsel(pto.vmi.vcmp(k1, thr_b, mask, "gt"), neg_inf, k1)
        k2 = pto.vmi.vsel(pto.vmi.vcmp(k2, thr_b, mask, "gt"), neg_inf, k2)
        k3 = pto.vmi.vsel(pto.vmi.vcmp(k3, thr_b, mask, "gt"), neg_inf, k3)
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
    name="CF7d_armB_g8_n8",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_B_g8_n8(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)
    gate_thr_v = pto.vmi.vbrc(pto.f32(GATE_THR), size=VL)

    for b in range(8):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        gate_s = pto.vmi.vload(gate_ub, b, size=1)
        gate_b = pto.vmi.vbrc(gate_s, size=VL)
        p_outer = pto.vmi.vcmp(gate_b, gate_thr_v, mask, "gt")
        k0 = s0
        k1 = s1
        k2 = s2
        k3 = s3
        for _t in range(8):
            k0 = pto.vmi.vmul(k0, scale_v, mask)
            k1 = pto.vmi.vmul(k1, scale_v, mask)
            k2 = pto.vmi.vmul(k2, scale_v, mask)
            k3 = pto.vmi.vmul(k3, scale_v, mask)
        kill0 = pto.vmi.vsel(pto.vmi.vcmp(k0, thr_b, mask, "gt"), neg_inf, k0)
        kill1 = pto.vmi.vsel(pto.vmi.vcmp(k1, thr_b, mask, "gt"), neg_inf, k1)
        kill2 = pto.vmi.vsel(pto.vmi.vcmp(k2, thr_b, mask, "gt"), neg_inf, k2)
        kill3 = pto.vmi.vsel(pto.vmi.vcmp(k3, thr_b, mask, "gt"), neg_inf, k3)
        s0 = pto.vmi.vsel(p_outer, kill0, s0)
        s1 = pto.vmi.vsel(p_outer, kill1, s1)
        s2 = pto.vmi.vsel(p_outer, kill2, s2)
        s3 = pto.vmi.vsel(p_outer, kill3, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g8_n0",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g8_n0(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(0):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(8):
            s0 = pto.vmi.vmul(s0, scale_v, mask)
            s1 = pto.vmi.vmul(s1, scale_v, mask)
            s2 = pto.vmi.vmul(s2, scale_v, mask)
            s3 = pto.vmi.vmul(s3, scale_v, mask)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g8_n2",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g8_n2(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(2):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(8):
            s0 = pto.vmi.vmul(s0, scale_v, mask)
            s1 = pto.vmi.vmul(s1, scale_v, mask)
            s2 = pto.vmi.vmul(s2, scale_v, mask)
            s3 = pto.vmi.vmul(s3, scale_v, mask)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g8_n4",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g8_n4(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(4):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(8):
            s0 = pto.vmi.vmul(s0, scale_v, mask)
            s1 = pto.vmi.vmul(s1, scale_v, mask)
            s2 = pto.vmi.vmul(s2, scale_v, mask)
            s3 = pto.vmi.vmul(s3, scale_v, mask)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g8_n6",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g8_n6(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(6):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(8):
            s0 = pto.vmi.vmul(s0, scale_v, mask)
            s1 = pto.vmi.vmul(s1, scale_v, mask)
            s2 = pto.vmi.vmul(s2, scale_v, mask)
            s3 = pto.vmi.vmul(s3, scale_v, mask)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g8_n8",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g8_n8(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(8):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(8):
            s0 = pto.vmi.vmul(s0, scale_v, mask)
            s1 = pto.vmi.vmul(s1, scale_v, mask)
            s2 = pto.vmi.vmul(s2, scale_v, mask)
            s3 = pto.vmi.vmul(s3, scale_v, mask)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armA_g16_n8",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_A_g16_n8(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)
    gate_thr_v = pto.vmi.vbrc(pto.f32(GATE_THR), size=VL)

    for b in range(8):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        gate_s = pto.vmi.vload(gate_ub, b, size=1)
        gate_b = pto.vmi.vbrc(gate_s, size=VL)
        p_outer = pto.vmi.vcmp(gate_b, gate_thr_v, mask, "gt")
        k0 = s0
        k1 = s1
        k2 = s2
        k3 = s3
        for _t in range(16):
            k0 = pto.vmi.vmul(k0, scale_v, mask)
            k1 = pto.vmi.vmul(k1, scale_v, mask)
            k2 = pto.vmi.vmul(k2, scale_v, mask)
            k3 = pto.vmi.vmul(k3, scale_v, mask)
        k0 = pto.vmi.vsel(pto.vmi.vcmp(k0, thr_b, mask, "gt"), neg_inf, k0)
        k1 = pto.vmi.vsel(pto.vmi.vcmp(k1, thr_b, mask, "gt"), neg_inf, k1)
        k2 = pto.vmi.vsel(pto.vmi.vcmp(k2, thr_b, mask, "gt"), neg_inf, k2)
        k3 = pto.vmi.vsel(pto.vmi.vcmp(k3, thr_b, mask, "gt"), neg_inf, k3)
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
    name="CF7d_armB_g16_n8",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_B_g16_n8(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)
    gate_thr_v = pto.vmi.vbrc(pto.f32(GATE_THR), size=VL)

    for b in range(8):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        gate_s = pto.vmi.vload(gate_ub, b, size=1)
        gate_b = pto.vmi.vbrc(gate_s, size=VL)
        p_outer = pto.vmi.vcmp(gate_b, gate_thr_v, mask, "gt")
        k0 = s0
        k1 = s1
        k2 = s2
        k3 = s3
        for _t in range(16):
            k0 = pto.vmi.vmul(k0, scale_v, mask)
            k1 = pto.vmi.vmul(k1, scale_v, mask)
            k2 = pto.vmi.vmul(k2, scale_v, mask)
            k3 = pto.vmi.vmul(k3, scale_v, mask)
        kill0 = pto.vmi.vsel(pto.vmi.vcmp(k0, thr_b, mask, "gt"), neg_inf, k0)
        kill1 = pto.vmi.vsel(pto.vmi.vcmp(k1, thr_b, mask, "gt"), neg_inf, k1)
        kill2 = pto.vmi.vsel(pto.vmi.vcmp(k2, thr_b, mask, "gt"), neg_inf, k2)
        kill3 = pto.vmi.vsel(pto.vmi.vcmp(k3, thr_b, mask, "gt"), neg_inf, k3)
        s0 = pto.vmi.vsel(p_outer, kill0, s0)
        s1 = pto.vmi.vsel(p_outer, kill1, s1)
        s2 = pto.vmi.vsel(p_outer, kill2, s2)
        s3 = pto.vmi.vsel(p_outer, kill3, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g16_n0",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g16_n0(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(0):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(16):
            s0 = pto.vmi.vmul(s0, scale_v, mask)
            s1 = pto.vmi.vmul(s1, scale_v, mask)
            s2 = pto.vmi.vmul(s2, scale_v, mask)
            s3 = pto.vmi.vmul(s3, scale_v, mask)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g16_n2",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g16_n2(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(2):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(16):
            s0 = pto.vmi.vmul(s0, scale_v, mask)
            s1 = pto.vmi.vmul(s1, scale_v, mask)
            s2 = pto.vmi.vmul(s2, scale_v, mask)
            s3 = pto.vmi.vmul(s3, scale_v, mask)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g16_n4",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g16_n4(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(4):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(16):
            s0 = pto.vmi.vmul(s0, scale_v, mask)
            s1 = pto.vmi.vmul(s1, scale_v, mask)
            s2 = pto.vmi.vmul(s2, scale_v, mask)
            s3 = pto.vmi.vmul(s3, scale_v, mask)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g16_n6",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g16_n6(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(6):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(16):
            s0 = pto.vmi.vmul(s0, scale_v, mask)
            s1 = pto.vmi.vmul(s1, scale_v, mask)
            s2 = pto.vmi.vmul(s2, scale_v, mask)
            s3 = pto.vmi.vmul(s3, scale_v, mask)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g16_n8",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g16_n8(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(8):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(16):
            s0 = pto.vmi.vmul(s0, scale_v, mask)
            s1 = pto.vmi.vmul(s1, scale_v, mask)
            s2 = pto.vmi.vmul(s2, scale_v, mask)
            s3 = pto.vmi.vmul(s3, scale_v, mask)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armA_g32_n8",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_A_g32_n8(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)
    gate_thr_v = pto.vmi.vbrc(pto.f32(GATE_THR), size=VL)

    for b in range(8):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        gate_s = pto.vmi.vload(gate_ub, b, size=1)
        gate_b = pto.vmi.vbrc(gate_s, size=VL)
        p_outer = pto.vmi.vcmp(gate_b, gate_thr_v, mask, "gt")
        k0 = s0
        k1 = s1
        k2 = s2
        k3 = s3
        for _t in range(32):
            k0 = pto.vmi.vmul(k0, scale_v, mask)
            k1 = pto.vmi.vmul(k1, scale_v, mask)
            k2 = pto.vmi.vmul(k2, scale_v, mask)
            k3 = pto.vmi.vmul(k3, scale_v, mask)
        k0 = pto.vmi.vsel(pto.vmi.vcmp(k0, thr_b, mask, "gt"), neg_inf, k0)
        k1 = pto.vmi.vsel(pto.vmi.vcmp(k1, thr_b, mask, "gt"), neg_inf, k1)
        k2 = pto.vmi.vsel(pto.vmi.vcmp(k2, thr_b, mask, "gt"), neg_inf, k2)
        k3 = pto.vmi.vsel(pto.vmi.vcmp(k3, thr_b, mask, "gt"), neg_inf, k3)
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
    name="CF7d_armB_g32_n8",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_B_g32_n8(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)
    gate_thr_v = pto.vmi.vbrc(pto.f32(GATE_THR), size=VL)

    for b in range(8):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        gate_s = pto.vmi.vload(gate_ub, b, size=1)
        gate_b = pto.vmi.vbrc(gate_s, size=VL)
        p_outer = pto.vmi.vcmp(gate_b, gate_thr_v, mask, "gt")
        k0 = s0
        k1 = s1
        k2 = s2
        k3 = s3
        for _t in range(32):
            k0 = pto.vmi.vmul(k0, scale_v, mask)
            k1 = pto.vmi.vmul(k1, scale_v, mask)
            k2 = pto.vmi.vmul(k2, scale_v, mask)
            k3 = pto.vmi.vmul(k3, scale_v, mask)
        kill0 = pto.vmi.vsel(pto.vmi.vcmp(k0, thr_b, mask, "gt"), neg_inf, k0)
        kill1 = pto.vmi.vsel(pto.vmi.vcmp(k1, thr_b, mask, "gt"), neg_inf, k1)
        kill2 = pto.vmi.vsel(pto.vmi.vcmp(k2, thr_b, mask, "gt"), neg_inf, k2)
        kill3 = pto.vmi.vsel(pto.vmi.vcmp(k3, thr_b, mask, "gt"), neg_inf, k3)
        s0 = pto.vmi.vsel(p_outer, kill0, s0)
        s1 = pto.vmi.vsel(p_outer, kill1, s1)
        s2 = pto.vmi.vsel(p_outer, kill2, s2)
        s3 = pto.vmi.vsel(p_outer, kill3, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g32_n0",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g32_n0(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(0):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(32):
            s0 = pto.vmi.vmul(s0, scale_v, mask)
            s1 = pto.vmi.vmul(s1, scale_v, mask)
            s2 = pto.vmi.vmul(s2, scale_v, mask)
            s3 = pto.vmi.vmul(s3, scale_v, mask)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g32_n2",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g32_n2(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(2):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(32):
            s0 = pto.vmi.vmul(s0, scale_v, mask)
            s1 = pto.vmi.vmul(s1, scale_v, mask)
            s2 = pto.vmi.vmul(s2, scale_v, mask)
            s3 = pto.vmi.vmul(s3, scale_v, mask)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g32_n4",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g32_n4(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(4):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(32):
            s0 = pto.vmi.vmul(s0, scale_v, mask)
            s1 = pto.vmi.vmul(s1, scale_v, mask)
            s2 = pto.vmi.vmul(s2, scale_v, mask)
            s3 = pto.vmi.vmul(s3, scale_v, mask)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g32_n6",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g32_n6(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(6):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(32):
            s0 = pto.vmi.vmul(s0, scale_v, mask)
            s1 = pto.vmi.vmul(s1, scale_v, mask)
            s2 = pto.vmi.vmul(s2, scale_v, mask)
            s3 = pto.vmi.vmul(s3, scale_v, mask)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g32_n8",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g32_n8(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(8):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(32):
            s0 = pto.vmi.vmul(s0, scale_v, mask)
            s1 = pto.vmi.vmul(s1, scale_v, mask)
            s2 = pto.vmi.vmul(s2, scale_v, mask)
            s3 = pto.vmi.vmul(s3, scale_v, mask)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armA_g64_n8",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_A_g64_n8(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)
    gate_thr_v = pto.vmi.vbrc(pto.f32(GATE_THR), size=VL)

    for b in range(8):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        gate_s = pto.vmi.vload(gate_ub, b, size=1)
        gate_b = pto.vmi.vbrc(gate_s, size=VL)
        p_outer = pto.vmi.vcmp(gate_b, gate_thr_v, mask, "gt")
        k0 = s0
        k1 = s1
        k2 = s2
        k3 = s3
        for _t in range(64):
            k0 = pto.vmi.vmul(k0, scale_v, mask)
            k1 = pto.vmi.vmul(k1, scale_v, mask)
            k2 = pto.vmi.vmul(k2, scale_v, mask)
            k3 = pto.vmi.vmul(k3, scale_v, mask)
        k0 = pto.vmi.vsel(pto.vmi.vcmp(k0, thr_b, mask, "gt"), neg_inf, k0)
        k1 = pto.vmi.vsel(pto.vmi.vcmp(k1, thr_b, mask, "gt"), neg_inf, k1)
        k2 = pto.vmi.vsel(pto.vmi.vcmp(k2, thr_b, mask, "gt"), neg_inf, k2)
        k3 = pto.vmi.vsel(pto.vmi.vcmp(k3, thr_b, mask, "gt"), neg_inf, k3)
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
    name="CF7d_armB_g64_n8",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_B_g64_n8(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)
    gate_thr_v = pto.vmi.vbrc(pto.f32(GATE_THR), size=VL)

    for b in range(8):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        gate_s = pto.vmi.vload(gate_ub, b, size=1)
        gate_b = pto.vmi.vbrc(gate_s, size=VL)
        p_outer = pto.vmi.vcmp(gate_b, gate_thr_v, mask, "gt")
        k0 = s0
        k1 = s1
        k2 = s2
        k3 = s3
        for _t in range(64):
            k0 = pto.vmi.vmul(k0, scale_v, mask)
            k1 = pto.vmi.vmul(k1, scale_v, mask)
            k2 = pto.vmi.vmul(k2, scale_v, mask)
            k3 = pto.vmi.vmul(k3, scale_v, mask)
        kill0 = pto.vmi.vsel(pto.vmi.vcmp(k0, thr_b, mask, "gt"), neg_inf, k0)
        kill1 = pto.vmi.vsel(pto.vmi.vcmp(k1, thr_b, mask, "gt"), neg_inf, k1)
        kill2 = pto.vmi.vsel(pto.vmi.vcmp(k2, thr_b, mask, "gt"), neg_inf, k2)
        kill3 = pto.vmi.vsel(pto.vmi.vcmp(k3, thr_b, mask, "gt"), neg_inf, k3)
        s0 = pto.vmi.vsel(p_outer, kill0, s0)
        s1 = pto.vmi.vsel(p_outer, kill1, s1)
        s2 = pto.vmi.vsel(p_outer, kill2, s2)
        s3 = pto.vmi.vsel(p_outer, kill3, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g64_n0",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g64_n0(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(0):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(64):
            s0 = pto.vmi.vmul(s0, scale_v, mask)
            s1 = pto.vmi.vmul(s1, scale_v, mask)
            s2 = pto.vmi.vmul(s2, scale_v, mask)
            s3 = pto.vmi.vmul(s3, scale_v, mask)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g64_n2",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g64_n2(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(2):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(64):
            s0 = pto.vmi.vmul(s0, scale_v, mask)
            s1 = pto.vmi.vmul(s1, scale_v, mask)
            s2 = pto.vmi.vmul(s2, scale_v, mask)
            s3 = pto.vmi.vmul(s3, scale_v, mask)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g64_n4",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g64_n4(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(4):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(64):
            s0 = pto.vmi.vmul(s0, scale_v, mask)
            s1 = pto.vmi.vmul(s1, scale_v, mask)
            s2 = pto.vmi.vmul(s2, scale_v, mask)
            s3 = pto.vmi.vmul(s3, scale_v, mask)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g64_n6",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g64_n6(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(6):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(64):
            s0 = pto.vmi.vmul(s0, scale_v, mask)
            s1 = pto.vmi.vmul(s1, scale_v, mask)
            s2 = pto.vmi.vmul(s2, scale_v, mask)
            s3 = pto.vmi.vmul(s3, scale_v, mask)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

@pto.jit(
    name="CF7d_armC_g64_n8",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel_C_g64_n8(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
    gate_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(OFF_THR, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    gate_ub = pto.castptr(pto.const(OFF_GATE, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.mte_load(gate_gm, gate_ub, 0, GATE_BYTES, nburst=(1, GATE_BYTES, GATE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)
    scale_v = pto.vmi.vbrc(pto.f32(SCALE), size=VL)

    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for b in range(8):
        thr_s = pto.vmi.vload(thr_ub, b, size=1)
        thr_b = pto.vmi.vbrc(thr_s, size=VL)
        for _t in range(64):
            s0 = pto.vmi.vmul(s0, scale_v, mask)
            s1 = pto.vmi.vmul(s1, scale_v, mask)
            s2 = pto.vmi.vmul(s2, scale_v, mask)
            s3 = pto.vmi.vmul(s3, scale_v, mask)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

KERNELS = {
    ("A", 0, 8): kernel_A_g0_n8,
    ("B", 0, 8): kernel_B_g0_n8,
    ("C", 0, 0): kernel_C_g0_n0,
    ("C", 0, 2): kernel_C_g0_n2,
    ("C", 0, 4): kernel_C_g0_n4,
    ("C", 0, 6): kernel_C_g0_n6,
    ("C", 0, 8): kernel_C_g0_n8,
    ("A", 8, 8): kernel_A_g8_n8,
    ("B", 8, 8): kernel_B_g8_n8,
    ("C", 8, 0): kernel_C_g8_n0,
    ("C", 8, 2): kernel_C_g8_n2,
    ("C", 8, 4): kernel_C_g8_n4,
    ("C", 8, 6): kernel_C_g8_n6,
    ("C", 8, 8): kernel_C_g8_n8,
    ("A", 16, 8): kernel_A_g16_n8,
    ("B", 16, 8): kernel_B_g16_n8,
    ("C", 16, 0): kernel_C_g16_n0,
    ("C", 16, 2): kernel_C_g16_n2,
    ("C", 16, 4): kernel_C_g16_n4,
    ("C", 16, 6): kernel_C_g16_n6,
    ("C", 16, 8): kernel_C_g16_n8,
    ("A", 32, 8): kernel_A_g32_n8,
    ("B", 32, 8): kernel_B_g32_n8,
    ("C", 32, 0): kernel_C_g32_n0,
    ("C", 32, 2): kernel_C_g32_n2,
    ("C", 32, 4): kernel_C_g32_n4,
    ("C", 32, 6): kernel_C_g32_n6,
    ("C", 32, 8): kernel_C_g32_n8,
    ("A", 64, 8): kernel_A_g64_n8,
    ("B", 64, 8): kernel_B_g64_n8,
    ("C", 64, 0): kernel_C_g64_n0,
    ("C", 64, 2): kernel_C_g64_n2,
    ("C", 64, 4): kernel_C_g64_n4,
    ("C", 64, 6): kernel_C_g64_n6,
    ("C", 64, 8): kernel_C_g64_n8,
}


def build_kernel(arm, G, n_trips):
    arm = arm.upper()
    key = (arm, int(G), int(n_trips))
    if key not in KERNELS:
        raise KeyError(f"no module kernel for {key}")
    return KERNELS[key]


def reference(scores, thr, gate, G, gate_thr=GATE_THR):
    out = scores.astype(np.float32).copy()
    thr = thr.astype(np.float32)
    gate = gate.astype(np.float32)
    scale = np.float32(SCALE)
    for b in range(len(gate)):
        if float(gate[b]) > float(gate_thr):
            for _ in range(G):
                out = (out * scale).astype(np.float32)
            out = np.where(out > thr[b], np.float32(NEG_INF_F32), out).astype(np.float32)
    return out


def make_inputs(pgate, seed=27):
    rng = np.random.default_rng(seed)
    scores = rng.uniform(-3, 3, E).astype(np.float32)
    thr = rng.uniform(-1, 1, B).astype(np.float32)
    n_true = n_true_for(pgate)
    gate = np.full(B, -1.0, dtype=np.float32)
    if n_true > 0:
        gate[:n_true] = rng.uniform(0.1, 2.0, size=n_true).astype(np.float32)
    return scores, thr, gate, n_true


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
    scores, thr, gate, n_true = make_inputs(pgate)
    ref = reference(scores, thr, gate, G)
    arm = arm.upper()
    n_trips = n_true if arm == "C" else B
    kernel = build_kernel(arm, G, n_trips)

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
            f"oh[:4]={oh[:4].tolist()} scores[:4]={scores[:4].tolist()} gate={gate.tolist()}",
            flush=True,
        )
        raise AssertionError(f"mismatch max_diff={diff[idx]}")
    print(
        f"PASS {tag}  compile={cs:.3f}s launch={ls:.3f}s "
        f"maxabs={float(np.max(np.abs(oh - ref)))} n_true={n_true} n_trips={n_trips}"
    )
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("E", nargs="?", type=int, default=E)
    ap.add_argument("B", nargs="?", type=int, default=B)
    ap.add_argument("G", nargs="?", type=int, default=0)
    ap.add_argument("pgate", nargs="?", type=int, default=50)
    ap.add_argument("arm", nargs="?", type=str, default="A")
    ap.add_argument("--emit-mlir", action="store_true")
    args = ap.parse_args(argv)
    arm = args.arm.upper()
    tag = f"cf7d_e{E}_b{B}_g{args.G}_vl{VL}_pgate{args.pgate}_arm{arm}"
    if args.emit_mlir:
        n_true = n_true_for(args.pgate)
        n_trips = n_true if arm == "C" else B
        k = build_kernel(arm, args.G, n_trips)
        print(k.compile().mlir_text())
        return 0
    return run_acl(tag, arm, args.G, args.pgate)


if __name__ == "__main__":
    raise SystemExit(main())
