# ST ANALYSIS — CF7d Outer-Inner-If (PTO-DSL *d) — global-gate rev

**ST id:** `cf7d`  
**Date:** 2026-11-03 ~16:00 HKT (Asia/Hong_Kong) · global-gate 2-arm + **reduce-in-b** sibling  
**Frontend primary:** PTO-DSL / VMI Layer-D (`*d`) — arms **B / C** only (arm A dropped)  
**Sources:**
- Design: `benches/cf7d/ST_CF7_OUTER_INNER_IF_DESIGN.md` (+ `reports/ST_CF7_GLOBAL_GATE_REV.md` / pack)
- Kernel (hoisted): `benches/cf7d/cf7d_outer_inner_if.py` (rev2 global gate)
- Kernel (reduce-in-b): `benches/cf7d/cf7d_reduce_in_b.py` (+ `ST_CF7_REDUCE_IN_B.md`)
- Prior RESULT (STALE era): `benches/cf7d/ST_CF7D_FIRSTCUT_RESULT.md` · `cf7d_SUMMARY_firstcut.tsv`
- Trace: `benches/cf7d/EXPLORATION_TRACE.md`
- Machine target: pto-b10 · Ascend950PR_9599 opsim · `venv-ptoas-vmi-0.1.3`

**Status:** **PARTIAL** — (1) hoisted global-gate B/C land + smoke; (2) **NEW** reduce-in-b sibling (`cf7d_reduce_in_b.py`) so C cannot fake a win by hoisting `gate_on` once. Live VF/µs **UNMEASURED** for both eras. Prior first-cut numbers **STALE** (`era=per-b-outer`, Appendix H).

**Clean-run:** Do **not** invent el/cyc / VF / IPC. New-grid RESULT = UNMEASURED until oneshot re-run.

---

## 1. Aims

| Id | Aim |
|----|-----|
| **A1** | On a **global-gate** grid `G × {gate_on, gate_off}` (pgate labels as on/off proxies), find whether **arm C (`pto.if_`)** beats **arm B (preg+PAND / vand)** in **opsim VF cycle** (primary; µs secondary). Crossover rule (when both measured): smallest `G` s.t. `VF(C) ≤ 0.90 × VF(B)` at fixed gate_on (and document gate_off skip). |
| **A2** | Confirm **B** lowers a **real** `vcadd → vbrc → vcmp → vand` (PAND) path into kill — distinct opcode mix from always-enter slack; false outer still schedules G×B body but publish/squash via `vsel(p_outer)`. |
| **A3** | Confirm **C** uses **in-VF** `pto.if_(gate_on)` around the G nest (real scalar skip) — **no** host `n_trips = n_true` specialization. Gold: when `gate_sum ≤ GATE_SUM_THR`, neither scale nor kill applies. |
| **A4** | Land 2-arm *d kernels + ANALYSIS spine + `arms.yaml` under `benches/cf7d/`; compile smoke PASS; remeasure first-cut before suite DONE. SimtVF (*v) / SoftLib not Aims this cut. |
| **A5** | Land **reduce-in-b** sibling (`cf7d_reduce_in_b.py`): same `sum(gate)>GATE_SUM_THR` gold **recomputed inside each outer `b`** (no hoist) so arm C pays reduce dep every trip; arms B=vand / C=`pto.if_` mirror hoisted lowering. Compare B vs C under `era=cf7d-rib-ptodsl` (G primary). Compile smoke PASS; live KPI UNMEASURED until oneshot. |

Lineage: CF1–CF6 Parallel-only / nested predicates — **none** cover **one global outer gate** × inner kill with B=PAND vs C=Sreg `if_` (see design §1). Prior per-b outer CF7d first-cut is historical only (`era=per-b-outer`).

---

## How to run / reproduce

**Contract (t628u):** spine *d arms B/C share one checked-in kernel (`script:` in `report/arms.yaml`). SIMT Ascend default + SoftLib SIMT are `script_status: MISSING` this cut.

```bash
# From repo root on pto-b10 (Ascend950PR_9599 opsim). Do NOT source ~/projects/env.sh.
# Match Oct-8 PASS: ASC/set_env.sh + PYTHONPATH=DEPS:MLIR_PY only.
# Optional harness gate:
#   python3 -m scripts.st_report check --st cf7d

# Thin compile smoke (patterns only; no ACL):
python3 benches/cf7d/smoke_thin.py B
python3 benches/cf7d/smoke_thin.py C
# Reduce-in-b sibling smoke (Aims A5):
python3 benches/cf7d/smoke_thin_rib.py B
python3 benches/cf7d/smoke_thin_rib.py C

# Single cell — HOISTED (gate reduce once):
#   pgate 25 → gate_off; pgate 50/95 → gate_on
python3 benches/cf7d/cf7d_outer_inner_if.py 256 8 0 25 B
python3 benches/cf7d/cf7d_outer_inner_if.py 256 8 0 50 C
python3 benches/cf7d/cf7d_outer_inner_if.py 256 8 8 50 B
python3 benches/cf7d/cf7d_outer_inner_if.py 256 8 16 95 C

# Single cell — REDUCE-IN-B (same gold; reduce inside each b; tag cf7d_rib_*):
python3 benches/cf7d/cf7d_reduce_in_b.py 256 8 0 25 B
python3 benches/cf7d/cf7d_reduce_in_b.py 256 8 8 50 C
python3 benches/cf7d/cf7d_reduce_in_b.py 256 8 16 95 B

# Full first-cut oneshot (UPDATE after global-gate land — STALE until re-run):
# bash benches/cf7d/oneshot_ptodsl_cf7d.sh   # must drop arm A + n_trips path
```

**Expected PASS (functional):** gold `allclose(..., equal_nan=True)` with global `gate_on`. **VF/µs:** UNMEASURED this rev — do not copy `era=per-b-outer` tables into live §4.

**SIMT Ascend / SoftLib:** MISSING in-repo — next-cmd in §3.2 / §3.3.

---

## 2. Test-case design

| Knob | Value | Role |
|------|-------|------|
| `E` | 256 | elems |
| `B` | 8 | outer trip count (always scheduled on B; C may skip nest via `if_`) |
| `VL` | 64 | PTO-DSL vector length (`N_CHUNKS = E/VL = 4`) |
| `G` | {0, 8, 16, 32, 64} | fat inner serial under `gate_on`; kill **inside** G |
| `pgate` | {25, 50, 95} labels | **proxy only**: `pgate≥50` → `gate_on` (sum(gate)>0); else `gate_off` |
| `GATE_SUM_THR` | 0.0 | global outer compare |
| dtype | f32 | scores / thr / gate / out |
| Gold | `allclose(..., equal_nan=True)` | **one** global `if sum(gate) > THR:` wrapping `for b: for G: scale+kill` |

**Arms (same gold) — 2 only:**

| Arm | Outer gate | How outer reaches inner | Tag |
|-----|------------|-------------------------|-----|
| **B · preg+PAND** | `vcadd(gate)→vstore→vload→vbrc→vcmp` → `p_outer`; always enter B×G | kill: `vsel(vand(vcmp(scores>thr), p_outer), -inf, …)` then `vsel(p_outer, k, s)` publish | `_armB` |
| **C · scalar_if** | same reduce → `load_scalar` → `gate_on = gs > THR` | `with pto.if_(gate_on)` wraps G×(scale+kill); else assign identity — **no** host `n_trips` | `_armC` |

**Dropped:** arm **A** pure_slack (Lok 2026-11-03). Prior 3-arm / per-b tables → **Appendix H** (`era=per-b-outer`, STALE).

Tag scheme: `cf7d_e{E}_b{B}_g{G}_vl{VL}_pgate{P}_arm{B|C}`.

**Pass criteria (cell):** status PASS + gold; crossover uses **VF cycle** only when remeasured (Lok).

---

## 2b. Key code slices — global gate × arm lowering

Source: `benches/cf7d/cf7d_outer_inner_if.py` (module-level `@pto.jit` per `(arm, G)`).  
Gold host `reference()` is true global outer `if`; *d arms differ in how that outer gate is lowered.

| Knob | How it enters the kernel |
|------|--------------------------|
| **`G`** | Unrolled fat inner `for _t in range(G): vmul + kill` under each outer `b`. `G=0` → gate reduce + B loop overhead only (no kill). |
| **`pgate` / gate_on** | Host builds `gate[]` all-positive (`on`) or all-negative (`off`) so `sum(gate) ? GATE_SUM_THR`. **Both** B and C compile `for b in range(8)` — C skips body via `pto.if_`, not via `n_trips`. |

### Gold (host)

```python
gate_on = float(np.sum(gate)) > float(gate_sum_thr)
for b in range(len(thr)):
    if gate_on:
        for _ in range(G):
            out = (out * scale).astype(np.float32)
            out = np.where(out > thr[b], -inf, out)
```

### Arm B — preg+PAND (`kernel_B_g8` shape)

```python
gate_v = pto.vmi.vload(gate_ub, 0, size=B)
gs = pto.vmi.vcadd(gate_v, mask_B)
pto.vmi.vstore(gs, scratch_ub, 0, mask_B)
gs_s = pto.vmi.vload(scratch_ub, 0, size=1)
p_outer = pto.vmi.vcmp(pto.vmi.vbrc(gs_s, size=VL), thr_sum_v, mask, "gt")
for b in range(8):
    thr_b = pto.vmi.vbrc(pto.vmi.vload(thr_ub, b, size=1), size=VL)
    for _t in range(G):
        k0 = pto.vmi.vmul(s0, scale_v, mask)
        k0 = pto.vmi.vsel(pto.vmi.vand(pto.vmi.vcmp(k0, thr_b, mask, "gt"), p_outer), neg_inf, k0)
        s0 = pto.vmi.vsel(p_outer, k0, s0)   # publish only when gate_on
```

### Arm C — `pto.if_` (`kernel_C_g8` shape)

```python
gs = pto.vmi.vcadd(pto.vmi.vload(gate_ub, 0, size=B), mask_B)
pto.vmi.vstore(gs, scratch_ub, 0, mask_B)
gs_sc = pto.load_scalar(scratch_ub, 0)
gate_on = gs_sc > pto.f32(GATE_SUM_THR)
for b in range(8):
    thr_b = pto.vmi.vbrc(pto.vmi.vload(thr_ub, b, size=1), size=VL)
    with pto.if_(gate_on) as br:
        with br.then_:
            t0, t1, t2, t3 = s0, s1, s2, s3
            for _t in range(G):
                t0 = pto.vmi.vmul(t0, scale_v, mask)
                t0 = pto.vmi.vsel(pto.vmi.vcmp(t0, thr_b, mask, "gt"), neg_inf, t0)
                # … chunks …
            br.assign(s0=t0, s1=t1, s2=t2, s3=t3)
        with br.else_:
            br.assign(s0=s0, s1=s1, s2=s2, s3=s3)
    s0, s1, s2, s3 = br.s0, br.s1, br.s2, br.s3
```

**Read-out:** B always pays `vmul`/kill issue under false outer (squash via vand/vsel); C is intended to **skip** the then-nest when `gate_on` false. Remeasure must prove C body/VF drop on `gate_off`.

---


## 2c. Sibling case — reduce-in-b (Aims A5 · `era=cf7d-rib-ptodsl`)

**Why:** Hoisted C computes `gate_on` once then reuses across B trips → reduce tax paid once; can fake C win vs B. Lok: need a case where reduce runs **inside** the `b` loop.

**Gold (numerically = hoisted):** for each `b`: recompute `gate_on_b = sum(gate) > GATE_SUM_THR`; if on, run G×(scale+kill). Kernel **must not hoist** the reduce.

**Arm B (rib):** inside `for b`: `vcadd→vstore→vload→vbrc→vcmp` → `p_outer`; then G×(vmul + `vand` kill + `vsel` publish).

**Arm C (rib):** inside `for b`: same reduce → `load_scalar` → `with pto.if_(gate_on)` wraps G nest.

Source: `benches/cf7d/cf7d_reduce_in_b.py` · design note `ST_CF7_REDUCE_IN_B.md` · smoke `smoke_thin_rib.py`.  
Tag: `cf7d_rib_e{E}_b{B}_g{G}_vl{VL}_pgate{P}_arm{B|C}`.

### Key slice (arm B g8 — reduce inside b)

```python
for b in range(8):  # NO hoist
    gate_v = pto.vmi.vload(gate_ub, 0, size=B)
    gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
    pto.vmi.vstore(gs, scratch_ub, 0, mask1)
    gs_s = pto.vmi.vload(scratch_ub, 0, size=1)
    p_outer = pto.vmi.vcmp(pto.vmi.vbrc(gs_s, size=VL), thr_sum_v, mask, "gt")
    # ... thr_b; for _t in range(G): vmul + vand(kill,p_outer) + vsel publish
```

### Key slice (arm C g8 — reduce + if_ inside b)

```python
for b in range(8):  # NO hoist
    gate_v = pto.vmi.vload(gate_ub, 0, size=B)
    gs = pto.vmi.vcadd(gate_v, mask_B, reassoc=True)
    # ... store; load_scalar
    gate_on = gs_sc > pto.f32(GATE_SUM_THR)
    with pto.if_(gate_on) as br:
        with br.then_:
            # G×(scale+kill)
            br.assign(...)
        with br.else_:
            br.assign(s0=s0, s1=s1, s2=s2, s3=s3)
```

**Live KPI:** UNMEASURED (`era=cf7d-rib-ptodsl`). Do not invent VF. Hoisted case remains the A1 primary until both measured.

---
## 3. Layout and schedule (SIMT and SIMD)

### 3.1 SIMD PTO-DSL (*d) — landed, KPI UNMEASURED this era

- **Layout:** GM→UB MTE2 load scores/thr/gate; 4× VL=64 RF chunks; global `vcadd` gate reduce; per `b`: thr `vbrc`; inner `G` scale+kill; store MTE3.
- **Arm B:** always `for b in range(8)` + `for G`; `p_outer` preg AND'd into kill; `vsel(p_outer)` publish.
- **Arm C:** same B loop; `pto.if_(gate_on)` wraps G nest — **in-VF** scalar if (not host `n_trips`).
- **Era (live):** `era=cf7d-global-gate-ptodsl` — **UNMEASURED** VF until oneshot re-run.
- **Era (historical appendix):** `era=per-b-outer` — see Appendix H.

### 3.2 SIMT Ascend default — **MISSING**

No SimtVF / `T.Parallel` CF7 (*v) land.  
**Next-cmd:** design+land Simt CF7 B/C per design; oneshot twin TBD.

### 3.3 SIMT SoftLib hand-craft — **MISSING**

Not in Aims for this cut.  
**Next-cmd:** only if Aims add SoftLib SIMT A5.

---

## 4. SIMD PTO-DSL performance (primary)

**Primary KPI:** opsim `VF_cycles`. Secondary: wall µs, `body_instr`.  
**el/cyc** = `E / VF_cycles` = `256 / VF`.

### 4.1 Live grid (`era=cf7d-global-gate-ptodsl`) — **UNMEASURED**

| G | gate | B VF / µs / body | C VF / µs / body | C ≤ 0.90·B? |
|--:|------|------------------|------------------|:-----------:|
| * | on/off | **UNMEASURED** | **UNMEASURED** | — |

**Next-cmd:** update `oneshot_ptodsl_cf7d.sh` for arms `{B,C}` + global gold; run on pto-b10; fill TSV + RESULT; then refresh §4 / SEMANTICS_GATE / PR9.

### 4.2–4.4 Prior first-cut numbers — **STALE**

See **Appendix H** (`era=per-b-outer`). Do **not** use those VF values as live A1 crossover for global-gate arms.

### 4.5 SEMANTICS_GATE (per arm · before any DONE/SoT)

CF7d is **outer/inner if control-flow**, not SoftLib≻SIMD reduce. No SoftLib≻SIMD claim. No suite DONE.

```
SEMANTICS_GATE: INCONCLUSIVE — armB preg+PAND — S1 gold code matches global Aims; S2 body has vcadd/vand path as designed; S3/S4 OPPROF+VF UNMEASURED this era (compile smoke only) — not DONE
SEMANTICS_GATE: INCONCLUSIVE — armC pto.if_ — S1 gold matches; S2 uses in-VF if_ (no n_trips); S3/S4 need dumps proving skip on gate_off + VF vs B — UNMEASURED — not DONE
SEMANTICS_GATE: INCONCLUSIVE — rib armB preg+PAND — S1/S2 reduce-in-b gold+code match Aims A5; S3/S4 OPPROF+VF UNMEASURED (era=cf7d-rib-ptodsl) — not DONE
SEMANTICS_GATE: INCONCLUSIVE — rib armC pto.if_ — S1/S2 per-b reduce then if_ (no hoist); S3/S4 need dumps proving reduce×B + skip — UNMEASURED — not DONE
```

---

## 5. SIMT (Ascend default) lowering performance

**MISSING** — no CF7 SimtVF (*v) this cut.  
**Next-cmd:** land Simt CF7 B/C + harvest VF/body.

---

## 6. SIMT hand-craft (SoftLib) analysis

**MISSING** — not measured; not Aims.  
**Next-cmd:** only if Aims add SoftLib SIMT for CF7.

---

## 7. Running IPC + PR9 overlay figures

**Trigger note:** SoftLib≻SIMD / both-frontend bar = **N/A** (no Simt/SoftLib). Intra-*d B vs C PR9 **required after** dumps exist.

**Status (live era):** **MISSING** — no global-gate dumps yet.  
**Next-cmd:** after oneshot, `make_pr9_running_ipc_overlay.py` per cell → `benches/cf7d/plots/*_running_ipc_pr9.png` with `plot_style=opcode_lanes_ipc`.

**Historical:** 27/27 PR9 under `reports/cf7d_plots/` / `benches/cf7d/plots/` for `era=per-b-outer` first-cut — retained on disk; **not** evidence for global-gate KPI.

---

## 8. Per-segment opcode mix + stage mid-IPC

**SoftLib≻SIMD / STAGE_TAXONOMY reduce map:** **N/A** (CF policy).  
**Live mix:** **MISSING** until `instr_exe.csv` for B/C global-gate cells.  
Expect (design, not measured): B shows `VCADD`/`VAND`/`VCMP`/`VSEL`/`VMUL`; C shows branch / if lowering + fewer body ops on `gate_off`.

**Historical A≡B mix** (`era=per-b-outer`): Appendix H / `cf7d_OPCODE_MIX.md`.

---

## 9. ISA mix SIMT vs SIMD (per N×4B → el/cyc bridge)

**SIMT vs SIMD both-frontend:** **MISSING / N/A**.  
**Intra-*d B vs C:** **MISSING** until remeasure.  
**Historical A vs C bridge** (`era=per-b-outer`): Appendix H — |pred/meas−1|=0 on first-cut cells.

---

## 10. Conclusion

| Arm | Role | This cut |
|-----|------|----------|
| **SIMD PTO-DSL B preg+PAND** | always-enter + vand into kill | Kernel landed; compile smoke; **VF UNMEASURED** |
| **SIMD PTO-DSL C pto.if_** | in-VF scalar skip of G nest | Kernel landed; compile smoke; **VF UNMEASURED** |
| **SIMT Ascend default** | — | **MISSING** |
| **SIMT SoftLib** | — | **MISSING** |

**Do not claim:** SoftLib≻SIMD; suite / MB DONE; live crossover; reuse of per-b VF as global-gate SoT.

**Still open:** oneshot remeasure **hoisted** global-gate grid + **reduce-in-b** G sweep; fill-in G∈{8,32} if not in first pass; Simt CF7 (*v); SoftLib only if Aims add A5; PR merge needs Lok yes.

---

## 11. Trace pointer

`benches/cf7d/EXPLORATION_TRACE.md` — global-gate redesign + smoke + doc land (2026-11-03). Prior dump-sync/PR9 steps retained in TRACE timeline.

---

## After RESULT — COMPARE

### KPI COMPARE harvest

| vs | Status |
|----|--------|
| Prior dig / SoftLib≻SIMD el/cyc | **KPI N/A** — CF policy ST |
| `era=per-b-outer` first-cut VF tables | **STALE** — different gold (per-b outer + host n_trips C); kept in Appendix H; **not** folded into live §4 |
| Live global-gate B vs C | **UNMEASURED** — next oneshot |

### Motif-seed Coverage COMPARE

| Motif / recipe tag | Coverage | Note |
|--------------------|:--------:|------|
| `outer_inner_if` nest | ✓ | design + kernel |
| `global_gate_sum` | ✓ | gold + B/C reduce |
| `armB_preg_pand` / vand | ✓ code | KPI UNMEASURED |
| `armC_pto_if_` in-VF | ✓ code | KPI UNMEASURED; **no** n_trips |
| `armA_pure_slack` | ✗ dropped | Lok 2026-11-03; historical only |
| `R*_x_pgate` per-b first-cut | ✓ historical | Appendix H STALE |
| `crossover_VF_0.90` live | ✗ / stop | needs remeasure |
| PR9 `opcode_lanes_ipc` live | ✗ / MISSING | next-cmd after dumps |
| SimtVF CF7 (*v) | ✗ / stop | Aims A4 before *v |
| SoftLib SIMT CF7 | ✗ / stop | not Aims |
| SoftLib≻SIMD both-frontend | N/A | no both-frontend claim |

---

## TODO / open clarification

1. **Remeasure** global-gate B/C (hoisted) on pto-b10; replace UNMEASURED §4; refresh SEMANTICS_GATE to PASS|PARTIAL only with dumps.
1b. **Remeasure** reduce-in-b B/C (`cf7d_reduce_in_b.py`) G sweep; fill `dsl_rib_*` SUMMARY rows; do not fold rib VF into hoisted A1 without era= label.
2. **Update oneshot** to drop arm A / `n_trips` and emit `gate_on` column.
3. **t821u:** Appendix H retains all prior 3-arm / per-b / first-cut VF + el/cyc + crossover + opcode A≡B tables — do not delete. Marked **STALE** pending prove-by-remeasure that global-gate supersedes them for A1.
4. Prior ANALYSIS claimed arm-C = host `n_trips` and OPPROF zero `SIMT_BRANCH` — **contradicts** new C intent (`pto.if_`). Until dumps prove branch/skip, keep both claims: historical OPPROF under Appendix H; live C expected-instr = if_/scalar branch family (S3 pending).

---

## Appendix H — Historical first-cut (`era=per-b-outer`, **STALE**)

> Copied from ANALYSIS tip `0bf351e` / `ST_CF7D_FIRSTCUT_RESULT.md` for t821u no-info-loss.  
> Gold was **per-b** `if gate[b] > thr`; arm C was **host `n_trips=n_true`**. **Not** valid KPI for global-gate rev.

### H.1 First-cut grid (27/27 PASS) — STALE

| G | p_gate | A VF / µs / body† | B VF / µs / body† | C VF / µs / body† | C ≤ 0.90·min(A,B)? |
|--:|-------:|------------------|------------------|------------------|:------------------:|
| 0 | 25 | **125** / 0.89 / 215 | **125** / 0.89 / 215 | **69** / 0.89 / 94 | **YES** (thr 112.5) |
| 0 | 50 | 125 / 0.89 / 215 | 125 / 0.89 / 215 | **77** / 0.85 / 114 | **YES** |
| 0 | 95 | 125 / 0.89 / 215 | 125 / 0.89 / 215 | **96** / 0.90 / 154 | **YES** |
| 16 | 25 | **643** / 1.17 / 727 | **643** / 1.17 / 727 | **189** / 0.96 / 226 | **YES** (thr 578.7) |
| 16 | 50 | 643 / 1.17 / 727 | 643 / 1.17 / 727 | **342** / NA / 374 | **YES** |
| 16 | 95 | 643 / 1.17 / 727 | 643 / 1.17 / 727 | **634** / 1.19 / 670 | **NO** |
| 64 | 25 | **2161** / 1.98 / 2263 | **2161** / 1.98 / 2263 | **573** / 1.15 / 610 | **YES** (thr 1944.9) |
| 64 | 50 | 2161 / 1.98 / 2263 | 2161 / 1.98 / 2263 | **1111** / 1.45 / 1142 | **YES** |
| 64 | 95 | 2161 / 1.98 / 2263 | 2161 / 1.98 / 2263 | **2171** / 2.01 / 2206 | **NO** (C ≳ A) |

† oneshot TSV body. Dump framing-excluded: A/B G0=204, G16=716, G64=2252; C G0 p25=83.

### H.2 el/cyc (E=256) — STALE

| G | p_gate | A el/cyc | B el/cyc | C el/cyc |
|--:|-------:|---------:|---------:|---------:|
| 0 | 25 | 2.048 | 2.048 | **3.710** |
| 0 | 50 | 2.048 | 2.048 | 3.325 |
| 0 | 95 | 2.048 | 2.048 | 2.667 |
| 16 | 25 | 0.398 | 0.398 | **1.355** |
| 16 | 50 | 0.398 | 0.398 | 0.749 |
| 16 | 95 | 0.398 | 0.398 | 0.404 |
| 64 | 25 | 0.118 | 0.118 | **0.447** |
| 64 | 50 | 0.118 | 0.118 | 0.230 |
| 64 | 95 | 0.118 | 0.118 | 0.118 |

### H.3 Crossover call (A1) — STALE first-cut

| p_gate | crossover `G` | notes |
|-------:|--------------:|-------|
| 25 | **0** | C wins every first-cut G |
| 50 | **0** | C wins every first-cut G |
| 95 | **0** | wins only at G=0; fails 0.90-bar at G=16 / G=64 |

**STALE call:** crossover R\* = G = 0; **A≡B** every cell; C = host trip skip (not in-VF `if_`).

### H.4 SEMANTICS_GATE (historical era) — STALE

```
SEMANTICS_GATE: PASS — armA pure_slack — S1 gold PASS; S2 VF flat vs p_gate; S3 VCMP/VSEL; S4 VF=125/643/2161; PARTIAL suite
SEMANTICS_GATE: PASS — armB slack_and — S1 gold PASS; identical opcode to A (AND ineffective as-lowered); S4 VF≡A
SEMANTICS_GATE: PASS — armC scalar_if — S1 gold PASS; S2/S4 track n_true; crossover G=0 first-cut only
```

### H.5 Headline ratios — STALE

At G0 p25, `VF(A)/VF(C) = 125/69 ≈ 1.81`; G64 p25 `2161/573 ≈ 3.77`; G64 p95 `2161/2171 ≈ 0.995`.

---

## Checklist (st-report)

- [x] Aims `A1…A5` cited (hoisted global-gate + reduce-in-b sibling)
- [x] **How to run / reproduce** (t628u) + `report/arms.yaml` (`script:` for B/C; MISSING for SIMT/SoftLib)
- [x] Primary path `benches/cf7d/ANALYSIS.md`
- [x] Key code slices §2b (B vand/PAND / C `pto.if_`; G kill-inside)
- [x] SEMANTICS_GATE one-liner per **new** arm (**INCONCLUSIVE** pending remeasure)
- [x] t821u: prior 3-arm / per-b / crossover tables retained in Appendix H as **STALE**
- [x] No SoftLib≻SIMD / no invented live VF
- [x] TRACE linked
- [x] Kernel check-in `benches/cf7d/cf7d_outer_inner_if.py` (+ `smoke_thin.py`)
- [ ] Live RESULT / PR9 / opcode mix — **MISSING** next oneshot
- [ ] Suite / MB DONE — **forbidden** (**PARTIAL**)
- [ ] Merge — **needs Lok yes**
