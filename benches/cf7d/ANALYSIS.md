# ST ANALYSIS — CF7d Outer-Inner-If (PTO-DSL *d) — global-gate rev

**ST id:** `cf7d`  
**Date:** 2026-11-03 ~18:30 HKT (Asia/Hong_Kong) · dual-era oneshot RESULT + conclusion  
**Frontend primary:** PTO-DSL / VMI Layer-D (`*d`) — arms **B / C** only (arm A dropped)  
**Sources:**
- Design: `benches/cf7d/ST_CF7_OUTER_INNER_IF_DESIGN.md` (+ `reports/ST_CF7_GLOBAL_GATE_REV.md` / pack)
- Kernel (hoisted): `benches/cf7d/cf7d_outer_inner_if.py` (rev2 global gate)
- Kernel (reduce-in-b): `benches/cf7d/cf7d_reduce_in_b.py` (+ `ST_CF7_REDUCE_IN_B.md`)
- Prior RESULT (STALE era): `benches/cf7d/ST_CF7D_FIRSTCUT_RESULT.md` · `cf7d_SUMMARY_firstcut.tsv`
- Trace: `benches/cf7d/EXPLORATION_TRACE.md`
- Machine target: pto-b10 · Ascend950PR_9599 opsim · `venv-ptoas-vmi-0.1.3`

**Status:** **MEASURED + PR9/S3 filled** — dual-era oneshot `oneshot_ptodsl_cf7d_rev.sh` on pto-b10 **2026-11-03 17:44–18:26 HKT (Asia/Hong_Kong)** · tip kernel `0e5f603` · OUT `/tmp/st_simtvf_parallel/ptodsl_cf7d_rev/` · TSV `benches/cf7d/report/SUMMARY_dual_era_oneshot.tsv` · **24/24 PASS**. Live VF/µs for `era=cf7d-global-gate-ptodsl` (hoisted) and `era=cf7d-rib-ptodsl` (reduce-in-b). Prior first-cut **STALE** (`era=per-b-outer`, Appendix H). **PR9** `plot_style=opcode_lanes_ipc` overlays **24/24** key cells under `benches/cf7d/plots/{cf7d_,cf7d_rib_}e256_b8_g{0,16,64}_vl64_pgate{25,50}_arm{B,C}_running_ipc_pr9.png` (+ JSON / staged). **S3** from latest `instr_exe.csv` → `plots/cf7d_dual_era_S3_opcode_fragment.md` + `cf7d_dual_era_pr9_s3_harvest.json`.

**Clean-run:** VF/µs from SUMMARY; PR9 sidecars for dump `vf_cyc`/`n_issues`/`el_cyc`; opcode counts from `instr_exe.csv` only — do **not** invent. SoftLib/SimtVF still MISSING; merge needs Lok yes. Rib C gate_off VF growth G16→G64 (flat body=152) remains open for pipe/IQ cause (not invented).

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

# Dual-era oneshot (hoisted + rib; G in {0,16,64} x pgate in {25,50} x B,C):
bash benches/cf7d/oneshot_ptodsl_cf7d_rev.sh
# OUT default: /tmp/st_simtvf_parallel/ptodsl_cf7d_rev/SUMMARY.tsv
# Checked-in copy: benches/cf7d/report/SUMMARY_dual_era_oneshot.tsv
```

**Expected PASS (functional):** gold `allclose(..., equal_nan=True)` with global `gate_on`. Note: `gate_on` + G>0 cells report `maxabs=NA` in TSV (kill to -inf / NaN path) but status **PASS** via `equal_nan=True`. **VF/µs:** §4 tables from 2026-11-03 rev oneshot.

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
- **Era (live):** `era=cf7d-global-gate-ptodsl` + `era=cf7d-rib-ptodsl` — **MEASURED** §4 (2026-11-03 oneshot_rev).
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
**Source:** `benches/cf7d/report/SUMMARY_dual_era_oneshot.tsv` (pto-b10 · Ascend950PR_9599 · 2026-11-03 17:44–18:26 HKT). Grid: G∈{0,16,64} × pgate∈{25,50} × arms B,C × eras hoisted+rib = **24/24 PASS**.

`pgate=25` → `gate_off` (`gate_on=False`); `pgate=50` → `gate_on=True`.

### 4.1 Hoisted (`era=cf7d-global-gate-ptodsl`)

| G | gate | B VF / µs / body | C VF / µs / body | C/B | C ≤ 0.90·B? |
|--:|------|------------------|------------------|----:|:-----------:|
| 0 | off | **68** / 0.89 / 77 | **68** / 0.89 / 77 | 1.000 | NO (tie) |
| 0 | on | **68** / 0.89 / 77 | **68** / 0.89 / 77 | 1.000 | NO (tie) |
| 16 | off | **2237** / 2.03 / 2665 | **821** / 1.58 / 111 | 0.367 | **YES** |
| 16 | on | **2237** / 2.03 / 2665 | **1283** / 1.84 / 1674 | 0.574 | **YES** |
| 64 | off | **8827** / 5.69 / 10345 | **830** / 1.62 / 111 | 0.094 | **YES** |
| 64 | on | **8827** / 5.69 / 10345 | **5800** / 4.35 / 6282 | 0.657 | **YES** |

**A1 crossover (hoisted, gate_on):** smallest measured G with `VF(C) ≤ 0.90×VF(B)` is **G=16** (G=0 is tie). At gate_off, C already wins at G=16 (and G=64); body stays **111** while B body scales with G → C **skips** the G nest.

**B gate sensitivity:** B VF **flat vs gate** at fixed G (always-enter + vand squash) — matches A2 intent.

### 4.2 Reduce-in-b (`era=cf7d-rib-ptodsl`)

| G | gate | B VF / µs / body | C VF / µs / body | C/B | C ≤ 0.90·B? |
|--:|------|------------------|------------------|----:|:-----------:|
| 0 | off | **75** / 0.89 / 84 | **75** / 0.89 / 84 | 1.000 | NO (tie) |
| 0 | on | **75** / 0.89 / 84 | **75** / 0.89 / 84 | 1.000 | NO (tie) |
| 16 | off | **2118** / 1.96 / 2693 | **1386** / 1.92 / 152 | 0.654 | **YES** |
| 16 | on | **2118** / 1.96 / 2693 | **1609** / 1.91 / 1786 | 0.760 | **YES** |
| 64 | off | **8865** / 5.71 / 10373 | **3234** / 2.89 / 152 | 0.365 | **YES** |
| 64 | on | **8865** / 5.71 / 10373 | **5471** / 4.20 / 6348 | 0.617 | **YES** |

**A1 crossover (rib, gate_on):** likewise **G=16** first measured win (G=0 tie). C still ≤0.90·B at G=16/64 on and off.

**Rib tax vs hoisted (same G/gate):** B G0 75 vs 68 (+7 VF); C gate_off G16 1386 vs 821; C gate_off G64 3234 vs 830 — rib pays clear extra when C would otherwise hoist `gate_on` once. C body on gate_off stays small (152) vs B thousands.

### 4.3 el/cyc (E=256) — both eras

| Era | G | gate | B el/cyc | C el/cyc |
|-----|--:|------|---------:|---------:|
| hoisted | 0 | off/on | 3.765 | 3.765 |
| hoisted | 16 | off | 0.114 | **0.312** |
| hoisted | 16 | on | 0.114 | **0.199** |
| hoisted | 64 | off | 0.029 | **0.308** |
| hoisted | 64 | on | 0.029 | **0.044** |
| rib | 0 | off/on | 3.413 | 3.413 |
| rib | 16 | off | 0.121 | **0.185** |
| rib | 16 | on | 0.121 | **0.159** |
| rib | 64 | off | 0.029 | **0.079** |
| rib | 64 | on | 0.029 | **0.047** |

### 4.4 Does rib remove the hoist artifact? (A5)

| Evidence | Hoisted C | Rib C | Read |
|----------|-----------|-------|------|
| gate_off body @G16/G64 | 111 / 111 | 152 / 152 | both skip fat body |
| gate_off VF @G16→G64 | 821 → 830 (~flat) | 1386 → 3234 (**grows**) | hoisted C win partly from **once** reduce + flat skip; rib still skips body but VF grows (S3/PR9 open — do not invent cause) |
| C/B @G64 gate_off | 0.094 | 0.365 | rib **shrinks** C's relative win vs B |
| C/B @G16 gate_on | 0.574 | 0.760 | rib still C≻B but closer |
| C/B @G64 gate_on | 0.657 | 0.617 | both C≻B under 0.90 bar |

**Call:** rib **reduces** the hoist artifact (C no longer nearly free on gate_off as G grows) but **does not remove** C≻B — scalar `pto.if_` skip of G×(scale+kill) still beats always-enter B+vand on this grid. Do **not** fold rib VF into hoisted A1 without `era=`.

### 4.5 Prior first-cut — **STALE**

See **Appendix H** (`era=per-b-outer`). Not live SoT.

### 4.6 SEMANTICS_GATE (per arm · S1–S4 refreshed 2026-11-03 HKT)

CF7d is **outer/inner if control-flow**, not SoftLib≻SIMD reduce. No SoftLib≻SIMD claim. S3 from latest `instr_exe.csv` under `/tmp/st_simtvf_parallel/ptodsl_cf7d_rev/` (see `plots/cf7d_dual_era_S3_opcode_fragment.md`).

**Expected instr (from lowering):**
- **B:** always enter G nest → `RV_VCMP_GT` + `RV_PAND` + `RV_VMUL` + `RV_VSEL` (preg+PAND / vand squash); body scales with G; **flat vs gate**.
- **C:** `pto.if_` → scalar gate (`RV_SJUMPI` / `RV_SCBZI` / `RV_SAND` / `RV_SMOVI` / `RV_SZEROEXT`); **gate_off** skips fat nest (no VMUL/PAND/VSEL); **gate_on** pays `RV_VMUL`+`RV_VCMP_GT`+`RV_VSEL` **without** `RV_PAND`.

```
SEMANTICS_GATE: PASS — armB preg+PAND hoisted — S1 gold PASS; S2 vcadd→preg→PAND path; S3 G16/64: PAND=512/2048 VMUL=512/2048 VCMP=513/2049 VSEL=1024/4096 identical on/off; S4 VF flat vs gate 68/2237/8827
SEMANTICS_GATE: PASS — armC pto.if_ hoisted — S1 gold PASS; S2 in-VF if_ (no n_trips); S3 gate_off body=111 (SJUMPI/SCBZI, no fat VMUL); gate_on G16/64 VMUL=512/2048 + VSEL=512/2048 + PAND=0; S4 gate_off VF~flat 821→830; gate_on tracks G
SEMANTICS_GATE: PASS — rib armB preg+PAND — S1/S2 reduce-in-b; S3 same PAND/VMUL/VCMP/VSEL family + rib VLDI/VSTI tax; S4 VF≈B-hoisted at G≥16
SEMANTICS_GATE: PASS — rib armC pto.if_ — S1/S2 per-b reduce+if_; S3 gate_off body=152 (SJUMPI×8 + VCADD×9, no fat VMUL); gate_on VMUL/VSEL without PAND; S4 C≤0.90·B at G≥16; VF growth G16→G64 gate_off with flat body remains open for pipe cause (not invented)
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

**Trigger note:** SoftLib≻SIMD / both-frontend bar = **N/A** (no Simt/SoftLib). Intra-*d B vs C both-arm VF compare → PR9 **required** for both arms.

**Status (live eras):** **OK — 24/24** key cells with `plot_style=opcode_lanes_ipc` dual-pane `*_running_ipc_pr9.png` + JSON sidecars under `benches/cf7d/plots/` (harvest 2026-11-03 HKT from `core0.veccore0.instr_popped_log.dump`). IPC-only PNGs rejected — these are PR9 overlays. Manifest: `plots/cf7d_dual_era_pr9_s3_harvest.json`.

| era | G | gate | arm | dump vf_cyc | n_issues | el/cyc (E/vf) | PNG |
|-----|--:|------|-----|------------:|---------:|--------------:|-----|
| hoisted | 0 | off/on | B/C | 32 | 18 | 8.00 | `cf7d_e256_b8_g0_vl64_pgate{25,50}_arm{B,C}_running_ipc_pr9.png` |
| hoisted | 16 | off/on | B | 2201 | 2600 | 0.116 | `…_g16_…_pgate{25,50}_armB_…` |
| hoisted | 16 | off | C | 32 | 9 | 8.00 | `…_g16_…_pgate25_armC_…` (skip) |
| hoisted | 16 | on | C | 768 | 1119 | 0.333 | `…_g16_…_pgate50_armC_…` |
| hoisted | 64 | off/on | B | 8791 | 10280 | 0.029 | `…_g64_…_armB_…` |
| hoisted | 64 | off | C | 32 | 9 | 8.00 | `…_g64_…_pgate25_armC_…` (skip) |
| hoisted | 64 | on | C | 1640 | 2111 | 0.156 | `…_g64_…_pgate50_armC_…` |
| rib | 0 | off/on | B/C | 39 | 25 | 6.56 | `cf7d_rib_e256_b8_g0_…` |
| rib | 16 | off/on | B | 2081 | 2628 | 0.123 | `cf7d_rib_…_g16_…_armB_…` |
| rib | 16 | off | C | 63 | 21 | 4.06 | `cf7d_rib_…_g16_…_pgate25_armC_…` |
| rib | 16 | on | C | 761 | 1109 | 0.336 | `cf7d_rib_…_g16_…_pgate50_armC_…` |
| rib | 64 | off/on | B | 8828 | 10308 | 0.029 | `cf7d_rib_…_g64_…_armB_…` |
| rib | 64 | off | C | 31 | 12 | 8.26 | `cf7d_rib_…_g64_…_pgate25_armC_…` (skip; SUMMARY VF still grows — open) |
| rib | 64 | on | C | 2085 | 2523 | 0.123 | `cf7d_rib_…_g64_…_pgate50_armC_…` |

Dump `vf_cyc` is framing-excluded vs SUMMARY VF (e.g. hoisted B G16 dump 2201 vs SUMMARY 2237) — use SUMMARY for KPI; dump for PR9 overlay only.

**Reproduce (one cell):**
```bash
cd /tmp/tilelangGym-cf7d
PYTHONPATH=. /home/happybot/projects/venv-ptoas-vmi-0.1.3/bin/python \
  REGRESSION_salvage_2026-09-17/scripts/make_pr9_running_ipc_overlay.py \
  --dump-dir /tmp/st_simtvf_parallel/ptodsl_cf7d_rev/opsim_cf7d_e256_b8_g16_vl64_pgate50_armB \
  --out benches/cf7d/plots/cf7d_e256_b8_g16_vl64_pgate50_armB_running_ipc_pr9.png
# Require plot_style=opcode_lanes_ipc (reject IPC-only).
```

**Historical:** prior `era=per-b-outer` PR9 (incl. armA / pgate95) under same `plots/` — retained; **not** live SoT for global-gate / rib KPI (Appendix H).
## 8. Per-segment opcode mix + stage mid-IPC

**SoftLib≻SIMD / STAGE_TAXONOMY reduce map:** **N/A** (CF policy).  
**Live mix (both eras):** **OK (S3 tabulated)** — body call_counts from latest `instr_exe.csv` (exclude VF/VF_SIMT wrappers). Full tables: `plots/cf7d_dual_era_S3_opcode_fragment.md`. Stage mid-IPC sequential walls **N/A** (fused control-flow nest; no SoftLib stage taxonomy split claimed). Headline dump IPC proxy = `n_issues/vf_cyc` from PR9 JSON (§7).

**Pin cells (key hits; totals = body_calls):**

| era | G | gate | arm | body_calls | key hits |
|-----|--:|------|-----|----------:|----------|
| hoisted | 16 | off | B | 2665 | VSEL:1024 VCMP_GT:513 PAND:512 VMUL:512 |
| hoisted | 16 | off | C | 111 | SJUMPI/SCBZI/SAND; **no** VMUL/PAND/VSEL |
| hoisted | 16 | on | B | 2665 | identical to B off |
| hoisted | 16 | on | C | 1674 | VCMP/VMUL/VSEL:512 each; **PAND=0**; SJUMPI:1 |
| hoisted | 64 | off | B | 10345 | VSEL:4096 PAND:2048 VMUL:2048 VCMP:2049 |
| hoisted | 64 | off | C | 111 | same skip as G16 off |
| hoisted | 64 | on | C | 6282 | VCMP/VMUL/VSEL:2048; PAND=0 |
| rib | 16 | off | B | 2693 | PAND path (+rib VLDI/VSTI) |
| rib | 16 | off | C | 152 | SJUMPI:8 VCADD:9; no fat VMUL |
| rib | 16 | on | C | 1740 | VMUL/VSEL:512; VCADD:9 |
| rib | 64 | off | C | 152 | same body as G16 off (SUMMARY VF 1386→3234 — open) |
| rib | 64 | on | C | 6348 | VMUL/VSEL:2048; VCADD:9 |

**Attribution (no invent):** B pays G-fat PAND/VMUL regardless of gate; C gate_off skips that nest (tiny body); C gate_on pays VMUL/VSEL without PAND → explains C≻B VF. Rib multiplies SJUMPI/VCADD vs hoisted on gate_off but keeps skip of fat VMUL.

**Historical A≡B mix** (`era=per-b-outer`): Appendix H / `cf7d_OPCODE_MIX.md`.
## 9. ISA mix SIMT vs SIMD (per N×4B → el/cyc bridge)

**SIMT vs SIMD both-frontend:** **MISSING / N/A** (no Simt/SoftLib this cut).

**Intra-*d B vs C (N=32, E=256, units=8):** body_calls from §8; IPC proxy = body_calls / SUMMARY_VF. Full helper: `plots/cf7d_isa_mix_perN.json`.

| era | G | gate | arm | VF | body | IPC_body | ops/N (body/8) | pred_el/cyc≈IPC×(32/ops_N) | meas el/cyc=E/VF |
|-----|--:|------|-----|---:|-----:|---------:|---------------:|---------------------------:|-----------------:|
| hoisted | 16 | on | B | 2237 | 2665 | 1.191 | 333.1 | 0.114 | 0.114 |
| hoisted | 16 | on | C | 1283 | 1674 | 1.305 | 209.2 | 0.200 | 0.199 |
| hoisted | 16 | off | B | 2237 | 2665 | 1.191 | 333.1 | 0.114 | 0.114 |
| hoisted | 16 | off | C | 821 | 111 | 0.135 | 13.9 | 0.311 | 0.312 |
| hoisted | 64 | on | B | 8827 | 10345 | 1.172 | 1293.1 | 0.029 | 0.029 |
| hoisted | 64 | on | C | 5800 | 6282 | 1.083 | 785.2 | 0.044 | 0.044 |
| rib | 16 | on | B | 2118 | 2693 | 1.271 | 336.6 | 0.121 | 0.121 |
| rib | 16 | on | C | 1609 | 1740 | 1.081 | 217.5 | 0.159 | 0.159 |
| rib | 64 | off | C | 3234 | 152 | 0.047 | 19.0 | 0.079 | 0.079 |

Bridge `|pred/meas−1|` ≤ ~0.01 on pin cells above (label: IPC=body/VF). **Attribution:** C wins mainly by **lower ops/N** (skip PAND nest / skip entire G body on gate_off), not by exotic IPC alone; gate_on C still drops PAND vs B.

**Historical A vs C bridge** (`era=per-b-outer`): Appendix H.
## 10. Conclusion

| Arm | Role | This cut (measured) |
|-----|------|---------------------|
| **SIMD PTO-DSL B preg+PAND (hoisted)** | always-enter + vand | VF flat vs gate; scales with G (68→2237→8827) |
| **SIMD PTO-DSL C pto.if_ (hoisted)** | in-VF scalar skip | **C≻B** for all G∈{16,64} on/off; G=0 tie; A1 crossover **G=16** (gate_on) |
| **SIMD PTO-DSL B (rib)** | reduce inside each b | ≈B-hoisted at G≥16; +7 VF at G0 |
| **SIMD PTO-DSL C (rib)** | per-b reduce + if_ | **Still C≻B** at G≥16; C/B worse than hoisted on gate_off → hoist artifact **reduced not removed** |
| **SIMT Ascend default** | — | **MISSING** |
| **SIMT SoftLib** | — | **MISSING** |

**Headline (do claim):** On this dual-era grid, **C (`pto.if_`) beats B (vand/PAND)** once G>0 under A1's 0.90 bar for **both** hoisted and reduce-in-b eras; B pays G-fat regardless of gate; C gate_off keeps tiny body_instr.

**Do not claim:** SoftLib≻SIMD; suite / MB DONE; PR9/IPC causation; that rib **eliminates** C's win; reuse of Appendix H per-b VF as live SoT.

**Still open:** optional G∈{8,32}/pgate95 PR9 fill-in; rib C gate_off VF growth pipe/IQ cause; Simt CF7 (*v); SoftLib; **merge needs Lok yes** (no suite DONE claim beyond *d B/C measured+PR9/S3).

## 11. Trace pointer

`benches/cf7d/EXPLORATION_TRACE.md` — global-gate redesign + smoke + doc land (2026-11-03). Prior dump-sync/PR9 steps retained in TRACE timeline.

---

## After RESULT — COMPARE

### KPI COMPARE harvest

| vs | Status |
|----|--------|
| Prior dig / SoftLib≻SIMD el/cyc | **KPI N/A** — CF policy ST |
| `era=per-b-outer` first-cut VF tables | **STALE** — different gold (per-b outer + host n_trips C); kept in Appendix H; **not** folded into live §4 |
| Live hoisted B vs C | **MEASURED** — §4.1; C≻B for G∈{16,64}; crossover G=16 (gate_on) |
| Live rib B vs C | **MEASURED** — §4.2; C≻B remains; hoist artifact reduced (§4.4) |

### Motif-seed Coverage COMPARE

| Motif / recipe tag | Coverage | Note |
|--------------------|:--------:|------|
| `outer_inner_if` nest | ✓ | design + kernel |
| `global_gate_sum` | ✓ | gold + B/C reduce |
| `armB_preg_pand` / vand | ✓ | KPI measured both eras |
| `armC_pto_if_` in-VF | ✓ | KPI measured; **no** n_trips; gate_off body skip |
| `reduce_in_b` / Aims A5 | ✓ | rib oneshot 12/12 PASS |
| `armA_pure_slack` | ✗ dropped | Lok 2026-11-03; historical only |
| `R*_x_pgate` per-b first-cut | ✓ historical | Appendix H STALE |
| `crossover_VF_0.90` live | ✓ | G=16 both eras (gate_on); G=0 tie |
| PR9 `opcode_lanes_ipc` live | ✓ 24/24 key cells | `plots/*_running_ipc_pr9.png` + harvest JSON |
| SimtVF CF7 (*v) | ✗ / stop | Aims A4 before *v |
| SoftLib SIMT CF7 | ✗ / stop | not Aims |
| SoftLib≻SIMD both-frontend | N/A | no both-frontend claim |

---

## TODO / open clarification

1. **PR9 + S3 mix:** **DONE** for G∈{0,16,64}×gate_off/on×B/C×both eras (24 cells). Rib C gate_off VF growth G16→G64 with flat body=152 still needs pipe/IQ dig (not invented).
2. Optional denser grid: G∈{8,32}, pgate95 — not required for current A1 call.
3. **t821u:** Appendix H retains all prior 3-arm / per-b / first-cut VF + el/cyc + crossover + opcode A≡B tables — do not delete. Marked **STALE**; live SoT is §4 dual-era global-gate / rib (proved by SUMMARY_dual_era_oneshot.tsv).
4. Prior ANALYSIS claimed arm-C = host `n_trips` and OPPROF zero `SIMT_BRANCH` — **contradicts** new C intent (`pto.if_`). Live body_instr gate_off skip supports if_; full S3 branch family still pending harvest (Appendix H keeps historical OPPROF).
5. Rib C gate_off VF growth G16→G64 with flat body=152 — open (needs PR9/IQ); do not invent pipe cause.

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
- [x] SEMANTICS_GATE one-liner per arm (**PASS** — S1–S4; S3 from instr_exe; see §4.6)
- [x] t821u: prior 3-arm / per-b / crossover tables retained in Appendix H as **STALE**
- [x] No SoftLib≻SIMD / no invented live VF
- [x] TRACE linked
- [x] Kernel check-in `benches/cf7d/cf7d_outer_inner_if.py` (+ `smoke_thin.py`)
- [x] Live RESULT dual-era SUMMARY — **24/24 PASS** (§4)
- [x] PR9 / opcode mix — **OK** 24/24 key cells (§7–§8); ISA-mix B vs C bridge (§9)
- [ ] Suite / MB DONE — **forbidden** (**PARTIAL**)
- [ ] Merge — **needs Lok yes**
