# SV6 analysis — mid-block reduce G=32 (inner 1×G + outer N×1)

**Date:** 2026-10-29 Asia/Hong_Kong · ST-spine · tip refresh **t725u** (outer VMAX t720u + Nx1 SIMT t708u) · prior 1×G/Gmap close t705/t706  
**Format:** ST spine · `tilelanggym-st-report` + softlib-simt-schedule · st-semantics-gate · vmi-gmap-close · simd-per-g-mapping  
**Metric:** el/cyc = E/VF · E=8192 · **era=fp32-absmax** · Case-1 · no recip.  
**SOC:** Ascend950PR_9599 · cann_92b2.

**ST map:** SV6 = gold **G=32**. Inner and outer axes kept separate (§21/§22).

### Headline — `reduce_axis=inner` (1×G)

| # | Path | Best | Era |
|---|------|------|-----|
| 1 | **SIMD PTO-DSL / VMI** | VL256/NG8 · i32-VAND + **VCGMAX** · dig **R32×C256** · **22.14** (VF 370) | fp32-absmax |
| 2 | **SIMT-ASC stock** | default `__simt_vf__` @ T128 · wc LDS.32 · **1.57** (VF 5225) | fp32-absmax |
| 3 | **SIMT SoftLib** | `axis=1×G` wc loadall+absall **dual-drain** T512 · **8.22** | fp32-absmax |

> Footnote SoftLib CCE half-VL VCMAX **21.28** (−4% vs VMI) — off main (t706u).

### Headline — `reduce_axis=outer` (N×1)

| # | Path | Best | Era |
|---|------|------|-----|
| 1 | **SIMD VMI** | pure RV_VMAX fold + i32-VAND · VL256 · **45.77** (VF 179) · `outer_g32_vmax_vand_fold_vl256` | t720u |
| 2 | **SIMT-ASC stock** | nx1-col fused · T=256 · **4.74** | t708u |
| 3 | **SIMT SoftLib** | nx1-col **membar** · T=256 · **9.58** | t708u |

---

## 1. Aims

| Id | Aim |
|----|-----|
| **A1** | Case-1 absmax · gold G=32 · E=8192 · both reduce axes |
| **A2** | VMI Gmap DONE or explicit stop; outer pure VMAX measured (§22) |
| **A3** | SoftLib SIMT **axis=1×G** / **axis=Nx1** labeled; §20 COMPLETE vs residual |
| **A4** | SoftLib CCE off main when VMI ≥ SoftLib (t706u) |
| **A5** | SoftLib SIMT measured both axes |
| **A6** | SIMD schedule/instr · SIMT layout · running IPC (W=20 mid or MISSING) |

---

## 2. Test-case design

| Axis | Choice |
|------|--------|
| Gold G | 32 · E=8192 |
| VMI dig shape (inner) | **R32×C256** VL=256 NG=8 |
| VMI live peer (inner) | R64×C128 VL=256 NG=8 — does **not** beat dig |
| VMI outer | VL256 VMAX fold · i32-VAND |
| SoftLib SIMT inner | wc · LDS.32 · dual-drain · T512 |
| SoftLib SIMT outer | nx1-col · T=`n_cols`=256 · membar best |
| SoftLib CCE | footnote only |

---

## 3. Layout and schedule

### 3.1 VMI inner

- **Schedule:** `vcmax(..., group=8)` on VL256 → **VCGMAX** lowering · abs i32-VAND · storeB compact dig shape R32×C256.
- **Instr (live peer mix):** VCGMAX:32 · VAND:128 · VMAX:96 · VDINTLV:64 · VLD:64 · VST:32.
- **SoT note (t706u):** VMAX→VCGMAX + split/unroll ILP **did not beat** 22.14 (split-ILP evidence SoftLib G128 `split_vmax_ilp` 13.54 ≪ SoT-class).

### 3.2 VMI outer

- **Schedule:** VL256 column tiles · G=32 row fold with **RV_VMAX** only · i32-VAND · VCGMAX=VCMAX=0.
- **Dump:** RV_VMAX=124 · RV_VAND=128 · VF 179 · el/cyc **45.77**.

### 3.3 SoftLib SIMT inner `axis=1×G`

- **Layout:** warp-coop **full-warp** (G=32) · contig LDS.32 · REDUX · loadall+absall **dual-drain** at T512.
- Stock T128 = **1.57**; dual-drain T512 = **8.22** (serial re-emit same pack 7.48 loses to dual).

### 3.4 SoftLib SIMT outer `axis=Nx1`

- **Layout:** per-thread column `a[r]=A0[r*n_cols+tid]` · T=`n_cols`=256 · no warp redux on SoT.
- Stock fused **4.74** · best membar **9.58**.

---

## 4. SIMD PTO-DSL / VMI Gmap

| Layer | Inner status |
|-------|--------------|
| Functional | PASS dig + live |
| Tried | dig storeB; live vand+vabs R64×C128 |
| Best-sweep | **DONE** — dig SoT **22.14**; live does not supersede |
| SoftLib CCE | footnote 21.28 |

| Outer SoT | VF | el/cyc | gate |
|-----------|---:|-------:|------|
| `outer_g32_vmax_vand_fold_vl256` | 179 | **45.77** | PASS §22 |

Close note: [`../sv5/reports/VMI_GMAP_ABSMAX_CLOSE_t705u.md`](../sv5/reports/VMI_GMAP_ABSMAX_CLOSE_t705u.md) · outer [`../sv5/reports/VMI_ABSMAX_OUTER_VMAX_t720u.md`](../sv5/reports/VMI_ABSMAX_OUTER_VMAX_t720u.md).

**SEMANTICS_GATE (inner VMI):** PASS — VCGMAX primary · era=fp32-absmax.  
**SEMANTICS_GATE (outer VMI):** PASS — pure VMAX · VCG/VCMAX=0.

---

## 5. SIMT-ASC stock

| Axis | el/cyc | stock T | tag |
|------|-------:|--------:|-----|
| inner | **1.57** | 128 | `wc_abs_lds32_redux_g32_t128_e8192` |
| outer | **4.74** | 256 | `nx1_col_lds32_fused_g32_t256_e8192` |

---

## 6. SoftLib SIMT (A5)

| Axis | motif | T | el/cyc | end |
|------|-------|--:|-------:|-----|
| inner 1×G | wc dual-drain | **512** | **8.22** | **DONE** |
| inner stock | wc bare | 128 | 1.57 | loses (latency hide) |
| outer Nx1 | nx1-col membar | **256** | **9.58** | **DONE** |
| outer stock | nx1-col fused | 256 | 4.74 | |

Residual: misched=0 @ SoT T (inner). **SEMANTICS_GATE PASS** both axes · no unlabeled Nx1↔1×G merge.

Harness checklist: [`../sv5/reports/SOFTLIB_SIMT_1xG_HARNESS_GATE_t705u.md`](../sv5/reports/SOFTLIB_SIMT_1xG_HARNESS_GATE_t705u.md).

---

## 7. Running IPC (W=20 mid)

| Arm | Axis | mid IPC W20 | note |
|-----|------|------------:|------|
| VMI outer SoT | outer | **3.00** | mean body 2.76 · t725u |
| VMI inner live `g32_vcg_vl256_ng8_vand` | inner | **1.35** | mean 1.26 · dig SoT same recipe class |
| SoftLib SIMT inner stock T128 | inner | **0.45** | mean 0.32 |
| SoftLib SIMT inner dual-drain best | inner | **2.10** | mean 1.61 |
| SoftLib SIMT outer stock fused | outer | **1.10** | mean 0.64 |
| SoftLib SIMT outer membar best | outer | **0.90** | mean 0.96 · window-sensitive |
| Full matrix | — | — | [`../../SV567_IPC_HARVEST_t725u.md`](../../SV567_IPC_HARVEST_t725u.md) |

---

## 8. Conclusion

### Inner 1×G

| Arm | el/cyc |
|-----|-------:|
| VMI dig | **22.14** |
| SoftLib SIMT 1×G best | **8.22** |
| SIMT-ASC stock | **1.57** |

### Outer N×1

| Arm | el/cyc |
|-----|-------:|
| VMI outer VMAX | **45.77** |
| SoftLib SIMT nx1 best | **9.58** |
| SIMT-ASC stock | **4.74** |

> SoftLib CCE half-VL VCMAX **21.28** — footnote only (t706u).

**COMPARE:** SoftLib SIMT tip = dual-drain **8.22** (not stock 1.57). Outer VMI **45.77** replaces wrong-arm inner reuse. VMAX→VCG + split ILP did not beat inner SoT.

**PR:** https://github.com/PTO-ISA/tilelangGym/pull/8
