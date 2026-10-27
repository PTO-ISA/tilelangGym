# SV6 analysis — mid-block reduce (G=32)

**Date:** 2026-10-27 Asia/Hong_Kong · **t705u** SoftLib SIMT 1×G + VMI Gmap close · **t706u:** SoftLib CCE off main + G32 ILP note · Ascend950PR_9599 · cann_92b2  
**Format:** ST spine · `tilelanggym-st-report` + softlib-simt-schedule · st-semantics-gate · vmi-gmap-close  
**Metric:** el/cyc = E/VF · E=8192 · **era=fp32-absmax** · Case-1 · no recip.

**Conclusion arms (headline — SoftLib CCE omitted when VMI ≈ SoftLib):**

| # | Path | Best | Era |
|---|------|------|-----|
| 1 | **SIMD PTO-DSL / VMI** | VL256/NG8 VAND dig **22.14** (R32×C256) | fp32-absmax |
| 2 | **SIMT-ASC stock** | **1.57** wc LDS.32 T128 (VF 5225) · default SoftLib `__simt_vf__` lowering | fp32-absmax |
| 3 | **SIMT SoftLib** | **axis=1×G** wc dual-drain T512 **8.22** | fp32-absmax |

> Footnote (not a required arm): SoftLib CCE half-VL VCMAX **21.28** — VMI already best (−4% vs 22.14); off main table per t706u.

---

## 1. Aims

| Id | Aim |
|----|-----|
| **A1** | Case-1 absmax · gold G=32 · E=8192 |
| **A2** | VMI Gmap DONE or explicit stop |
| **A3** | SoftLib SIMT **axis=1×G** §20/§21 COMPLETE vs residual |
| **A4** | SoftLib CCE **off main table** when VMI ≥ SoftLib (t706u); optional footnote only |
| **A5** | SoftLib SIMT measured |

---

## 2. Test-case design

| Axis | Choice |
|------|--------|
| Gold G | 32 · E=8192 |
| VMI dig shape | **R32×C256** VL=256 NG=8 |
| VMI live peer | R64×C128 VL=256 NG=8 (t699u) — does **not** beat dig |
| SoftLib SIMT | wc · LDS.32 · dual-drain · T512 |
| SoftLib CCE (footnote) | half-VL VCMAX — not a conclusion arm |

**Matrix (spine):**

| # | schedule | el/cyc | VF | note |
|---|----------|-------:|---:|------|
| 1 | VMI VL256/NG8 VAND dig | **22.14** | 370 | SoT |
| 1b | VMI live R64×C128 VAND / VABS | 21.90 / 22.02 | 374 / 372 | shape peer |
| 3 | SoftLib SIMT dual-drain T512 | **8.22** | ~997 | 1×G SoT |
| 3s | SoftLib SIMT wc T128 stock | 1.57 | 5225 | |

---

## 3. Layout and schedule

### 3.1 VMI
`vcmax(..., group=8)` on VL256 → VCGMAX lowering · VAND abs · storeB compact. Dig instr: VCGMAX:32 VAND:128 VMAX:96 VDINTLV:64 …

### 3.2 SoftLib SIMT 1×G
warp-coop full-warp · loadall+absall **dual-drain** T512 · LDS.32. Serial re-emit same pack **7.48** loses to dual **8.22**.

---

## 4. SIMD PTO-DSL / VMI Gmap

| Layer | Status |
|-------|--------|
| Functional | PASS dig + live |
| Tried | dig storeB; live vand+vabs R64×C128 |
| Best-sweep | **DONE** — dig SoT 22.14; live does not supersede (shape delta) |
| SoftLib CCE | footnote only — half-VL VCMAX 21.28 loses to VMI |

Dig-vs-live checklist + mix: [`reports/VMI_GMAP_ABSMAX_CLOSE_t705u.md`](../sv5/reports/VMI_GMAP_ABSMAX_CLOSE_t705u.md) · [`reports/RESULT_absmax_vmi.tsv`](reports/RESULT_absmax_vmi.tsv).

**G32 SoT note (t706u):** SoT remains **VL256/NG8 VCGMAX 22.14**. SoftLib half-VL VCMAX **21.28** is worse. **VMAX→VCGMAX + split/unroll ILP did not beat this** (split-ILP evidence: SoftLib G128 `split_vmax_ilp` **13.54** ≪ VMAX→VCMAX **32.38** — split/unroll does not recover toward SoT-class throughput).

SEMANTICS_GATE: **PASS** (VCGMAX primary · era=fp32-absmax).

---

## 5. SIMT-ASC stock

**SIMT-ASC stock** = SoftLib `__simt_vf__` default lowering @ legacy T (= SoftLib SIMT stock).

el/cyc **1.57** · wc LDS.32 **T128** · VF 5225 · tag `wc_abs_lds32_redux_g32_t128_e8192`.

---

## 6. SoftLib SIMT (A5)

| motif | axis | T | el/cyc | end |
|-------|------|--:|-------:|-----|
| wc dual-drain | **1×G** | **512** | **8.22** | **DONE** |
| wc bare T512 | 1×G | 512 | 5.72 | measured |
| wc stock | 1×G | 128 | 1.57 | stock loses (latency hide) |
| wg LDS.64 | 1×G | 128 | ~0.98 | not SoT |

**SEMANTICS_GATE: PASS** · no Nx1. Full §20 table: [`../sv5/reports/SOFTLIB_SIMT_1xG_HARNESS_GATE_t705u.md`](../sv5/reports/SOFTLIB_SIMT_1xG_HARNESS_GATE_t705u.md).

Residual: misched=0 @ SoT T.

---

## 7. Conclusion

| Arm | el/cyc |
|-----|-------:|
| VMI dig | **22.14** |
| SoftLib SIMT 1×G best | **8.22** |
| **SIMT-ASC stock** | **1.57** |

> SoftLib CCE half-VL VCMAX **21.28** — footnote only (t706u).

**COMPARE:** SoftLib SIMT tip = dual-drain **8.22** (not stock 1.57). VMI remains dig **22.14**; VMAX→VCGMAX + split/unroll ILP did not beat it.

**PR:** https://github.com/PTO-ISA/tilelangGym/pull/8
