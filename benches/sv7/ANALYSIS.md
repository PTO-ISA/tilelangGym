# SV7 analysis — large-block reduce (G=64 / G=128)

**Date:** 2026-10-27 Asia/Hong_Kong · **t705u** SoftLib SIMT 1×G + VMI Gmap live close · **t706u:** SoftLib CCE off main / footnote peer · Ascend950PR_9599 · cann_92b2  
**Format:** ST spine · st-report + softlib-simt-schedule · st-semantics-gate · vmi-gmap-close  
**Metric:** el/cyc = E/VF · E=8192 · **era=fp32-absmax** · Case-1 · no recip.

**Conclusion arms (per gold G — SoftLib CCE not a required arm):**

| G | VMI SoT | SoftLib SIMT 1×G best | **SIMT-ASC stock** |
|--:|--------:|----------------------:|--------------------|
| 64 | **30.68** VABS (VF 267) · VAND 30.57 | **5.54** wc-multi T512 | **2.39** wc-multi per=2 T128 (VF 3434) |
| 128 | **32.25** VMAX→VCMAX (VF 254) | **3.72** wc-multi T256 | **1.67** wc-multi per=4 T64 (VF 4916) |

> **SIMT-ASC stock** = SoftLib `__simt_vf__` **default lowering** at legacy stock T (same numbers as SoftLib SIMT stock) — not a separate unmeasured Ascend-only arm.

> Footnote SoftLib CCE peers (not required 4th arm): G64 **30.68** tied with VMI; G128 **32.38** (+0.4% vs VMI 32.25) — footnote peer only (t706u).

---

## 1. Aims

| Id | Aim |
|----|-----|
| **A1** | Case-1 absmax · gold G∈{64,128} · E=8192 |
| **A2** | VMI Gmap DONE (was OPEN dig) |
| **A3** | SoftLib SIMT **axis=1×G** · wc-multi · §20/§21 COMPLETE vs residual |
| **A4** | SoftLib CCE **off main table** when ≈ VMI (tie / ≤0.5% peer → footnote only; t706u) |
| **A5** | SoftLib SIMT measured |

---

## 2. Test-case design

| Axis | Choice |
|------|--------|
| G64 VMI | whole-VL **VCMAX** VL64 · VAND + VABS_FP |
| G128 VMI | **VMAX-fold→VCMAX** and whole-VL VCMAX VL128 · VAND |
| SoftLib SIMT G64 | wc-multi **per=2** · LDS.32 · T512 |
| SoftLib SIMT G128 | wc-multi **per=4** · LDS.32 · T256 |
| WC-only (per=1) | **no path** for G>32 — need in-warp fold (t704u) |
| SoftLib CCE | footnote peer only (tied @G64; +0.4% @G128) |

**Matrix (spine — SoftLib CCE not required):**

| G | schedule | el/cyc | VF | end |
|--:|----------|-------:|---:|-----|
| 64 | VMI VCMAX VABS | **30.68** | 267 | DONE |
| 64 | SoftLib SIMT wc-multi T512 | **5.54** | 1478 | DONE |
| 64 | SoftLib SIMT stock T128 | 2.39 | 3434 | loses |
| 128 | VMI vmaxfold / vcmax_vl128 | **32.25** | 254 | DONE |
| 128 | SoftLib SIMT wc-multi T256 | **3.72** | 2201 | DONE |
| 128 | SoftLib SIMT stock T64 | 1.67 | 4916 | loses |

Live RESULT: [`reports/RESULT_absmax_vmi.tsv`](reports/RESULT_absmax_vmi.tsv).

---

## 3. Layout and schedule

### 3.1 VMI G64
Whole-VL VCMAX · abs VAND or VABS · mix VCMAX:128 + ABS:128 + VLD/VST:128.

### 3.2 VMI G128
VMAX pre-fold + VCMAX · mix VCMAX:64 VMAX:64 VAND:128 VLD:128 VST:64 · both tags same VF 254.

### 3.3 SoftLib SIMT 1×G (wc-multi)
- G64: per_thread=2 elems inside gold G · T512 best among {128,256,512}.
- G128: per_thread=4 · T256 best among {64,128,256}; **T512 never measured** (residual).
- Why not small-T + WC-only: measured T-sweep + structural per≥2 — see t704u note / harness gate doc.

---

## 4. SIMD PTO-DSL / VMI Gmap

| G | Functional | Tried | Best-sweep | SoT |
|--:|------------|-------|------------|-----|
| 64 | PASS | vand + vabs VCMAX VL64 | **DONE** | **30.68** VABS |
| 128 | PASS | vmaxfold + vcmax_vl128 | **DONE** | **32.25** |

SEMANTICS_GATE: **PASS**. Close note: [`../sv5/reports/VMI_GMAP_ABSMAX_CLOSE_t705u.md`](../sv5/reports/VMI_GMAP_ABSMAX_CLOSE_t705u.md).

SoftLib CCE: **tied** @G64; **+0.4%** @G128 → **footnote peer only** (not a required conclusion arm; t706u).

---

## 5. SIMT-ASC stock

**SIMT-ASC stock** = SoftLib `__simt_vf__` default lowering @ legacy T (= SoftLib SIMT stock).

| G | el/cyc | stock T | note |
|--:|-------:|--------:|------|
| 64 | **2.39** | T128 | wc-multi per=2 · VF 3434 · `wc_abs_lds32_redux_g64_t128_e8192` |
| 128 | **1.67** | T64 | wc-multi per=4 · VF 4916 · `wc_abs_lds32_redux_g128_t64_e8192` |

---

## 6. SoftLib SIMT (A5)

| G | motif | axis | T | el/cyc | why stock loses | end |
|--:|-------|------|--:|-------:|-----------------|-----|
| 64 | wc-multi per2 | 1×G | **512** | **5.54** | T128=2.39 latency hide | **DONE** |
| 128 | wc-multi per4 | 1×G | **256** | **3.72** | T64=1.67 / T128=3.36 | **DONE** (T512 residual) |

wg ~0.98 both G — not SoT. **SEMANTICS_GATE PASS** · no Nx1.

Harness checklist: [`../sv5/reports/SOFTLIB_SIMT_1xG_HARNESS_GATE_t705u.md`](../sv5/reports/SOFTLIB_SIMT_1xG_HARNESS_GATE_t705u.md).

---

## 7. Conclusion

| G | VMI | SoftLib SIMT 1×G best | **SIMT-ASC stock** |
|--:|----:|----------------------:|--------------------:|
| 64 | **30.68** | **5.54** | **2.39** |
| 128 | **32.25** | **3.72** | **1.67** |

> SoftLib CCE footnote: G64 **30.68** (tie); G128 **32.38** (+0.4% peer) — not a required arm (t706u).

**COMPARE:** VMI G64/G128 were OPEN dig → live **DONE**. SoftLib SIMT main SoT = multi-warp wc-multi (not stock T128). SoftLib CCE ≈ VMI → off main headline.

**PR:** https://github.com/PTO-ISA/tilelangGym/pull/8
