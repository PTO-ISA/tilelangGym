# Corrected rooftops — SV1d (SIMD) + SimtVF SV1

**Date:** 2026-10-15 HKT · Lok corrections: UB BW / issue mix / **Simt roof = 16**  
**Traces:** `data/sv1d_e16384_vl64/`, `data/sv1tt_e16384_tile2048_t128/`, `data/ilp_keeps/sv1t_e16384_t128_ilp16/`

**Correction (SimtVF mix-cap):** Body4 mix-cap IPC is **~4** (four wrap schedulers), **not** 1.  
**Correction (Simt roof, Lok 2026-10-15):** **1R+1W @ 128B each → 16 el/cyc**, **not** 32. Prior “32” treated `.64` as if R port were 256B; Lok: **1R = 128B**. Pack on the path remains **LDS/STS `.64` only** (no `.128`).

---

## 1. SIMD / SV1d — corrected mem rooftop

**Wrong prior roof:** 64 el/cyc (α=1, one VL=64 chunk per cycle).

**Correct (Lok):** UB R+W BW = **512 B/cyc**. VADD pattern **2R+1W**:

| Quantity | Value |
|----------|------:|
| VL | 64 FP32 |
| Bytes / chunk | 3 × 64 × 4 = **768 B** |
| Mem-bound cyc / chunk | 768 / 512 = **1.5** |
| **Rooftop el/cyc** | 64 / 1.5 = **42.667** |

### Measured vs corrected roof (E=16384)

| Denominator | Cycles | el/cyc | **% of 42.67** |
|-------------|-------:|-------:|---------------:|
| Body 1st VLDI → last VSTI **complete** | 482.2 | **33.98** | **79.6%** |
| Body 1st VLDI → last VSTI **issue** | 473.2 | 34.62 | 81.1% |
| VF wrapper (SUMMARY) | 520 | 31.51 | 73.8% |
| α≈1.875 asymptote (size ladder) | — | ~34.1 | **80.0%** |

### Mid-body 30-cyc windows → add-IPC (balanced 2LD+EX+1ST)

| Window | vec IPC | add-IPC (=IPC/4) | el/cyc (=add×64) | **% of 42.67** |
|--------|--------:|-----------------:|-----------------:|---------------:|
| Mid **median** | **2.13** | **0.533** | **34.13** | **80.0%** |
| Mid **peak** (central 50%) | **2.27** | **0.567** | **36.27** | **85.0%** |
| Mid p95 el/cyc (VSTI credit) | — | — | 36.27 | 85.0% |

**Verdict:** ~34 el/cyc body is **~80% of the true 42.7 roof**; mid peak **~85%**.

---

## 2. SimtVF — roof **16** (1R+1W @ 128B), pack **.64 only**

> **Pack remark:** Body4 LDS/STS shares are **`SIMT_LDS.64` / `SIMT_STS.64` only** (float2). **`.128` was never used.**  
> **Roof remark (Lok):** UB **1R+1W @ 128B each → 16 el/cyc**. Do **not** use 32.

### Body instruction counts (keep-trace, TILE2048@T128) — pack **.64 only**

| Class | Count | Notes |
|-------|------:|-------|
| **SIMT_LDS** (LD) | **512** | **`.64` only** (float2) — **not `.128`** |
| **SIMT_STS** (ST) | **256** | **`.64` only**; 256×64 el = 16384 |
| **SIMT_IADD + IADD_I** | **580** | 32 + 548 (addr tax) |
| **SIMT_FADD** | **512** | 512×32 el = 16384 |
| **Body4 total** | **1860** | LD+ST+IADD+FADD |

### Mix ratios (Body4 = 1860, pack **.64**)

| Class | Share |
|-------|------:|
| LDS (**.64**) | **27.5%** |
| STS (**.64**) | **13.8%** |
| IADD | **31.2%** |
| **FADD** | **27.5%** |

Per FADD quantum (32 el): **1.0 LDS + 0.5 STS + 1.13 IADD + 1.0 FADD ≈ 3.63 instr**.

### Mix-cap IPC: **~4** (wrap scheduler width)

| Model | IPC_cap | Notes |
|-------|--------:|-------|
| One issue total (old) | **1** | **Wrong** — contradicts 4-wrap multi-issue |
| **4 wrap schedulers (Lok)** | **4** | Max same-cycle Body4 issues observed = 3–4 |

FADD pipe / mem roof still clamps throughput to **16 el/cyc** under Lok’s 128B R+W model.

### Throughput vs **16** roof (pack .64) — TILE2048@T128

| Metric | Value | **% of 16** |
|--------|------:|------------:|
| el/cyc VF (SUMMARY) | **3.22** | **20.1%** |
| el/cyc body (1st→last FADD) | 3.30 | 20.6% |
| Mid-body median **Body4 IPC** | **0.367** | peak **0.667** |
| Mid-body median FADD-IPC | **0.100** | → 3.20 el/cyc (**20.0%**) |

Measured Body4 IPC ~0.37 / cap 4 ≈ **~9% of issue width** (stall).

---

## 3. SimtVF T-dim bars (E=16K, pack .64) — vs roof **16**

> **TILE clarification:** T=128/256/512 bars are **untiled TILE=E=16384**. **NOT tile 8192.** TILE2048@T128 is a separate tiled ref. ILP16@T128 is software-ILP reschedule at T=128.

| Case | T | **TILE** | VF_cyc | el/cyc | **% of 16** |
|------|--:|---------:|-------:|-------:|------------:|
| `sv1t_…_t128` untiled | 128 | **16384** | 5700 | **2.87** | **18.0%** |
| `sv1t_…_t256` untiled | 256 | **16384** | 4864 | **3.37** | **21.1%** |
| `sv1t_…_t512` untiled | 512 | **16384** | 4415 | **3.71** | **23.2%** |
| TILE2048@T128 (ref) | 128 | **2048** | 5088 | **3.22** | **20.1%** |
| **ILP16@T128 .64** | 128 | **16384** | **4015** | **4.08** | **25.5%** |

T↑ helps a little; **ILP16@T128 beats T2048** (4.08 vs ~3.83) still on `.64` only. All remain **≲26% of roof 16**.

Plot: `plots/sv1_rooftop_corrected_summary.png` · TSV: `data/SUMMARY_sv1_simt_T_dim_rooftop.tsv`.

---

## 4. ILP16@T128 = 4.08 — overhead breakdown (keep-trace)

**Keep:** `data/ilp_keeps/sv1t_e16384_t128_ilp16/` · detail: `NOTE_sv1_simt_ilp64_T128.md` · JSON: `OVERHEAD_ilp16.json`

| Piece | Cycles | % of VF | Role |
|-------|-------:|--------:|------|
| **VF prologue** (→ 1st Body4) | **30.6** | **0.8%** | S2R / SHFI / LEA |
| **Body** (1st Body4 → last complete) | **3712.7** | **92.5%** | LDS+STS+IADD+FADD |
| **VF epilogue** (last Body4 → VF end) | **271.7** | **6.8%** | SET_FLAG + MOV |
| **VF total** | **4015** | 100% | SUMMARY denominator |

| Rate | el/cyc | **% of roof 16** |
|------|-------:|-----------------:|
| **VF (SUMMARY) = 4.08** | **4.081** | **25.5%** |
| Body complete | 4.413 | 27.6% |
| **Mid-body avg (FADD×32 / mid-span)** | **4.507** | **28.2%** |
| Mid Body4 IPC median / peak | 0.467 / 1.867 | — |
| Mid idle (issue width=0) | **65.1%** of mid cycles | — |

**Kernel launch:** keep-trace is **VF-scoped**. Host launch is **not** inside `VF_cyc`. **4.08 already excludes launch.**

**Where the gap is:**

| Gap | Δ el/cyc | Cause |
|-----|--------:|-------|
| Roof 16 → mid ~4.51 | **≈11.5** | **Mid-body stall** (65% idle, Body4 IPC≪4, IADD tax) |
| Mid 4.51 → VF 4.08 | **≈0.43** | Prologue+epilogue (~7.5% of VF) + mild head/tail |

**Peak is not sustained:** mid Body4 peak 1.87 and burst FADD windows (30-cyc peak ~40 el/cyc) are **ILP FADD batches** after LDS hoists — only **~3%** of mid windows near 85% of Body4 peak; FADD-rate median in 30/60/120-cyc windows is **0** (bursty). **Sustained mid ≈ 4.5 el/cyc**, not peak.

Waterfall plot: `plots/sv1_ilp16_overhead_waterfall.png`.

---

## 5. Side-by-side

| | SV1d SIMD E=16K | SimtVF (best ILP16@T128) |
|--|----------------:|-------------------------:|
| Corrected roof | **42.67** el/cyc | **16** el/cyc (1R+1W@128B) |
| Body / VF el/cyc | **34.0** / 31.5 | mid **~4.5** / VF **4.08** |
| % of roof | **~80%** (mid peak ~85%) | **~25–28%** |
| Bound | near UB 2R+1W | **mid-body stall vs issue width**; pack **.64**; no launch tax in VF |

---

## Artifacts

- This note: `data/NOTE_sv1_rooftop_corrected.md`
- ILP note + overhead: `NOTE_sv1_simt_ilp64_T128.md`, `OVERHEAD_ilp16.json`
- Plots: `plots/sv1_rooftop_corrected_summary.png`, `plots/sv1_ilp16_overhead_waterfall.png`
- Body-mix IPC (aligned): `plots/sv1d_e16k_ex_ipc_30cyc.png`, `plots/sv1_simt_e16k_body4_ex_ipc_30cyc.png`
- TSV: `data/SUMMARY_sv1_simt_T_dim_rooftop.tsv`
