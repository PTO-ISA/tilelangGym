# SoftLib SIMD absmax VAND — vs rooftop (not UB-bound)

**Date:** 2026-10-22 Asia/Hong_Kong  
**Arm:** SoftLib dual64 VAND clear-sign absmax · E=8192 fp32 · VF **261** · whole **31.39** el/cyc  
**Style:** W=20 mid IPC **2.50** / mid input **41.6** el/cyc (same SoT as SIMD running IPC).

## Claim

This arm is **compute / EXU-bound**, **not UB-bound**.

## Roofs (per dual64 row = 128 el)

| resource | cyc/row | theory el/cyc | notes |
|---|---:|---:|---|
| **UB LD** (dual VLD) | **1** | **128** | A-only SoftLib LDS.32 peer ~101 B ≈ 25 el is SIMT; SIMD dual VLD class is far above measured |
| **ST** (2× VST @ 1/cyc) | **2** | **64** | tiny reduced Y; ST ties SLIDE on *reduce-max*, not the binder here |
| **EXU** | **3** | **~42.7** | `ceil(n_non/2)+n_SLIDE = ceil(2/2)+2` — 2×VAND (ADD, dual) + 2×VCGMAX (EXQ0-only) |

**Binding resource = EXU @ ~42.7 el/cyc.** UB LD roof (128) and ST roof (64) both sit **above** EXU → not memory-bound.

## % of rooftop

| metric | value | vs EXU ~42.7 | vs UB 128 |
|---|---:|---:|---:|
| **mid input el/cyc** | **41.6** | **~97.4%** | ~32.5% |
| whole-kernel el/cyc | **31.39** | **~73.5%** | ~24.5% |
| mid IPC | **2.50** | — (issue mix, not el roof) | — |

Whole-kernel sits below mid because of head/tail; mid already hugs the EXU roof.

## Why not UB

1. Measured mid **41.6** ≪ UB theory **128** (and ≪ even a conservative ~64–101 load-only class).
2. Mix is `VLD + VAND + VCGMAX + VST` ×128 — VCGMAX on EXQ0 + VAND stealing ~½ Q0 → **3 EXU cyc/row**, not load issue.
3. Contrast: SoftLib **reduce-max** (no VAND) mid ~**51** / EXU roof **64** (~80%); adding VAND drops EXU roof **64→42.7** and mid **51→41.6** — classic compute step-down, not a BW cliff.

## Theory compute el/cyc (summary)

```
el/cyc_theory = 128 / max(LD=1, EXU=3, ST=2) = 128/3 ≈ 42.67
```

Measured: mid **41.6 ≈ 97%** of theory; whole **31.39 ≈ 73%**.

## Artifacts
- `SIMD_RUNNING_IPC.md`, `ipc/ABSMAX_VAND_MIDPOINT_IPC.md`
- Plot: `plots/sv5_simd_dual64_absmax_vand_rolling20_ipc.png`
- Source: `sources/sv5_ccs_softlib_absmax_vand.asc`
