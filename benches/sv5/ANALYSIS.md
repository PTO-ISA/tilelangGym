# SV5 SoftLib SIMD absmax — VAND clear-sign ANALYSIS

**Date:** 2026-10-22 Asia/Hong_Kong  
**Abs impl (primary):** `VAND(~0x80000000)` clear-sign — **not** legacy `VMUL(-1)+VMAX`.  
**Shape:** E=8192 fp32 · SoftLib `__simd_vf__` dual64 · VCGMAX G=8 + pair-max → gold G=16.

## Headline

| arm | VF | **el/cyc** | mid IPC (W=20) | mid input el/cyc | % EXU roof |
|---|---:|---:|---:|---:|---:|
| SoftLib reduce-max | 234 | **35.01** | ~2.40 | **51.2** | ~80% of **64** |
| **SoftLib absmax VAND** | 261 | **31.39** | **2.50** | **41.6** | **~97% of ~42.7** |
| SoftLib absmax VMUL+VMAX *(legacy)* | 417 | 19.65 | ~1.95 | 25.6 | ~80% of **32** |

Source: `softlib/sv5_ccs_softlib_absmax_vand.asc` · gold maxabs = 0.

## Avg-window / running IPC (SIMD)

Style: W=20, `(t−W, t]` — same as SoftLib mid-IPC SoT.

**VAND primary:** mid IPC **2.50**, mid input **41.6** el/cyc, mean body IPC ~2.31–2.41, mean VLD-win el_L **43.3**, peak el_L **83.2**. Mid VAND/cyc ~0.60 · mid VCGMAX/cyc ~0.65.

Plot: `plots/sv5_simd_dual64_absmax_vand_rolling20_ipc.png`  
Detail: `ipc/ABSMAX_VAND_MIDPOINT_IPC.md`, `SIMD_RUNNING_IPC.md`.

## Rooftop — **not UB-bound**

Per dual64 row (128 el):

| resource | cyc/row | theory el/cyc |
|---|---:|---:|
| UB LD (dual VLD) | 1 | **128** |
| ST (2× VST) | 2 | **64** |
| **EXU** = `ceil(2/2)+2` (2×VAND + 2×VCGMAX) | **3** | **~42.7** |

```
el/cyc_theory = 128 / max(LD=1, EXU=3, ST=2) = 128/3 ≈ 42.67
```

| metric | value | vs EXU ~42.7 | vs UB 128 |
|---|---:|---:|---:|
| mid input el/cyc | **41.6** | **~97%** | ~32% |
| whole-kernel el/cyc | **31.39** | **~73%** | ~25% |

Binding resource is **EXU**, not UB. Adding VAND drops EXU roof 64→42.7 and mid 51→41.6 (compute step-down). See `ROOFTOP_NOT_UB.md`.

## Related notes

- SoftLib SIMD **fused** VAND remains the SIMD SoT (~31); membar SoftLib (~12.7) is stage-diagnostic only.
- Simt contig+REDUX upgrade band is separate (membar 6.35 / aicore-misched=0 clean stages 5.08) — not this ANALYSIS’s primary.
