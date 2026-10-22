# SV5 SoftLib SIMD absmax — VAND clear-sign ANALYSIS

**Date:** 2026-10-22 Asia/Hong_Kong  
**Abs impl (primary):** `VAND(~0x80000000)` clear-sign — **not** legacy `VMUL(-1)+VMAX`.  
**Shape:** E=8192 fp32 · SoftLib `__simd_vf__` dual64 · VCGMAX G=8 + pair-max → gold G=16.

## Headline

| arm | status | VF | **el/cyc** | mid IPC (W=20) | mid input el/cyc | % EXU roof |
|---|---|---:|---:|---:|---:|---:|
| SoftLib reduce-max | PASS | 234 | **35.01** | ~2.40 | **51.2** | ~80% of **64** |
| **SoftLib absmax VAND** | PASS | 261 | **31.39** | **2.50** | **41.6** | **~97% of ~42.7** |
| SoftLib absmax VMUL+VMAX *(legacy)* | PASS | 417 | 19.65 | ~1.95 | 25.6 | ~80% of **32** |
| VMI DSL VL=128 G=8 *(mask-fixed)* | PASS | 394 | **20.79** | — | — | n/a (≠ SoftLib schedule) |
| VMI DSL FA dual64 VL64×2 | PASS | 912 | **8.98** | — | — | — |

Source: `softlib/sv5_ccs_softlib_absmax_vand.asc` · gold maxabs = 0.

**SoftLib VAND − VL128 G=8 ≈ 31.39 − 20.79 ≈ 10.6 el/cyc** (~1.51×). See [VMI DSL vs SoftLib gap](#vmi-dsl-vl128-g8-vs-softlib-handwritten--why--10-elcyc).

## Avg-window / running IPC (SIMD)

Style: W=20, `(t−W, t]` — same as SoftLib mid-IPC SoT.

**VAND primary:** mid IPC **2.50**, mid input **41.6** el/cyc, mean body IPC ~2.31–2.41, mean VLD-win el_L **43.3**, peak el_L **83.2**. Mid VAND/cyc ~0.60 · mid VCGMAX/cyc ~0.65.

### Running-IPC plots (inline)

![SoftLib VAND absmax rolling-20 IPC](plots/sv5_simd_dual64_absmax_vand_rolling20_ipc.png)

![SoftLib reduce-max rolling-20 IPC](plots/sv5_simd_dual64_reducemax_rolling20_ipc.png)

![reduce-max vs absmax IPC compare](plots/sv5_reducemax_vs_absmax_ipc_compare.png)

Detail: [`SIMD_RUNNING_IPC.md`](SIMD_RUNNING_IPC.md), [`ipc/ABSMAX_VAND_MIDPOINT_IPC.md`](ipc/ABSMAX_VAND_MIDPOINT_IPC.md), [`ipc/ROLLING20_IPC.md`](ipc/ROLLING20_IPC.md).

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

Binding resource is **EXU**, not UB. Adding VAND drops EXU roof 64→42.7 and mid 51→41.6 (compute step-down). See [`ROOFTOP_NOT_UB.md`](ROOFTOP_NOT_UB.md).

## VMI DSL VL=128 G=8 vs SoftLib handwritten — why ~10 el/cyc

| | SoftLib VAND (handwritten) | VMI DSL VL=128 G=8 (mask-fixed) |
|---|---|---|
| status | PASS | PASS (`create_mask(VL,size=VL)`) |
| **el/cyc** | **31.39** | **20.79** |
| Δ | — | SoftLib **+10.6** (~**1.51×**) |
| abs | **2× VAND** clear-sign | **VMUL+VMAX** (legacy abs) |
| load / row | **2×** RV_VLD NORMAL | **1×** RV_VLDI **DINTLV_B32** |
| reduce / row | **2×** VCGMAX (G=8) + scalar pair-max | **1×** VCGMAX group=8 |
| store / row | **2×** VST + pack | **1×** VSTI compact |
| instr Σ (R=64) | VLD/VCGMAX/VST = **128** | VLDI/VCGMAX/VSTI = **64** |

### Why the gap stays ~10 after the mask fix

1. **Mask fix only unblocked gold** (`create_mask(64→128)` → slots 4–7 no longer `-inf`). CSV mix stayed **64/64/64** — schedule did not become SoftLib dual64.
2. **Different ISA shape:** SoftLib dual-issues two phys-64 pipes (2×VLD + 2×VAND + 2×VCGMAX + 2×VST). VL128 is a **single-trip** DINTLV load + one reduce + one store per row — not the SoftLib schedule.
3. **Abs cost:** SoftLib SoT uses cheap **VAND**; VL128 still emits **VMUL+VMAX** (closer to SoftLib legacy VMUL absmax **19.65**, which VL128 slightly beats at 20.79).
4. **EXU roof:** SoftLib VAND mid ~97% of ~42.7; VL128 is not on that dual64 EXU plan, so it cannot close the SoftLib VAND gap by mask alone.

**Bottom line:** matching SoftLib **31.39** means dual64 + VAND (or harden FA dual64, today **8.98**), **not** chasing VL128 single-trip further. VL128 is a useful wide-mask path (~2.3× FA) but not SoftLib-equivalent.

Artifacts: `out/sv5_vl128_g8_bug_20261022/FIX_RESULT.md`, `BUG.md`, `DIFF_SCHEDULE.md`.

## Related notes

- SoftLib SIMD **fused** VAND remains the SIMD SoT (~31); membar SoftLib (~12.7) is stage-diagnostic only.
- Simt contig+REDUX upgrade band is separate (membar 6.35 / aicore-misched=0 clean stages 5.08) — not this ANALYSIS’s primary.
