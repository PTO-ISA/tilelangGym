# VMI DSL VL=128 G=8 VAND — closes SoftLib gap

**Date:** 2026-10-22 Asia/Hong_Kong  
Full write-up: [`ANALYSIS.md`](ANALYSIS.md#vmi-dsl-vl128-g8-vand--closes-the-softlib-gap).

| | SoftLib VAND (handwritten) | VMI DSL VL=128 G=8 **VAND** | VMI prior (VMUL abs) |
|---|---|---|---|
| status | PASS | PASS (`create_mask(VL,size=VL)`) | PASS |
| **el/cyc** | **31.39** | **33.44** | 20.79 |
| Δ vs SoftLib | — | **beats ~1.07×** | SoftLib +10.6 (~1.51×) |
| abs | **2× VAND** clear-sign | **i32 VAND** + `vinterpret_cast` | VMUL+VMAX (legacy) |
| load / row | **2×** RV_VLD NORMAL | **1×** RV_VLDI **DINTLV_B32** | same |
| reduce / row | **2×** VCGMAX (G=8) + pair-max | **1×** VCGMAX group=8 | same |
| store / row | **2×** VST + pack | **1×** VSTI compact | same |

**What closed the gap**

1. Abs cost: VL128 source `vmax+vmul` → i32 clear-sign VAND → **20.79 → 33.44** (+60.8%).
2. f32 `vand` rejected; i32 `vinterpret_cast` path PASSes gold.
3. Schedule still single-trip (≠ SoftLib dual64), yet whole-kernel el/cyc now **above** SoftLib 31.39.
4. FA dual64 VAND is **11.89** (was 8.98) — still ~2.6× behind SoftLib.

**Bottom line:** VL128 + VAND is the VMI SoT (**33.44**). SoftLib remains the dual64 EXU-roof reference. Residual RV_VMAX×1/row is lowering, not source abs.
