# VMI DSL VL=128 G=8 vs SoftLib handwritten — ~10 el/cyc gap

**Date:** 2026-10-22 Asia/Hong_Kong  
Full write-up: [`ANALYSIS.md`](ANALYSIS.md#vmi-dsl-vl128-g8-vs-softlib-handwritten--why--10-elcyc).

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

**Why the gap stays ~10 after the mask fix**

1. Mask fix only unblocked gold — schedule stayed single-trip (CSV mix 64/64/64), not SoftLib dual64.
2. SoftLib dual-issues two phys-64 pipes; VL128 is one DINTLV load + one reduce + one store per row.
3. SoftLib SoT uses cheap **VAND**; VL128 still emits **VMUL+VMAX** (near SoftLib legacy 19.65).
4. SoftLib VAND mid sits ~97% of EXU ~42.7; VL128 is not on that plan.

**Bottom line:** matching SoftLib **31.39** needs dual64 + VAND (or harden FA dual64, today **8.98**), not more VL128 single-trip.
