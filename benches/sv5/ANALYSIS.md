# SV5 VMI DSL absmax — VL=128 G=8 VAND clear-sign

**Date:** 2026-10-22 Asia/Hong_Kong  
**Path:** PTO DSL / VMI · `create_mask(VL, size=VL)` · group=8 VCGMAX  
**Shape:** E=8192 fp32 · R=64 · VL=128  
**Abs:** i32 clear-sign `vand` (not `vmul`+`vmax`)

## Headline

| arm | status | VF | **el/cyc** | notes |
|---|---|---:|---:|---|
| **VL=128 G=8 VAND** | **PASS** | 245 | **33.44** | primary · storeA/storeB same |
| VL=128 G=8 prior VMUL abs | PASS | 394 | 20.79 | superseded (+60.8% with VAND) |
| FA dual64 VAND | PASS | 689 | 11.89 | still behind VL128 · was 8.98 VMUL |

Gold maxabs = 0. Artifacts: opsim pack `sv5_vmi_vand_20261022`.

## Abs source

f32 `vand` fails verify (`requires integer-like VMI element type`). Working path:

```python
clear_u = pto.vmi.vbrc(pto.i32(0x7FFFFFFF), size=VL)
xi = pto.vmi.vinterpret_cast(x, pto.i32)
abs_i = pto.vmi.vand(xi, clear_u, mask)
absx = pto.vmi.vinterpret_cast(abs_i, pto.f32)
```

Kernels: `sv5d_absmax_vl128_g8.py` (primary), `sv5d_absmax_amaxstyle.py`, `sv5d_absmax_vcg.py`.

## Instr mix (VL128 storeB, R=64)

| opcode | per-addr | notes |
|---|---:|---|
| **RV_VAND** | ×2 | DINTLV splits VL128 → 2× phys-64 |
| RV_VLDI DINTLV | ×1 | |
| RV_VCGMAX | ×1 | group=8 |
| RV_VSTI | ×1 | |
| RV_VMAX | ×1 | residual from lowering — **not** source abs |
| RV_VMUL | **0** | |

## Notes

1. VAND alone closed the old VMUL abs tax: **20.79 → 33.44**.
2. Residual RV_VMAX×1/row is a lowering artifact; source abs is VAND-only.
3. FA dual64 gains less (+32% → 11.89) and remains a separate schedule track.

## Related

- Mask fix that unblocked gold: `create_mask(VL, size=VL)` (earlier PHYS_VL=64 left slots 4–7 at `-inf`).
- Prior schedule/mask notes: `out/sv5_vl128_g8_bug_20261022/`.
