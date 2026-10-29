# SoftLib dual64 absmax via VAND(~0x80000000) — 2026-10-22 Asia/Hong_Kong

## Result
| arm | status | VF_cyc | **el/cyc** | wall_us | instr mix |
|---|---|---:|---:|---:|---|
| reduce-max (prior) | PASS | 234 | **35.01** | — | VLD+VCGMAX+VST ×128 |
| absmax VMUL(-1)+VMAX (prior) | PASS | 417 | **19.65** | — | VLD+VMUL+VMAX+VCGMAX+VST ×128 |
| **absmax VAND clear-sign (new)** | **PASS** | **261** | **31.39** | 2.99 | **VLD+VAND+VCGMAX+VST ×128** |

- gold maxabs = **0.0**
- E=8192 fp32, PHYS_VL=64, dual64, VCGMAX G=8 + pair-max → gold G=16
- source: `sources/sv5_ccs_softlib_absmax_vand.asc`
- MicroAPI: `asc_and(dst, x, clear_sign, mask)` with `clear_sign = bitcast_f32(~0x80000000u)` → emits **RV_VAND** (ADD unit, dual-issue)

## Mid-VF rolling IPC (W=20, `(t−W,t]`)
| | reduce-max | absmax VMUL+VMAX | **absmax VAND** |
|---|---:|---:|---:|
| mid IPC | 2.40 | 1.95 | **2.50** |
| mid input el/cyc | 51.2 | 25.6 | **41.6** |
| mean IPC body | 1.96 | 1.70 | **2.31** |
| whole-kernel el/cyc | 35.01 | 19.65 | **31.39** |

## Rooftop (A5 EXU = ceil(n_non_SLIDE/2) + n_SLIDE)
| arm | n_non | n_SLIDE | EXU cyc/row | resource el/cyc | measured whole | mid |
|---|---:|---:|---:|---:|---:|---:|
| reduce-max | 0 | 2 | 2 | **64** | 35.01 | 51.2 (~80% mid) |
| VAND absmax | 2 | 2 | **3** | **~42.7** | 31.39 | **41.6 (~97% mid)** |
| VMUL+VMAX absmax | 4 | 2 | 4 | **32** | 19.65 | 25.6 (~80% mid) |

VAND lands between reduce-max and old absmax as predicted: one dual-issueable ADD (VAND) instead of two (VMUL+VMAX). Mid already sits on the ~43 EXU roof; whole-kernel 31.39 is head/tail diluted.

## Artifacts
- out: `/workspace/sv5_simd_baseline/out/sv5_vand_absmax_20261022/`
- IPC plot: `/workspace/sv567_analysis_draft/sv5_ipc_20261022/sv5_simd_dual64_absmax_vand_rolling20_ipc.png`
- dump: `.../dumps/simd_dual64_absmax_vand.dump`
- pto-b10 OPPROF: `/tmp/sv5_vand_absmax_20261022/opsim/sv5_ccs_softlib_absmax_vand/OPPROF_20261022143621_HYDESHWSZULANWPS`
