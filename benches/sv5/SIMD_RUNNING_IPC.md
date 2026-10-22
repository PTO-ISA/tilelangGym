# SoftLib SIMD — avg-window / running IPC (SV5 absmax VAND)

**Date:** 2026-10-22 Asia/Hong_Kong  
**Style:** W=20 rolling, left-open right-closed `(t−W, t]` — same as SoftLib mid IPC SoT.

## Primary arm: absmax **VAND** clear-sign (dual64)

![SoftLib VAND absmax rolling-20 IPC](plots/sv5_simd_dual64_absmax_vand_rolling20_ipc.png)

| metric | value |
|---|---:|
| whole-kernel el/cyc | **31.39** (VF 261) |
| **midpoint IPC** | **2.50** |
| **midpoint input el/cyc** | **41.6** |
| mean IPC body | 2.31–2.41 |
| mean el_L over VLD-win | **43.3** |
| peak el_L | 83.2 |
| mid VAND/cyc | 0.60 |
| mid VCGMAX/cyc | 0.65 |

EXU roof ~42.7 → mid **41.6 ≈ 97%**.

## Peers (same W=20)

| arm | VF el/cyc | mid IPC | mid input el/cyc |
|---|---:|---:|---:|
| SoftLib reduce-max | 35.01 | ~2.40 | **51.2** |
| **SoftLib absmax VAND** | **31.39** | **2.50** | **41.6** |
| SoftLib absmax VMUL+VMAX (legacy) | 19.65 | ~1.95 | 25.6 |

Also (superseded VMUL abs): VMI DSL VL=128 G=8 **20.79** · FA dual64 **8.98**. **Update:** VL128 VAND now **33.44** (beats SoftLib **31.39**); FA VAND **11.89** — see [`ANALYSIS.md`](ANALYSIS.md#vmi-dsl-vl128-g8-vand--closes-the-softlib-gap).

## Artifacts
- Plot (inline above): `plots/sv5_simd_dual64_absmax_vand_rolling20_ipc.png`
- Detail: `ipc/ABSMAX_VAND_MIDPOINT_IPC.md`, `ipc/ROLLING20_IPC.md`
- Source: `softlib/sv5_ccs_softlib_absmax_vand.asc`

## vs rooftop (see `ROOFTOP_NOT_UB.md`)

**Not UB-bound.** EXU theory **~42.7** el/cyc (`128/3`); mid **41.6 ≈ 97%**; whole **31.39 ≈ 73%**. UB LD roof 128 and ST roof 64 both above EXU.
