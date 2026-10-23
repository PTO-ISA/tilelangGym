# SoftLib SIMD dual64 reduce-max — mid-point rolling IPC

**Date:** 2026-10-22 (Asia/Hong_Kong)
**Arm:** SoftLib SIMD dual64 + VCGMAX **absmax VAND clear-sign**
**E = 8192** fp32 · **VF = 261** · whole-kernel **E/VF = 31.387** el/cyc
**Style:** W=20, left-open right-closed `(t−W, t]` (same as absmax / SV1).

## Paths

- PNG: `/workspace/sv567_analysis_draft/sv5_ipc_20261022/sv5_simd_dual64_absmax_vand_rolling20_ipc.png`
- Dump (box): `/workspace/sv567_analysis_draft/sv5_ipc_20261022/dumps/simd_dual64_absmax_vand.dump`
- Dump source (pto-b10): `/tmp/sv5_reducemax_vs_absmax_20261022/opsim_sv5_ccs_softlib_reducemax/OPPROF_20261022120512_QCNPOHZWKPPKBFJB/dump/core0.veccore0.instr_popped_log.dump`

## Markers

| marker | cycle |
|--------|------:|
| firstVLD (body0) | 2654 |
| lastVLD | 2832 |
| firstVCGMAX | 2662 |
| firstVST | 2673 |
| lastVST (end) | 2871 |
| **t_mid = (body0+lastS)//2** | **2762** |
| t_mid_vld = (firstVLD+lastVLD)//2 | 2743 |

Body span: `2654 → 2871` = **217** cyc · VLD-win = **178** · post = **39**

Pipe raw counts: `{'RV_VLD': 128, 'RV_VAND': 128, 'RV_VCGMAX': 128, 'RV_VST': 128}` · pipe_n (VLD+VCGMAX+VST) = **512**

## Headline numbers

| metric | value |
|--------|------:|
| peak IPC | **2.900** @ 2681 |
| **mean IPC body** (firstVLD→lastVST) | **2.306** |
| mean IPC over VLD-win | 2.409 |
| **IPC at t_mid (body midpoint)** | **2.500** |
| IPC at t_mid_vld | 2.500 |
| **input el_L/cyc at t_mid** = (#VLD·64)/W | **41.60** |
| input el_L/cyc at t_mid_vld | 41.60 |
| **mean el_L over VLD-win** | **43.32** |
| **peak el_L** | **83.20** @ 2673 |
| VCGMAX ops/cyc at t_mid | 0.600 |
| peak VCGMAX ops/cyc | 0.750 @ 2787 |

## vs absmax dual64 (same style)

| arm | VF | E/VF | mean IPC body | mid IPC | mean el_L win | peak el_L |
|-----|---:|-----:|--------------:|--------:|--------------:|----------:|
| **absmax VAND clear-sign** | 261 | **31.39** | **2.306** | **2.500** | **43.32** | **83.20** |
| absmax dual64 (ref) | 417 | 19.65 | 1.699 | — | 24.82 | 57.60 |

Dropping abs (VMUL+VMAX) shortens the dep chain to `VLD→VCGMAX→VST`, so body IPC and
input el_L both rise vs absmax. Mid-point input rate is still below the SLIDE resource
roof (~64 el/cyc = 1 VCGMAX/cyc × 64 el) — consistent with shallow trip ILP / VCGMAX latency.

