# SoftLib SIMD dual64 reduce-max — mid-point rolling IPC

**Date:** 2026-10-22 (Asia/Hong_Kong)
**Arm:** SoftLib SIMD dual64 + VCGMAX **reduce-max (no abs)**
**E = 8192** fp32 · **VF = 234** · whole-kernel **E/VF = 35.009** el/cyc
**Style:** W=20, left-open right-closed `(t−W, t]` (same as absmax / SV1).

## Paths

- PNG: `/workspace/sv567_analysis_draft/sv5_ipc_20261022/sv5_simd_dual64_reducemax_rolling20_ipc.png`
- Dump (box): `/workspace/sv567_analysis_draft/sv5_ipc_20261022/dumps/simd_dual64_reducemax.dump`
- Dump source (pto-b10): `/tmp/sv5_reducemax_vs_absmax_20261022/opsim_sv5_ccs_softlib_reducemax/OPPROF_20261022120512_QCNPOHZWKPPKBFJB/dump/core0.veccore0.instr_popped_log.dump`

## Markers

| marker | cycle |
|--------|------:|
| firstVLD (body0) | 2656 |
| lastVLD | 2794 |
| firstVCGMAX | 2662 |
| firstVST | 2673 |
| lastVST (end) | 2846 |
| **t_mid = (body0+lastS)//2** | **2751** |
| t_mid_vld = (firstVLD+lastVLD)//2 | 2725 |

Body span: `2656 → 2846` = **190** cyc · VLD-win = **138** · post = **52**

Pipe raw counts: `{'RV_VLD': 128, 'RV_VCGMAX': 128, 'RV_VST': 128}` · pipe_n (VLD+VCGMAX+VST) = **384**

## Headline numbers

| metric | value |
|--------|------:|
| peak IPC | **2.950** @ 2680 |
| **mean IPC body** (firstVLD→lastVST) | **1.961** |
| mean IPC over VLD-win | 2.218 |
| **IPC at t_mid (body midpoint)** | **2.400** |
| IPC at t_mid_vld | 2.400 |
| **input el_L/cyc at t_mid** = (#VLD·64)/W | **51.20** |
| input el_L/cyc at t_mid_vld | 51.20 |
| **mean el_L over VLD-win** | **55.62** |
| **peak el_L** | **108.80** @ 2675 |
| VCGMAX ops/cyc at t_mid | 1.000 |
| peak VCGMAX ops/cyc | 1.000 @ 2681 |

## vs absmax dual64 (same style)

| arm | VF | E/VF | mean IPC body | mid IPC | mean el_L win | peak el_L |
|-----|---:|-----:|--------------:|--------:|--------------:|----------:|
| **reduce-max (no abs)** | 234 | **35.01** | **1.961** | **2.400** | **55.62** | **108.80** |
| absmax dual64 (ref) | 417 | 19.65 | 1.699 | — | 24.82 | 57.60 |

Dropping abs (VMUL+VMAX) shortens the dep chain to `VLD→VCGMAX→VST`, so body IPC and
input el_L both rise vs absmax. Mid-point input rate is still below the SLIDE resource
roof (~64 el/cyc = 1 VCGMAX/cyc × 64 el) — consistent with shallow trip ILP / VCGMAX latency.

