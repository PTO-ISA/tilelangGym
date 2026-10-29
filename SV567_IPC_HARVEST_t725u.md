# SV5/SV6/SV7 running IPC harvest — t725u

**When:** 2026-10-29 Asia/Hong_Kong  
**Method:** W=20 left-open right-closed `(t−W, t]` over issue timestamps in `core0.veccore0.instr_log.dump`  
**Script:** `/workspace/_harvest_mid_ipc_t725u_v2.py` (RV_ + SIMT_ issue names)  
**Raw JSON:** [`ipc/ipc_all_t725u.json`](ipc/ipc_all_t725u.json)  
**Do not invent.** Mid=0.0 at geometric issue-span midpoint = **MID_IDLE_GAP** (SIMT prologue/epilogue sparse) — cite **mean_ipc_body** and mark mid unreliable until Body4 firstLDS→lastSTS window is re-harvested.

## SoftLib CCE footnote peer (pre-existing)

| Arm | mid IPC W20 | mid input el/cyc | Source |
|-----|------------:|-----------------:|--------|
| SoftLib dual64 absmax VAND | **2.50** | **41.6** | `benches/sv5/ipc/ABSMAX_VAND_MIDPOINT_IPC.md` · VF261 · E/VF 31.39 |

## Outer VMI pure-VMAX (t720 dumps — OK)

| G | tag | mid W20 | mean body | t_mid | notes |
|--:|-----|--------:|----------:|------:|-------|
| 16 | `outer_g16_vmax_vand_fold_vl256` | **3.05** | 2.80 | 8478 | VAND+VMAX+VLDS dominant |
| 32 | `outer_g32_vmax_vand_fold_vl256` | **3.00** | 2.76 | 6646 | |
| 64 | `outer_g64_vmax_vand_fold_vl128` | **3.00** | 2.74 | 6729 | |
| 128 | `outer_g128_vmax_vand_dual4_vl64` | **3.00** | 2.72 | 7393 | dual4 SoT |

Pack: `/mnt/fluxdata/happybot/vmi_absmax_outer_vmax_t720u/out/<tag>/OPPROF_*/dump/core0.veccore0.instr_log.dump`

## Inner VMI Gmap (t699 live — G16 MISSING)

| G | tag | mid W20 | mean body | status |
|--:|-----|--------:|----------:|--------|
| 16 | VL128/NG8 VAND dig SoT | — | — | **MISSING** — not under `vmi_absmax_gmap_t699u_20261027/out/` (only g32/64/128). Dig pack `sv_simd_regress_semantics_20261026` not re-harvested this turn. |
| 32 | `g32_vcg_vl256_ng8_vand` | **1.35** | 1.26 | OK (live peer; dig SoT same recipe class) |
| 64 | `g64_vcmax_vl64_vabs` | **2.45** | 2.28 | OK SoT abs path |
| 128 | `g128_vmaxfold_vcmax_vand` | **2.60** | 2.12 | OK SoT |

**Next harvest (G16 inner):** re-opsim `benches/sv5/sv5d_absmax_vl128_g8.py` on pto-b10 →  
`python3 /workspace/_harvest_mid_ipc_t725u_v2.py out.json g16_inner $OUTDIR`

## SoftLib SIMT inner 1×G (t698 dumps)

| G | role | tag dir | mid W20 | mean body | note |
|--:|------|---------|--------:|----------:|------|
| 16 | stock | `…_g16_t128` | 0.00 | **0.34** | MID_IDLE_GAP |
| 16 | best peer membar 4.63 | `…_loadall_membar_g16_t512` | **1.45** | 1.34 | tip SoT **4.74** loadall+absall drain is **t697 pack** — dump **MISSING** this turn; peer IPC cited |
| 16 | dual_drain (not tip SoT) | `…_dual_drain_g16_t512` | 0.60 | 0.81 | el/cyc 3.40 on t698 — not SoT |
| 32 | stock | `…_g32_t128` | 0.45 | 0.32 | |
| 32 | best dual-drain 8.22 | `…_dual_drain_g32_t512` | **2.10** | 1.61 | OK |
| 64 | stock | `…_g64_t128` | 0.50 | 0.38 | |
| 64 | best T512 5.54 | `…_g64_t512` | **1.25** | 0.97 | OK |
| 128 | stock | `…_g128_t64` | 0.30 | 0.21 | |
| 128 | best T256 3.72 | `…_g128_t256` | **0.60** | 0.52 | OK |

Pack: `/mnt/fluxdata/happybot/glm53_softlib_simt_absmax_1xg_t698u/`

## SoftLib SIMT outer Nx1 (t708 dumps)

| G | role | mid W20 | mean body | note |
|--:|------|--------:|----------:|------|
| 16 | stock fused 9.68 | **1.55** | 1.21 | OK |
| 16 | best membar 11.69 | **1.55** | 1.21 | OK (same mid; schedule differs) |
| 32 | stock 4.74 | **1.10** | 0.64 | |
| 32 | best 9.58 | **0.90** | 0.96 | mid &lt; mean — window sensitivity |
| 64 | stock 3.46 | **0.50** | 0.42 | |
| 64 | best 6.08 | **0.85** | 0.59 | |
| 128 | stock 2.15 | 0.00 | **0.23** | MID_IDLE_GAP |
| 128 | best 2.19 | 0.00 | **0.24** | MID_IDLE_GAP |

Pack: `/mnt/fluxdata/happybot/glm53_softlib_simt_absmax_nx1_t708u/`

## Status rollup

| Family | Coverage |
|--------|----------|
| Outer VMI | **COMPLETE** G16–128 mid ≈ 3.0 |
| Inner VMI | G32/64/128 OK · **G16 MISSING** |
| SoftLib CCE VAND | **COMPLETE** mid 2.50 (prior) |
| SoftLib SIMT 1×G | stock/best dumps OK · tip G16 **4.74** exact dump MISSING (peer membar IPC OK) · several stock mid=IDLE_GAP |
| SoftLib SIMT Nx1 | G16–64 OK · G128 MID_IDLE_GAP (mean_body only) |

## Refine later (one-liner)

```bash
# Body4 window (first SIMT_LDS → last SIMT_STS) instead of full issue-span midpoint:
python3 /workspace/_harvest_mid_ipc_t725u_v2.py /tmp/ipc.json LABEL /path/to/out/tagdir
# Optional: extend script with --body4-window for SIMT idle-gap arms.
```
