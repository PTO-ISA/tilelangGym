# SV1 aligned matrix — results (pto-b10)

**When:** 2026-10-17 15:09–15:13 Asia/Hong_Kong  
**Out:** /tmp/sv1_aligned_e16k_tile8k_20261017  
**Kernel:** kernels/sv1t_tiled_add.py (vf_region / SimtVF, serial tiles, Parallel(TILE))  
**ST_VF_MODE:** simt-asc  

## Pins (NEW run — use these for PR#7 align tables)
- CANN: cann_92b2_local /usr/local/Ascend/cann_92b2/cann-9.2.0-beta.2
- TILELANG_DEPS: /mnt/fluxdata/happybot/projects/tilelang-pto-isa-st @ **e700041f** (float4 / max_vector_bits 64→128)
- CAMODEL: .camodel_deps_vmi018
- PTOAS sim_dsl: /home/happybot/PTOAS-vmi/scripts/sim_dsl.sh
- SOC: Ascend950PR_9599

## PASS table (all NEW under above pins)

### §4.2 Thread-dim — E=16384 TILE=8192
| T | VF_cyc | el/cyc | IPC_body | us | status |
|---|--------|--------|----------|-----|--------|
| 128 | 8221 | **1.9929** | 0.5060 | 8.92 | PASS |
| 256 | 7041 | **2.3269** | 0.5999 | 8.23 | PASS |
| 512 | 6825 | **2.4006** | 0.6377 | 8.08 | PASS |

### §4.1 Tile-size — E=16384 T=512
| TILE | VF_cyc | el/cyc | IPC_body | us | status |
|------|--------|--------|----------|-----|--------|
| 2048 | 6721 | **2.4377** | 0.6190 | 8.03 | PASS |
| 4096 | 6692 | **2.4483** | 0.6742 | 7.99 | PASS |
| 8192 | 6825 | **2.4006** | 0.6377 | 8.08 | PASS |
| 16384 | 6721 | **2.4377** | 0.6190 | 8.03 | PASS |

Files: SUMMARY_aligned_matrix.tsv, SUMMARY_aligned_matrix.json

## Already existed vs newly run
| Case | Prior artifact | This run |
|------|----------------|----------|
| TILE8192×T128 @ E16K | OLD /tmp/sv1_cost_sweep cann_91b3: **2.9942** el/cyc | **Re-run NEW** 1.9929 (do not mix) |
| TILE8192×T256 | missing | **NEW** 2.3269 |
| TILE8192×T512 | missing | **NEW** 2.4006 |
| TILE2048/4096/16384 ×T512 | missing (old tile ladder was @T128 only) | **NEW** |
| SIMD DSL E=16384 VL=64 | EXISTING SUMMARY_sv1d_size_ladder.tsv: **31.5077** el/cyc, VF=520, IPC=1.9788 | not re-run |
| Untiled T128/256/512 (TILE=E) old | EXISTS in tile16k TSV (2.87/3.37/3.71) | superseded for align by TILE16384@T512=2.4377 under new pins |

## SIMD head-to-head note
Handwritten SIMD DSL has **no TILE axis** (whole-E VL=64 chunks).  
Compare point: **Simt TILE=8192 @ T512 = 2.4006 el/cyc** vs **SIMD E=16384 = 31.5077 el/cyc** (whole-E, not tiled). Label clearly in §3.

## IPC plots
- **Not regenerated as PNGs** this run (no Body4 IPC plot script wired for TILE8192 under new OUT).
- Per-tag **IPC_body** is in SUMMARY_aligned_matrix.tsv (TILE8192@T512 = 0.6377).
- Instr CSVs live under opsim_<tag>/OPPROF_*/simulator/core0.veccore0/core0.veccore0_instr_exe.csv.
- Old report enches/sv1/data/IADD_IPC_REPORT_tile16k.txt was untiled / cann_91b3 — stale for align.
- To plot later: harvest VF + SIMT_* from those CSVs (same as scripts/harvest_aligned.py) and feed sv1_cost_sweep/scripts/plot_and_fit.py or a one-off matplotlib IPC bar.

## Recommended ANALYSIS table numbers (aligned, new pins)
- **§4.2 thread-dim (E=16K TILE=8K):** T128 **1.99** · T256 **2.33** · T512 **2.40** el/cyc  
- **§4.1 tile-size (E=16K T=512):** TILE2K **2.44** · 4K **2.45** · 8K **2.40** · 16K **2.44** el/cyc  
- **§3 Simt vs SIMD:** Simt TILE8K@T512 **2.40** vs SIMD whole-E16K **31.51** el/cyc (different axis — footnote)  
- Call out pin change vs prior ANALYSIS (old TILE8K@T128 2.99 under cann_91b3 / older tip) — numbers are **not** comparable without re-baselining the whole doc.

## Mix note (codegen)
New tip: LDS:FADD:STS ≈ 256:512:128 (vs old 512:512:256 .64). Consistent with float4 / wider vector default on e700041f — higher LDS width but lower el/cyc in this tight-add schedule.
