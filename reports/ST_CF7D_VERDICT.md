# VERDICT — CF7d: global-gate + reduce-in-b (PTO-DSL)

**Focus paths:** handwritten SIMD DSL (`*d`). **Deferred:** simt-asc, simd-pto (*v), SoftLib SIMT.

**Pins:** Ascend950PR_9599 opsim · tip 2026-11-03 · issue [#10](https://github.com/PTO-ISA/tilelangGym/issues/10) · oneshot 17:44–18:26 HKT.

## Status

| Case | Status |
|------|--------|
| **Hoisted global-gate** (`cf7d_outer_inner_if.py`) | **PARTIAL (measured)** — B VF flat vs gate, scales with G; **C≻B** for G∈{16,64} (A1 crossover **G=16** gate_on). `era=cf7d-global-gate-ptodsl`. |
| **Reduce-in-b sibling** (`cf7d_reduce_in_b.py`) | **PARTIAL (measured)** — still **C≻B** at G≥16; hoist artifact **reduced not removed**. `era=cf7d-rib-ptodsl`. |
| **simt / SoftLib** | Not in this PR (`arms.yaml` MISSING) |
| **PR9 / S3 mix** | **MISSING** — dumps on disk; next-cmd in ANALYSIS §7 |

## Gold (both cases)

- `gate_on = sum(gate) > GATE_SUM_THR` (global over B lanes) — **not** per-b outer host `n_trips`.
- When on: B×G nest (scale+kill). When false: skip nest (gold).
- **Hoisted:** reduce once outside `b`. **Rib:** reduce inside each `b` (Aims A5).

## Headline ratios (from SUMMARY_dual_era_oneshot.tsv)

| Era | G | gate | VF(B) | VF(C) | C/B |
|-----|--:|------|------:|------:|----:|
| hoisted | 16 | off | 2237 | 821 | 0.367 |
| hoisted | 16 | on | 2237 | 1283 | 0.574 |
| hoisted | 64 | off | 8827 | 830 | 0.094 |
| hoisted | 64 | on | 8827 | 5800 | 0.657 |
| rib | 16 | off | 2118 | 1386 | 0.654 |
| rib | 16 | on | 2118 | 1609 | 0.760 |
| rib | 64 | off | 8865 | 3234 | 0.365 |
| rib | 64 | on | 8865 | 5471 | 0.617 |

## STALE first-cut

Prior 3-arm / per-b-outer RESULT + PR9 plots + Appendix H remain **STALE** `era=per-b-outer` (t821u).

## Files

- Hoisted: `cf7d_outer_inner_if.py` · `smoke_thin.py`
- Rib: `cf7d_reduce_in_b.py` · `smoke_thin_rib.py` · `ST_CF7_REDUCE_IN_B.md`
- Arms: `report/arms.yaml` · `report/SUMMARY_dual_era_oneshot.tsv` · `report/SUMMARY.tsv`
- ANALYSIS Aims A1–A5 + How-to-run + SEMANTICS_GATE **PARTIAL**
