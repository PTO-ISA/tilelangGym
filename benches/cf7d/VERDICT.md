# VERDICT — CF7d: global-gate + reduce-in-b (PTO-DSL)

**Focus paths:** handwritten SIMD DSL (`*d`). **Deferred:** simt-asc, simd-pto (*v), SoftLib SIMT.

**Pins:** Ascend950PR_9599 opsim · tip 2026-11-03 · issue [#10](https://github.com/PTO-ISA/tilelangGym/issues/10).

## Status

| Case | Status |
|------|--------|
| **Hoisted global-gate** (`cf7d_outer_inner_if.py`) | **PARTIAL** — 2 arms B=vand/PAND, C=`pto.if_`; reduce once then reuse ×B. Compile smoke OK. Live VF **UNMEASURED** (`era=cf7d-global-gate-ptodsl`). |
| **Reduce-in-b sibling** (`cf7d_reduce_in_b.py`) | **PARTIAL** — same gold `sum(gate)>THR` but **recomputed inside each `b`** so C cannot fake win by hoisting. Arms B/C mirror. Compile smoke this land. Live VF **UNMEASURED** (`era=cf7d-rib-ptodsl`). |
| **simt / SoftLib** | Not in this PR (`arms.yaml` MISSING) |

## Gold (both cases)

- `gate_on = sum(gate) > GATE_SUM_THR` (global over B lanes) — **not** per-b outer host `n_trips`.
- When on: B×G nest (scale+kill). When false: skip nest (gold).
- **Hoisted:** reduce once outside `b`. **Rib:** reduce inside each `b` (Aims A5).

## STALE first-cut

Prior 3-arm / per-b-outer RESULT + PR9 plots + Appendix H remain **STALE** `era=per-b-outer` (t821u). Do **not** invent VF for either new era.

## Files

- Hoisted: `cf7d_outer_inner_if.py` · `smoke_thin.py`
- Rib: `cf7d_reduce_in_b.py` · `smoke_thin_rib.py` · `ST_CF7_REDUCE_IN_B.md`
- Arms: `report/arms.yaml` (4 *d spine rows) · `SUMMARY.tsv` TODO
- ANALYSIS Aims A1–A5 + How-to-run + SEMANTICS_GATE INCONCLUSIVE
