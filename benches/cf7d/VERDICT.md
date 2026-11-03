# VERDICT — CF7d: global-gate outer×inner if (PTO-DSL)

**Focus paths:** handwritten SIMD DSL (`*d`). **Deferred:** simt-asc, simd-pto (*v), SoftLib SIMT.

**Pins:** Ascend950PR_9599 opsim · tip redesign 2026-11-03 · issue [#10](https://github.com/PTO-ISA/tilelangGym/issues/10).

## Status (NEW era=`cf7d-global-gate-ptodsl`)

| Path | Status |
|------|--------|
| **Handwritten SIMD DSL** (`cf7d`) | **PARTIAL** — kernel redesigned to **2 arms** (B vand/PAND, C `pto.if_`) + **global** `sum(gate)>GATE_SUM_THR`. Compile smoke OK on pto-b10. Live VF KPI **UNMEASURED** (oneshot remeasure pending). `st_report check --st cf7d` OK. |
| **simt-asc / simd-pto (*v) / SoftLib** | **Not in this PR** — *d-only land; `report/arms.yaml` marks them MISSING |

## Gold shape (review this)

- Gate is **global** over B lanes: `gate_on = (sum(gate) > GATE_SUM_THR)` — **not** per-b outer, **no** host `n_trips`.
- When `gate_on`: run outer B × inner G nest (scale + kill). When false: skip whole nest (gold).
- **Arm B:** `vcadd(reassoc=True)` → `vbrc` → `vcmp` → **`vand`/PAND** (+ `vsel` publish).
- **Arm C:** in-VF scalar compare + **`pto.if_(gate_on)`** around the nest.
- G sweep intended: `G ∈ {0,8,16,32,64}` (B=8). Dropped arm A.

## STALE first-cut (do not treat as live KPI)

Prior 3-arm / per-b-outer / `p_gate`×`n_trips` RESULT + PR9 plots remain under `benches/cf7d/` and ANALYSIS **Appendix H** as **STALE** `era=per-b-outer` (t821u keep). Do **not** invent VF numbers for the new era.

- See [`ANALYSIS.md`](ANALYSIS.md) (Aims A1–A4, How-to-run, SEMANTICS_GATE INCONCLUSIVE until remeasure)
- Rev note: [`ST_CF7_GLOBAL_GATE_REV.md`](ST_CF7_GLOBAL_GATE_REV.md)
- Design: [`ST_CF7_OUTER_INNER_IF_DESIGN.md`](ST_CF7_OUTER_INNER_IF_DESIGN.md)

## Files

- Kernel: `cf7d_outer_inner_if.py` · smoke: `smoke_thin.py`
- Arms: `report/arms.yaml` · spine: `report/SUMMARY.tsv` (TODO until remeasure)
- Oneshot: `oneshot_ptodsl_cf7d.sh` — **still first-cut A/B/C grid**; needs refresh for global-gate B/C before live KPI
