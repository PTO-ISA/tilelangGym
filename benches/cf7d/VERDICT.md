# VERDICT — CF7: Nested outer×inner if (CF7d)

**Focus paths:** handwritten SIMD DSL (`*d`). **Deferred:** simt-asc, simd-pto (*v), simt-pto SoftLib.

**Pins:** Ascend950PR_9599 opsim · CF7d first-cut 2026-10-30 · issue [#10](https://github.com/PTO-ISA/tilelangGym/issues/10).

## Status

| Path | Status |
|------|--------|
| **Handwritten SIMD DSL** (`cf7d`) | **PARTIAL PASS** — 27/27 first-cut cells PASS; A≡B; VF_cycle crossover **G=0**; PR9 overlays 27/27; SEMANTICS_GATE filled in ANALYSIS. SimtVF / full G×p_gate grid still open. |
| **simt-asc / simd-pto (*v)** | **Not in this PR** — *d-only land |

## First-cut call (VF cycle primary)

- Arms: A=pure_slack · B=slack+AND · C=scalar_if
- Grid: `G∈{0,16,64}` × `p_gate∈{25,50,95}` × A/B/C
- **A ≡ B** everywhere; **C** wins at thin G (crossover **G=0**); at `p_gate=95` bar fails for G=16/64
- See [`ST_CF7D_FIRSTCUT_RESULT.md`](../../reports/ST_CF7D_FIRSTCUT_RESULT.md) and [`ST_CF7D_ANALYSIS.md`](../../reports/ST_CF7D_ANALYSIS.md)

## Files

- Kernel: `cf7d_outer_inner_if.py`
- Oneshot: `../../scripts/oneshot/oneshot_ptodsl_cf7d.sh` (or local `oneshot_ptodsl_cf7d.sh`)
- Design: `ST_CF7_OUTER_INNER_IF_DESIGN.md`
