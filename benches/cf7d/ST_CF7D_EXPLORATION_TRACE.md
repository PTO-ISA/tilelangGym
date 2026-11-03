# EXPLORATION_TRACE — CF7d

**ST id:** `cf7d`  
**Nature:** Reconstruct from DESIGN+RESULT+TSV/kernel/oneshot (2026-10-31), then **2026-11-03 global-gate 2-arm redesign** (Lok-approved): drop A; B=vand/PAND; C=`pto.if_`; gold = `sum(gate)>GATE_SUM_THR`.  
**Not** a full live `tilelanggym-st-explore-from-prompt` motif ladder beyond first-cut *d + redesign land.

---

## Timeline (as evidenced on disk)

| Step | Evidence | Outcome |
|------|----------|---------|
| 1. Gap vs CF1–CF6 | Design §1 | No prior outer-loop `if` × inner-loop `if` + AND-merge micro |
| 2. Aims lock (per-b era) | Design §0 / §3–§4 | Arms A/B/C; R\*×p_gate; 0.90·min(A,B) |
| 3. First-cut grid | Design §3.3 | G∈{0,16,64} × p∈{25,50,95} × A/B/C (=27) |
| 4. Kernel land (per-b) | `cf7d_outer_inner_if.py` | C via host `n_trips`; B as-landed ≡ A |
| 5. Env blocker | RESULT header | DEPS+MLIR_PY + ASC/set_env only |
| 6. Oneshot measure | `oneshot_ptodsl_cf7d.sh` | 27/27 PASS → TSV/RESULT (`era=per-b-outer`) |
| 7–12. ANALYSIS v1–v2 + dumps/PR9/gate | TRACE tip `0bf351e` | SEMANTICS_GATE PASS *d; PARTIAL suite |
| 13. Global-gate rev (Lok) | `ST_CF7_GLOBAL_GATE_REV.md` + chat t34u–t40u | Outer = `gate_sum > THR`; drop A; kill inside G; B outer/G inner |
| 14. API probe | `/tmp/cf7d_probe_apis.py` on pto-b10 | `vcadd` / `pto.if_` / `load_scalar` OK |
| 15. Kernel rewrite | `benches/cf7d/cf7d_outer_inner_if.py` rev2 | B/C only; SYNTAX_OK; gold global |
| 16. Docs spine | ANALYSIS / DESIGN / arms.yaml / this TRACE | Aims A1–A4 global; Appendix H STALE; SEMANTICS_GATE INCONCLUSIVE |
| 17. Compile smoke | `smoke_thin.py` B then C on pto-b10 (DEPS vmi018 + MLIR) | **COMPILE_OK B** + **COMPILE_OK C** (2026-11-03 ~15:45 HKT); vcadd needs `reassoc=True`, `create_mask(B,size=B)` + `mask1` for gs store |
| 18. Remeasure | oneshot update | **OPEN** — live VF UNMEASURED |

---

## Semantic lock (short) — live era

- Gold: **one** global `if sum(gate) > GATE_SUM_THR` wrapping B×(G×(scale+kill)).
- Arm B: `vcadd→vbrc→vcmp→vand` into kill + `vsel(p_outer)` publish; body still scheduled when false.
- Arm C: `pto.if_(gate_on)` — **forbidden** host `n_trips` specialization.
- Primary metric: opsim VF_cycles (not µs alone).
- Prior fail-mode **B≡A / C=host skip** is **STALE** (`era=per-b-outer`).

---

## Stop / open

- Global-gate kernel+docs **landed**; KPI **UNMEASURED**.
- Remeasure + PR9 + opcode mix + SEMANTICS_GATE refresh = open.
- Fill-in / Simt CF7 / SoftLib / merge = open (Lok yes for merge).
