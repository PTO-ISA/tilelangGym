# EXPLORATION_TRACE — CF7d

**ST id:** `cf7d`  
**Nature:** Reconstruct from DESIGN+RESULT+TSV/kernel/oneshot, then **2026-10-31 dump sync + PR9 + SEMANTICS_GATE refresh** on analysis box.  
**Not** a full live `tilelanggym-st-explore-from-prompt` motif ladder beyond first-cut *d.

---

## Timeline (as evidenced on disk)

| Step | Evidence | Outcome |
|------|----------|---------|
| 1. Gap vs CF1–CF6 | Design §1 | No prior outer-loop `if` × inner-loop `if` + AND-merge micro |
| 2. Aims lock | Design §0 / §3–§4 (Lok rev: crossover = VF cycle) | Arms A pure_slack / B slack_and / C scalar_if; R\*×p_gate; 0.90·min(A,B) rule |
| 3. First-cut grid choose | Design §3.3 | G∈{0,16,64} × p∈{25,50,95} × A/B/C (=27) |
| 4. Kernel land | `kernels_ptodsl/cf7d_outer_inner_if.py` | Module-level `@pto.jit`; C kernels per `(G,n_trips)`; A/B always `range(B)` |
| 5. Env blocker | RESULT header | `source ~/projects/env.sh` kills AIV; fix = DEPS+MLIR_PY + ASC/set_env only |
| 6. Oneshot measure | `oneshot_ptodsl_cf7d.sh` → peanut `/tmp/st_simtvf_parallel/ptodsl_cf7d/` | 27/27 PASS; VF/body/µs → `cf7d_SUMMARY_firstcut.tsv` |
| 7. RESULT write-up | `ST_CF7D_FIRSTCUT_RESULT.md` (~20:08 HKT 2026-10-30) | A≡B; crossover G=0; p95 non-monotone at G=16/64 |
| 8. ANALYSIS v1 | `ST_CF7D_ANALYSIS.md` (2026-10-31 ~14:11) | Spine + SEMANTICS_GATE INCONCLUSIVE (no dumps on box) |
| 9. Lean dump sync | pto-b10 pack → `cf7d_lean_sync.tgz` → `opsim_cf7d_lean/` | 27× (`instr_exe.csv` + `trace.json` + `instr_log.dump` + `instr_popped_log.dump`) |
| 10. PR9 overlays | `make_pr9_running_ipc_overlay.py` ×27 | all `plot_style=opcode_lanes_ipc` under `reports/cf7d_plots/` |
| 11. Opcode / ISA-mix harvest | `cf7d_OPCODE_MIX.md` + `cf7d_isa_mix_perN.json` | A≡B opcode Counters; zero SIMT_BRANCH; A vs C bridge |pred/meas−1|=0 |
| 12. ANALYSIS v2 + gate | this TRACE / ANALYSIS ~14:22 HKT | SEMANTICS_GATE **PASS** per *d arm; status **PARTIAL** (not suite DONE) |

---

## Semantic lock (short)

- Gold: outer-false → no kill / no fat G side effects; outer-true → thresh kill (+ optional scale×G).
- Primary metric: opsim VF_cycles (not µs alone).
- Fail-mode observed: **B≡A (VF + opcode), C wins** (AND-merge as-lowered does not cut issue).
- CF path: **predicated VCMP/VSEL**, not VF-internal Sreg if-else.

---

## Stop / open

- First-cut *d dumps+PR9+gate **closed for A1–A4 first-cut scope**.
- Fill-in grid + Simt CF7 + SoftLib (if Aims) + PR = open (Lok yes required for merge).
- SoftLib≻SIMD both-frontend enforce = N/A until Simt/SoftLib arm exists.
