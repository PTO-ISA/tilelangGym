# ST-CF7d first-cut RESULT — A/B/C × G × p_gate (VF cycle primary)

**Date:** 2026-10-30 ~20:08 HKT (Asia/Hong_Kong)  
**Machine:** pto-b10 · Ascend950PR_9599 opsim · `.camodel_deps_vmi018`  
**Kernel:** `kernels_ptodsl/cf7d_outer_inner_if.py` (PTO-DSL *d only; module-level `@pto.jit`)  
**Oneshot:** `oneshot_ptodsl_cf7d.sh`  
**Raw TSV:** `reports/cf7d_SUMMARY_firstcut.tsv`  
**Design:** `reports/ST_CF7_OUTER_INNER_IF_DESIGN.md`  
**Out:** `/tmp/st_simtvf_parallel/ptodsl_cf7d/` on pto-b10  

**Env fix (blocker):** do **not** `source ~/projects/env.sh` — it pollutes PYTHONPATH/LD and kills AIV `block_start` (CF5d + CF7d oh=0 / GetKernelAddr). Match Oct-8 PASS recipe: `PYTHONPATH=DEPS:MLIR_PY` + `ASC/set_env.sh` only. Keep `PTOAS_MSPROF_PRIVATE_ROOT=/tmp/ptoas-msprof` (msprof soft-link).

---

## 1. First-cut grid — all 27 PASS

`G ∈ {0,16,64}` × `p_gate ∈ {25,50,95}` × arms `{A,B,C}`.

| G | p_gate | A VF / µs | B VF / µs | C VF / µs | C ≤ 0.90·min(A,B)? |
|--:|-------:|-----------|-----------|-----------|:------------------:|
| 0 | 25 | **125** / 0.89 | **125** / 0.89 | **69** / 0.89 | **YES** (thr 112.5) |
| 0 | 50 | 125 / 0.89 | 125 / 0.89 | **77** / 0.85 | **YES** |
| 0 | 95 | 125 / 0.89 | 125 / 0.89 | **96** / 0.90 | **YES** |
| 16 | 25 | **643** / 1.17 | **643** / 1.17 | **189** / 0.96 | **YES** (thr 578.7) |
| 16 | 50 | 643 / 1.17 | 643 / 1.17 | **342** / NA | **YES** |
| 16 | 95 | 643 / 1.17 | 643 / 1.17 | **634** / 1.19 | **NO** (634 > 578.7) |
| 64 | 25 | **2161** / 1.98 | **2161** / 1.98 | **573** / 1.15 | **YES** (thr 1944.9) |
| 64 | 50 | 2161 / 1.98 | 2161 / 1.98 | **1111** / 1.45 | **YES** |
| 64 | 95 | 2161 / 1.98 | 2161 / 1.98 | **2171** / 2.01 | **NO** (2171 ≳ A) |

`body_instr` (harvest): A/B flat vs p_gate at fixed G; C scales with `n_true` (pgate). Example G0: A/B body=215; C = 94 / 114 / 154 for p25/50/95.

Gold: `allclose(..., equal_nan=True)`. For `G>0`, `maxabs` prints `nan` because both oh/ref hold `-inf` kill lanes → `|-inf−(-inf)|=nan` (cosmetic; PASS valid).

---

## 2. Crossover call (primary = VF cycle)

**Definition:** smallest `R*` (prefer `G`) s.t. `VF_cycle(C) ≤ 0.90 × min(VF_cycle(A), VF_cycle(B))`.

| p_gate | crossover `G` (=R* when G>0; G=0 → thin R*≈E/VL) | notes |
|-------:|--------------------------------------------------:|-------|
| 25 | **0** | C wins at every first-cut G |
| 50 | **0** | C wins at every first-cut G |
| 95 | **0** | wins only at G=0; **fails** at G=16 and G=64 (C≈A; at G64 C slightly **worse**) |

**Call:** first-cut **crossover R* = G = 0** for all measured `p_gate`.  
Not the “need large R*” story — **host/compile-time trip count skip (arm C)** already beats always-enter slack (A/B) on thin kill-only work. At `p_gate=95%` the 0.90 bar is **non-monotone in G**: more fat work under almost-always-true outer makes C ≈ A (nothing to skip).

µs secondary: same qualitative story; do **not** use µs to declare crossover.

---

## 3. Arm findings (fail-mode taxonomy)

1. **A ≡ B** on VF cycle **and** body_instr at every cell → AND-merge encoding does **not** reduce issue count vs pure_slack in this *d shape (both: always `B` trips; kill `vcmp+vsel` then outer `vsel(p_outer, …)`). Fail-mode: **B≈A, C wins** — only scalar/host trip skip helps.
2. **Arm A flat vs p_gate** (125 / 643 / 2161 by G only) — outer is pure slack; expected.
3. **Arm C tracks p_gate** (VF and body drop as `n_true`↓) — real skip of false outer trips (compile-time `n_trips=n_true` unrolled range).
4. No SimtVF / `T.Parallel` (*v) in this land — rooftop *d only.

---

## 4. Paths

| Item | Path |
|------|------|
| Kernel | `/workspace/st_simtvf_parallel/kernels_ptodsl/cf7d_outer_inner_if.py` (+ WT under tilelang-st-simtvf-parallel) |
| Oneshot | `oneshot_ptodsl_cf7d.sh` |
| SUMMARY | `reports/cf7d_SUMMARY_firstcut.tsv` · remote `/tmp/st_simtvf_parallel/ptodsl_cf7d/SUMMARY.tsv` |
| Logs | `/tmp/st_simtvf_parallel/ptodsl_cf7d/*.log` · `opsim_cf7d_*` |
| This RESULT | `reports/ST_CF7D_FIRSTCUT_RESULT.md` |

**Not done:** full grid `G∈{8,32}` × `p_gate∈{5,75}`; Simt CF7 (*v); PR; opcode-bucket heatmaps beyond VF/body harvest.

---

## 5. Status

**First-cut SUCCESS** — 27/27 PASS, VF+µs table above, crossover **G=0** (with p95 non-monotone caveat).
