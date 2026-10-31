# ST ANALYSIS — CF7d Outer-Inner-If (PTO-DSL *d)

**ST id:** `cf7d`  
**Date:** 2026-10-31 ~14:22 HKT (Asia/Hong_Kong) · dumps synced + PR9 + SEMANTICS_GATE refresh  
**Frontend primary:** PTO-DSL / VMI Layer-D (`*d`) — arms **A / B / C**  
**Sources (numbers only from disk):**
- Design: `reports/ST_CF7_OUTER_INNER_IF_DESIGN.md`
- RESULT: `reports/ST_CF7D_FIRSTCUT_RESULT.md`
- TSV: `reports/cf7d_SUMMARY_firstcut.tsv`
- Kernel: `kernels_ptodsl/cf7d_outer_inner_if.py`
- Oneshot: `oneshot_ptodsl_cf7d.sh`
- **Lean dumps (box):** `opsim_cf7d_lean/opsim_cf7d_*` ← sync from peanut `/tmp/st_simtvf_parallel/ptodsl_cf7d/` (27 cells: `instr_exe.csv` + `trace.json` + `core0.veccore0.instr_{log,popped}_log.dump`)
- Opcode mix: `reports/cf7d_OPCODE_MIX.md` · `reports/cf7d_opcode_family_firstcut.tsv` · `reports/cf7d_plots/cf7d_isa_mix_perN.json`
- PR9 plots: `reports/cf7d_plots/*_running_ipc_pr9.png` (**27/27**, `plot_style=opcode_lanes_ipc`)
- Machine (RESULT): pto-b10 · Ascend950PR_9599 opsim · `.camodel_deps_vmi018`

**Status:** **PARTIAL / first-cut *d complete** — 27/27 PASS + dumps + PR9 + SEMANTICS_GATE **PASS** for A/B/C *d arms. **Not** suite / MB DONE (fill-in grid + SimtVF CF7 open; SoftLib not Aims). **Not** SoftLib≻SIMD.

**Clean-run:** No invented el/cyc / IPC. VF / wall_us from TSV+RESULT; body/opcode from lean `instr_exe.csv` (framing-excluded body; see §4 note vs oneshot body). el/cyc = `E / VF_cycles` with **E=256**.

---

## 1. Aims

| Id | Aim |
|----|-----|
| **A1** | On an `R* × p_gate` grid, find the **crossover** where outer **scalar/host trip skip (arm C)** beats outer **slack/predicate** paths (A/B) in **opsim VF cycle** (primary; µs secondary only). Crossover = smallest `R*` (prefer `G`) at fixed `p_gate` s.t. `VF(C) ≤ 0.90 × min(VF(A), VF(B))`. |
| **A2** | Test whether **arm B (slack + AND-merge)** closes most of the A→C gap vs **arm A (pure_slack)** at the opcode / VF level. |
| **A3** | Confirm design signatures: **A flat vs `p_gate`**; **C tracks `p_gate`** (VF + body_instr scale with `n_true`); gold PASS when outer-false trips do not apply inner kill / fat side effects. |
| **A4** | Land **first-cut** PTO-DSL grid `G ∈ {0,16,64}` × `p_gate ∈ {25,50,95}` × arms `{A,B,C}` (=27 cells) before fill-in `G∈{8,32}` / `p∈{5,75}` and before SimtVF CF7 (*v). |

Lineage (motif tags only; **do not** copy CF1–CF6 KPI into CF7d cells): CF1–CF3 Parallel-only `if`; CF4 nested predicates same loop; CF5/CF6 divergence / Newton — **none** cover outer-loop dynamic `if` × inner-loop dynamic `if` with AND-merge of outer gate into inner mask (`ST_CF7_OUTER_INNER_IF_DESIGN.md` §1).

---

## 2. Test-case design

| Knob | Value | Role |
|------|-------|------|
| `E` | 256 | elems (match CF1–CF6) |
| `B` | 8 | outer trip count |
| `VL` | 64 | PTO-DSL vector length (`N_CHUNKS = E/VL = 4`) |
| `G` | {0, 16, 64} first-cut | fat inner serial under outer-true; **R\* = G when G>0**; G=0 → thin R\*≈E/VL=4 chunks / trip |
| `p_gate` | {25, 50, 95} first-cut | host fraction outer-true (`n_true = round(B·p/100)`) |
| dtype | f32 | scores / thr / gate / out |
| Gold | `allclose(..., equal_nan=True)` | outer-false → no kill / no fat `G` scale; outer-true → CF1-style kill (+ optional `SCALE` loop `G` times) |

**Arms (same gold):**

| Arm | Outer gate | How outer reaches inner | Tag |
|-----|------------|-------------------------|-----|
| **A · pure_slack** | always enter; `vcmp` gate → `vsel` after full inner kill | outer **not** AND'd into kill mask; kill always issues, then `vsel(p_outer, k, s)` | `_armA` |
| **B · slack_and** | always enter; intent = AND outer into inner | as-landed *d: kill `vcmp+vsel` then outer `vsel(p_outer, …)` (**A≡B** on VF **and** full opcode Counter) | `_armB` |
| **C · scalar_if** | host/compile-time trip skip | `n_trips = n_true`; only true outer trips unrolled into kernel | `_armC` |

Tag scheme: `cf7d_e{E}_b{B}_g{G}_vl{VL}_pgate{P}_arm{A\|B\|C}`.

**Pass criteria (cell):** status PASS + gold; crossover uses **VF cycle** only (Lok).

---

## 3. Layout and schedule (SIMT and SIMD)

### 3.1 SIMD PTO-DSL (*d) — measured

- **Layout:** GM→UB MTE2 load scores/thr/gate; 4× VL=64 RF chunks; per outer trip: load thr/gate scalar→`vbrc`/`vdup`, outer `vcmp`, inner kill `vcmp+vsel` to `-inf`, optional fat `G` scale (`vmul`×SCALE under true path); store via MTE3.
- **Arm A/B:** Python `for b in range(8)` always — full `B` trips in VF body.
- **Arm C:** separate `@pto.jit` kernels per `(G, n_trips)` with `n_trips ∈ {0,2,4,6,8}` matching `n_true` for pgate — **false outer trips omitted from the compiled body**.
- **Era:** `era=cf7d-firstcut-ptodsl` (control-flow policy; not Case-3 / absmax Gmap).
- **OPPROF signature (all 27 dumps):** body = `RV_VCMP_GT` + `RV_VSEL` (+ `RV_PSET`) predicated/select path; **zero** `SIMT_*` / `SIMT_BRANCH` / scalar `BR`/`JUMP` in any cell (suite-wide opcode scan).

### 3.2 SIMT Ascend default — **MISSING**

No SimtVF / `T.Parallel` CF7 (*v) land in this cut (RESULT §3.4; Aims A4 explicitly *before* *v*).  
**Next-cmd:** design+land Simt CF7 arms A/B/C per `ST_CF7_OUTER_INNER_IF_DESIGN.md` §2; oneshot twin TBD (no `oneshot_cf7.sh` yet).

### 3.3 SIMT SoftLib hand-craft — **MISSING**

Not in Aims for first-cut *d rooftop.  
**Next-cmd:** only if Aims add SoftLib SIMT A5 for CF7 — then SoftLib schedule ladder + dumps.

---

## 4. SIMD PTO-DSL performance (primary)

**Primary KPI:** opsim `VF_cycles` (VF / VF_SIMT row). Secondary: wall µs, `body_instr`.  
**el/cyc** = `E / VF_cycles` = `256 / VF` (label: elements / VF_cyc).

**Body note:** oneshot TSV `body_instr` (e.g. A G0=215) used a **case-sensitive** framing filter and still counted `SET_FLAG`/`WAIT_FLAG`/… . Dump harvest in `cf7d_OPCODE_MIX.md` excludes framing → A G0 body=**204**. VF cycles identical either way. Prefer dump body for mix/IPC; TSV body kept for RESULT continuity.

### 4.1 First-cut grid (all 27 PASS)

From `cf7d_SUMMARY_firstcut.tsv` / `ST_CF7D_FIRSTCUT_RESULT.md`:

| G | p_gate | A VF / µs / body† | B VF / µs / body† | C VF / µs / body† | C ≤ 0.90·min(A,B)? |
|--:|-------:|------------------|------------------|------------------|:------------------:|
| 0 | 25 | **125** / 0.89 / 215 | **125** / 0.89 / 215 | **69** / 0.89 / 94 | **YES** (thr 112.5) |
| 0 | 50 | 125 / 0.89 / 215 | 125 / 0.89 / 215 | **77** / 0.85 / 114 | **YES** |
| 0 | 95 | 125 / 0.89 / 215 | 125 / 0.89 / 215 | **96** / 0.90 / 154 | **YES** |
| 16 | 25 | **643** / 1.17 / 727 | **643** / 1.17 / 727 | **189** / 0.96 / 226 | **YES** (thr 578.7) |
| 16 | 50 | 643 / 1.17 / 727 | 643 / 1.17 / 727 | **342** / NA / 374 | **YES** |
| 16 | 95 | 643 / 1.17 / 727 | 643 / 1.17 / 727 | **634** / 1.19 / 670 | **NO** |
| 64 | 25 | **2161** / 1.98 / 2263 | **2161** / 1.98 / 2263 | **573** / 1.15 / 610 | **YES** (thr 1944.9) |
| 64 | 50 | 2161 / 1.98 / 2263 | 2161 / 1.98 / 2263 | **1111** / 1.45 / 1142 | **YES** |
| 64 | 95 | 2161 / 1.98 / 2263 | 2161 / 1.98 / 2263 | **2171** / 2.01 / 2206 | **NO** (C ≳ A) |

† oneshot TSV body (includes framing opcodes). Dump framing-excluded: A/B G0=204, G16=716, G64=2252; C G0 p25=83.

### 4.2 el/cyc (derived from measured VF; E=256)

| G | p_gate | A el/cyc | B el/cyc | C el/cyc |
|--:|-------:|---------:|---------:|---------:|
| 0 | 25 | 2.048 | 2.048 | **3.710** |
| 0 | 50 | 2.048 | 2.048 | 3.325 |
| 0 | 95 | 2.048 | 2.048 | 2.667 |
| 16 | 25 | 0.398 | 0.398 | **1.355** |
| 16 | 50 | 0.398 | 0.398 | 0.749 |
| 16 | 95 | 0.398 | 0.398 | 0.404 |
| 64 | 25 | 0.118 | 0.118 | **0.447** |
| 64 | 50 | 0.118 | 0.118 | 0.230 |
| 64 | 95 | 0.118 | 0.118 | 0.118 |

### 4.3 Crossover call (A1) — first-cut

| p_gate | crossover `G` | notes |
|-------:|--------------:|-------|
| 25 | **0** | C wins every first-cut G |
| 50 | **0** | C wins every first-cut G |
| 95 | **0** | wins only at G=0; **fails** 0.90-bar at G=16 / G=64 (p95 non-monotone in G) |

**Call:** first-cut **crossover R\* = G = 0** for all measured `p_gate`. Host/compile-time trip skip (C) already beats always-enter slack (A/B) on thin kill-only work — not a “need large R\*” story. At `p_gate=95%`, more fat work under almost-always-true outer → C ≈ A (nothing left to skip).

### 4.4 Arm findings (A2 / A3)

1. **A ≡ B** on VF **and** full opcode Counter at every cell (9/9) → AND-merge encoding does **not** reduce issue vs pure_slack in this *d shape (fail-mode: **B≈A, C wins**).
2. **A flat vs p_gate** (125 / 643 / 2161 by G only) — pure slack; expected.
3. **C tracks p_gate** (VF + body drop as `n_true`↓) — real skip of false outer trips.
4. Gold: 27/27 PASS; G>0 `maxabs` may print `nan` on `-inf` kill lanes (cosmetic; `equal_nan=True`).
5. **Control-flow path:** predicated / select (`RV_VCMP_GT`+`RV_VSEL`) — **not** scalar Sreg if-else inside VF.

### 4.5 SEMANTICS_GATE (per arm · before any DONE/SoT)

CF7d is **outer/inner if control-flow**, not SoftLib≻SIMD reduce. No SoftLib≻SIMD claim. No suite DONE.

```
SEMANTICS_GATE: PASS — armA pure_slack — S1 gold PASS; S2 VF flat vs p_gate + always-B trips; S3 OPPROF = VCMP_GT/VSEL/PSET (no SIMT_BRANCH/Sreg BR); S4 VF=125/643/2161 named; PARTIAL suite (fill-in/Simt open), not MB DONE
SEMANTICS_GATE: PASS — armB slack_and — S1 gold PASS; S2/S3 identical opcode Counter to A (AND-merge as-lowered ineffective — documented fail-mode, not wrong-arm for G-sensitivity); S4 VF≡A; not DONE as "AND recovers skip"
SEMANTICS_GATE: PASS — armC scalar_if — S1 gold PASS; S2/S4 VF/body track n_true; S3 fewer VCMP/VSEL (G0p25: VSEL 8 vs A 64); crossover G=0 first-cut call only; not suite DONE
```

---

## 5. SIMT (Ascend default) lowering performance

**MISSING** — no CF7 SimtVF (*v) measurement this cut.  
**Next-cmd:** land Simt CF7 A/B/C + harvest VF/body (mirror CF1–CF6 oneshot pattern); do not copy CF1–CF6 KPI into CF7d cells.

---

## 6. SIMT hand-craft (SoftLib) analysis

**MISSING** — not measured; not first-cut Aims.  
**Next-cmd:** only if Aims add SoftLib SIMT for CF7 — then SoftLib schedule + opsim dumps.

---

## 7. Running IPC + PR9 overlay figures

**Trigger note:** This ANALYSIS compares **PTO-DSL arms A/B/C** (same frontend), **not** SoftLib≻SIMD / SIMT≻SIMD. SoftLib≻SIMD PR9-both-frontends bar = **N/A**. Intra-*d PR9 still applied for every measured cell with a dump.

**Status:** **27/27** `*_running_ipc_pr9.png` under `reports/cf7d_plots/` — each with sidecar JSON proving `plot_style=opcode_lanes_ipc` + non-empty `lane_opcodes` (reject IPC-only). Source dump: `core0.veccore0.instr_popped_log.dump` via `make_pr9_running_ipc_overlay.py --repo /workspace/tilelangGym-pr8-tip`.

| Cell (examples) | mid-med IPC (W=20) | lane_opcodes |
|-----------------|-------------------:|--------------|
| G0 p25 A / B | **1.85** | VLDI, VDUP, VSTI, VCMP_GT, VSEL |
| G0 p25 C | **1.05** | same lanes (shorter issue stream) |
| G0 p50 C | 1.50 | same |
| G0 p95 C | 1.85 | same |
| G16/G64 A (fat) | ~1.00–1.05 | + **VMUL** lane |
| G16/G64 C | ~1.00 | + VMUL |

Cmd used:
```bash
python3 REGRESSION_salvage_2026-09-17/scripts/make_pr9_running_ipc_overlay.py \
  --dump opsim_cf7d_lean/opsim_<tag>/core0.veccore0.instr_popped_log.dump \
  --out-png reports/cf7d_plots/<tag>_running_ipc_pr9.png \
  --title "cf7d / <tag> — running IPC (W=20)" --E 256 \
  --repo /workspace/tilelangGym-pr8-tip
```

---

## 8. Per-segment opcode mix + stage mid-IPC

**SoftLib≻SIMD / STAGE_TAXONOMY reduce map:** **N/A** this ST (CF policy, not absmax/Case-3). Honest substitute = **family segments from `instr_exe.csv`** (balanced call_count; see `cf7d_OPCODE_MIX.md`).

### 8.1 Suite CF signature

- **Zero** `SIMT_*` / `SIMT_BRANCH` / scalar `BR`/`JUMP` across all 27 dumps.
- Dominant body ops: `RV_VMUL` (fat G), `RV_VSEL`, `RV_VCMP_GT`, `RV_VLDI`, `RV_VDUP`, scalar addr tax.

### 8.2 A ≡ B (opcode Counter)

All 9 (G,p) pairs: A body Counter **==** B body Counter (YES). Confirms A2 fail-mode at S3, not only VF.

### 8.3 Balanced mix — G0 p25 (headline crossover cell)

| opcode | A call_count | C call_count |
|--------|-------------:|-------------:|
| `RV_VSEL` | 64 | 8 |
| `RV_VCMP_GT` | 40 | 8 |
| `RV_VLDI` | 20 | 6 |
| `RV_VDUP` | 16 | (in C: via VDUPS/VDUP small) |
| `RV_VSTI` | 4 | 4 |
| … | … | … |
| **body SUM (framing-excl.)** | **204** | **83** |
| VF | 125 | 69 |

Full per-opcode tables + balance PASS: `reports/cf7d_OPCODE_MIX.md`.

### 8.4 Family cycle-share proxy (not STAGE_TAXONOMY mid-IPC)

| segment proxy | A G0p25 ops | C G0p25 ops | note |
|---------------|------------:|------------:|------|
| pred_select (VCMP/VSEL/PSET) | 105 | 17 | C skips false outer trips |
| load / store | 20 / 4 | 6 / 4 | |
| eltwise (VMUL) | 0 | 0 | G=0 thin |
| bcast | 18 | 3 | |
| scalar_tax + other | 54 | 50 | roughly fixed prologue/epilogue |

**Stage mid-IPC (STAGE_TAXONOMY S_load/S_reduce/…):** **MISSING / N/A** — taxonomy is reduce SoftLib/SIMD SoT; CF7d has no `S_reduce` defining-opcode map. Do not invent sequential walls. PR9 mid-med IPC (body window) is the rolling figure for this ST.

---

## 9. ISA mix SIMT vs SIMD (per N×4B → el/cyc bridge)

**SIMT vs SIMD both-frontend:** **MISSING / N/A** — no Simt / SoftLib arm; SoftLib≻SIMD trigger off.  
**Intra-*d A vs C ISA-mix (measured):** `reports/cf7d_plots/cf7d_isa_mix_perN.json` + `cf7d_OPCODE_MIX.md` §ISA-mix.

Unit: **N=32 · E=256 · units=8 · IPC=body_calls/VF_cyc**.

| Cell | A el/cyc | C el/cyc | meas A/C | pred_ratio (IPC×ops) | |pred/meas−1| | pred_select ops/N A→C | eltwise ops/N A→C |
|------|---------:|---------:|---------:|---------------------:|-------------:|----------------------:|------------------:|
| G0 p25 | 2.048 | **3.710** | 0.552 | 0.552 | **0.000** | 13.125 → 2.125 | 0 → 0 |
| G16 p25 | 0.398 | **1.355** | 0.294 | 0.294 | **0.000** | 13.125 → 2.125 | 64 → 16 |
| G64 p25 | 0.118 | **0.447** | 0.265 | 0.265 | **0.000** | 13.125 → 2.125 | 256 → 64 |
| G64 p95 | 0.118 | 0.118 | 1.005 | 1.005 | **0.000** | 13.125 → 8.125 | 256 → 256 |

**Attribution:** C's win is **ops/N collapse** (skip false outer trips → fewer pred_select + proportional eltwise), not a large IPC density miracle. At G64 p95, eltwise identical → C cannot beat A on VF.

**Next-cmd (SIMT≻SIMD bar):** when Simt CF7 lands — `harvest-isa-mix --norm-elems 32` for SoftLib/SIMT vs CF7d C rooftop + PR9 both frontends.

---

## 10. Conclusion

| Arm | Role | First-cut headline |
|-----|------|--------------------|
| **SIMD PTO-DSL A pure_slack** | always-enter slack baseline | VF flat vs p_gate: **125 / 643 / 2161** at G=0/16/64; OPPROF VCMP/VSEL |
| **SIMD PTO-DSL B slack_and** | AND-merge policy under test | **A≡B** every cell (VF + opcode Counter) — AND does not cut issue |
| **SIMD PTO-DSL C scalar_if** | host/compile trip skip (best *d) | Best measured arm; crossover **G=0**; G0 p25 VF **69** (el/cyc **3.71**); p95 non-monotone at fat G |
| **SIMT Ascend default** | — | **MISSING** |
| **SIMT SoftLib** | — | **MISSING** |

**Ratios (measured *d only, VF):** at G0 p25, `VF(A)/VF(C) = 125/69 ≈ 1.81`; at G64 p25, `2161/573 ≈ 3.77`; at G64 p95, `2161/2171 ≈ 0.995` (C not better).

**Do not claim:** SoftLib≻SIMD; suite / MB DONE; B recovers skip without Sreg (evidence opposite); Sreg scalar if-else inside VF (OPPROF contradicts).

**Still open:** fill-in grid `G∈{8,32}` × `p∈{5,75}`; Simt CF7 (*v); SoftLib only if Aims add A5; PR (needs Lok yes).

---

## 11. Trace pointer

`reports/ST_CF7D_EXPLORATION_TRACE.md` — reconstruct + dump-sync / PR9 / gate refresh (2026-10-31).

---

## After RESULT — COMPARE

### KPI COMPARE harvest

| vs | Status |
|----|--------|
| Prior dig / REGRESSION-2 SoftLib≻SIMD el/cyc | **KPI N/A** — CF7d is new outer×inner CF ST; no prior CF7 KPI to regress |
| CF1d–CF6d wall µs tables | **N/A for cells** — lineage/motif only; do not paste into CF7d grid |

### Motif-seed Coverage COMPARE

| Motif / recipe tag | Coverage | Note |
|--------------------|:--------:|------|
| `outer_inner_if` nest | ✓ | design §2 + kernel landed |
| `armA_pure_slack` | ✓ | measured 9/9 first-cut cells |
| `armB_slack_and` | ✓ | measured; **stop** on efficacy — A≡B (opcode-identical) |
| `armC_scalar_if` / host `n_trips=n_true` | ✓ | measured; crossover G=0 |
| `R*_x_pgate` first-cut grid G∈{0,16,64}×p∈{25,50,95} | ✓ | 27/27 PASS |
| `crossover_VF_0.90` rule | ✓ | called; p95 non-monotone caveat |
| Opcode-bucket / OPPROF dumps on box | ✓ | `opsim_cf7d_lean/` + `cf7d_OPCODE_MIX.md` |
| PR9 `opcode_lanes_ipc` | ✓ | 27/27 `*_running_ipc_pr9.png` |
| Intra-*d ISA-mix A vs C | ✓ | bridge |pred/meas−1|=0 |
| Fill-in G∈{8,32} / p∈{5,75} | ✗ / stop | RESULT “Not done” |
| SimtVF CF7 (*v) | ✗ / stop | not landed (Aims A4 before *v) |
| SoftLib SIMT CF7 | ✗ / stop | not Aims this cut |
| SoftLib≻SIMD both-frontend PR9/ISA-mix | N/A | no both-frontend claim |
| AND-merge lowers distinct issue vs A | ✗ | measured fail-mode B≡A |

---

## Checklist (st-report)

- [x] Aims `A1…` cited
- [x] SEMANTICS_GATE one-liner per arm (**PASS** *d; suite still PARTIAL)
- [x] No SoftLib≻SIMD / no Case-3 pack claim
- [x] After RESULT: KPI COMPARE N/A + motif Coverage COMPARE
- [x] PR9 `opcode_lanes_ipc` for all 27 measured *d cells (not IPC-only)
- [x] Per-opcode mix balanced + A≡B + CF signature (no Sreg branch)
- [x] Intra-*d ISA-mix → el/cyc bridge (SIMT≻SIMD bar N/A + next-cmd)
- [x] TRACE linked
- [ ] Suite / MB DONE — **forbidden** this cut (**PARTIAL** — fill-in + SimtVF open)
- [ ] Merge / PR — **needs Lok yes** (not done here)
