# tilelangGym ST report format

Write or rewrite one ST `ANALYSIS.md` (and supporting notes) so every ST follows the same report spine. Keep numbers measured; never invent el/cyc, IPC, or rooftop %.

**Contract only.** Inputs for a run are: ST id, Aims, and an optional design prompt. Measured values come from the harness (`scripts.st_report`), not from this document.

## When to use
- New ST ANALYSIS (SV*, CF*, SP*) on Ascend SimtVF / SIMD / SoftLib
- Rewriting an existing ST report to match this format
- Reviewing whether a report is complete before PR merge

## Confirm with the author first
Before rewriting a report, **ask and confirm Aims** (section 1). Propose a short draft Aims list for that ST; do not invent aims the author did not approve. Also confirm if any section should be skipped (e.g. no SoftLib yet). **Confirm conclusion arms** when unsure which paths belong in the main report focus (see below).

## Conclusion arms (main report focus)
The ANALYSIS is for **conclusion**, not the full intermediate trail.
Keep **mainly 3 arms** (4 only when SoftLib SIMD is meaningfully distinct from PTO-DSL best):
1. **SIMD PTO-DSL best** — if DSL cannot reach the best schedule, also mention (or promote) **handcraft SIMD SoftLib / CCE** as the SIMD SoT
2. **SIMT Ascend default** lowering
3. **SIMT SoftLib hand-craft**
4. *(optional)* **SIMD SoftLib** when meaningful vs DSL best
Drop FA / Fallback-A twins, legacy abs paths, bank-cliff stock, membar diagnostics, and other intermediate probes from headline tables and conclusion unless the author asks to keep them. Supporting notes may retain trail detail; ANALYSIS must not lead with them.

## Report spine (required section order)
Use these top-level sections in order. Subsections may expand, but do not reorder or rename the spine.

### 1. Aims
What this ST proves or measures (author-confirmed).

### 2. Test-case design
Kernels, E/TILE/T (or VL/G), datatypes, pass criteria, case matrix.
**Test-case design and plots focus on conclusion arms only** (see Conclusion arms above); intermediate probes belong in supporting notes if needed.

### 3. Layout and schedule (SIMT and SIMD)
How work maps to UB/MRF, thread/VL, load/store widths, and the intended pipeline stages for both arms.

### 4. SIMD PTO-DSL performance
- Measured el/cyc (and wall if useful)
- Theory-bound / rooftop from the **same** layout + instruction mapping used in the run
- Sensitivity (tile size, virtual VL / G, other named knobs)
- Instruction mix
- Running IPC (windowed) and input/output **element per cycle**
- Overall and mid-avg rooftop %

### 5. SIMT (Ascend default) lowering performance
- Stock Ascend / SimtVF path
- Layout mapping and instruction mix (call out LDS/STS **32 vs 64**)
- Theory-bound analysis from that layout + mapping
- Sensitivity, mainly **thread dim (T)**

### 6. SIMT hand-craft (SoftLib) analysis
- Handwritten CCE / SoftLib
- Choose best layout that is **not** LD/ST-bound when possible
- Prefer LDS.32 / STS.32 when that is the SoT instruction width for the study
- Stage-split LD / compute / ST; find best stage schedule
- Update theory rooftop to the **same** layout mapping + instructions used
- IPC timing plots with stage analysis

### 7. Running IPC + PR9 overlay figures (FORCED enforce list · t774u)

A report is **incomplete** if these are missing without an explicit **MISSING** row + next-cmd. Do **not** ship ANALYSIS/PR as “done” on tables alone.

For **each main conclusion arm** that has an opsim dump (`instr_log.dump` / OPPROF):

1. **Running IPC** — W=20 left-open right-closed `(t−W, t]` (or Body4 firstLDS→lastSTS if mid is IDLE_GAP); cite mid IPC and/or mean_body; never invent.
2. **Harness plot** — run `python -m scripts.st_report plot --st <id> …` (or pack-local equivalent harvest→plot) and embed/link outputs under `benches/<st>/plots/` (or `ipc/`).
3. **PR9/SV1-style figure set** (instr reference diagrams, not prose-only):
   - [ ] **Instr mix / breakdown** figure (useful mem+ALU vs tax)
   - [ ] **Running IPC + opcode-lane issue overlay FORCED** (`plot_style=opcode_lanes_ipc`: top = one lane per opcode kind, marks = issues; bottom = rolling W=20 on the same tick axis). Filename `*_running_ipc_pr9.png`. **Reject** IPC-only `*_running_ipc.png` (no lanes / no `_pr9`). See harness L6 § "PR9 running-IPC + instr-mix overlay (FORCED)" and `scripts/make_pr9_running_ipc_overlay.py`.
   - [ ] **Rooftop / el/cyc summary** when a named roof applies (else N/A + why)
   - [ ] **Stage mid-IPC FORCED** when SoftLib≻SIMD / both-arm perf compare / mix splits stages: defining-opcode windows; table mid_med W=20 per stage × **both** arms (fused loop → write mid≈body explicitly; do not invent walls)
4. If dump absent: one checklist line **`MISSING`** + exact next-cmd to re-opsim / harvest — **not** silent omit.
5. SoftLib SIMT rows: name **motif** on the figure caption (`axis=1xG` / `axis=Nx1` / `warp-redux` appendix).

**Gate:** Before tip SHA / “ready for review”, verify every main arm has either (plot path + mid IPC) or (MISSING + next-cmd). Tables without this enforce list = incomplete delivery.

### 8. Per-segment opcode mix + stage mid-IPC (FORCED on both-arm compare)

When SoftLib≻SIMD / SIMD≻SIMT / both-arm perf compare: both arms get per-segment opcode mix + IPC + stage mid-IPC (see `tilelanggym-st-report`). Pair with PR9 overlays.

### 9. ISA mix SIMT vs SIMD — per N×4B ops → el/cyc bridge (FORCED · t546u)

Same trigger as §8. Normalize opcode `call_count` to **N elements × 4B** (default **N=32**):

```text
ops_per_N = call_count / (E / N)
pred_el/cyc ≈ IPC × (N / ops_per_N)   # prefer body IPC = body_calls / VF_cyc
pred_ratio  ≈ (IPC_A/IPC_B) × (ops_per_N_B / ops_per_N_A)
```

Family table (load / store / reduce / scale / eltwise / pred_tax) for **both** arms + pred↔measured bridge. Harness: aggregate `instr_exe.csv` (`simtvf-parallel-opsim-report` §6); light harness `python -m scripts.st_report.isa_mix --norm-elems 32 … -o *_isa_mix_perN.json` (CLI alias `harvest-isa-mix` when full `scripts.st_report` lands). Missing without MISSING+next-cmd = incomplete. See skill for family map and 15% pred/meas gate.

### 10. Conclusion

SIMD best, SIMT Ascend default, SIMT SoftLib hand-craft; summary table and **performance ratios** (SIMD ÷ SIMT arms). Stick to the confirmed conclusion arms; do not re-list dropped intermediate probes.

## Naming and metrics conventions
- Prefer `TILE8192` (not untiled / TILE8K) when E and TILE are aligned SoT cases; keep multi-TILE ladders when sensitivity needs them.
- el/cyc = elements / VF cycles (state which VF).
- Mid-avg = interior window (document window width, e.g. W=20); overall = whole-kernel.
- Rooftop % always vs the binding resource named in that section (EXU, UB, serial stage roof, etc.).
- Plots: embed running-IPC (and stage) PNGs inline in ANALYSIS under `benches/<st>/plots/` **for conclusion arms** — see §7 enforce list (MISSING+next-cmd if dump absent).

## Repo placement
- Primary: `benches/<st>/ANALYSIS.md` (e.g. `benches/sv5/ANALYSIS.md`)
- Supporting notes may live beside it (`SIMD_RUNNING_IPC.md`, rooftop notes) but ANALYSIS must stand alone with the spine above.
- This file (`docs/ST_REPORT_FORMAT.md`) is the shared reviewer-facing copy of the format; keep it in sync with `docs/ST_REPORT_SKILL.md` (clean skill contract for open-code).

## Workflow
1. Confirm Aims (+ optional section skips) with the author; **confirm conclusion arms** when unsure.
2. `init` from Aims/design → inventory harness gaps (do not fabricate). Prefer inventory for conclusion arms.
3. Rewrite ANALYSIS to the spine; move/keep plots under the ST; update theory roofs to match chosen layout/instr.
4. After RESULT: COMPARE harvest filled or N/A.
5. **Enforce list FORCED:** PR9 `opcode_lanes_ipc` (`*_running_ipc_pr9.png`) via `make_pr9_running_ipc_overlay.py` / `st_report plot` (or MISSING+next-cmd) for every main arm — **reject** IPC-only PNGs; SoftLib≻SIMD → both arms + stage mid-IPC + ISA-mix per-N×4B → el/cyc bridge.
6. Land via the usual ST push path; tip SHA + PR URL back.
7. Ask before merge.

## Out of scope
- Changing kernel semantics or inventing new SoftLib numbers without a measured run
- Merging the PR without an explicit author yes
- Stuffing trail cases into main ANALYSIS without author confirm
- Putting host paths, dig el/cyc values, or agent prefs into this format doc


## Harness (measured spine)
Machine-readable conclusion arms, SUMMARY, plots / running-IPC, and gap list live under `scripts/st_report/` + `benches/<st>/report/`.
Writing contract remains this document + `docs/ST_REPORT_SKILL.md`; the harness never invents numbers and keeps trail out of ANALYSIS.

```bash
python -m scripts.st_report init       --st <id> --aims "..." [--design "..."]
python -m scripts.st_report harvest    --st <id>
python -m scripts.st_report plot       --st <id> [--window 20]
python -m scripts.st_report plot-ipc   --st <id> [--window 20]   # alias of plot
python -m scripts.st_report gaps       --st <id>
python -m scripts.st_report fill       --st <id> --dry-run
python -m scripts.st_report check      --st <id>
python -m scripts.st_report selftest-ipc   # synthetic fixture only
```

- `init` writes `benches/<st>/report/arms.yaml` from Aims/design (ANALYSIS stub only if missing; `--refresh-aims` updates Aims section only).
- `harvest` owns SUMMARY.tsv (schema stable; mid-IPC filled only from measured sources, never invented).
- `plot` / `plot-ipc` generate running-IPC PNGs from per-arm `artifacts` logs (`popped_dump`, `trace`+`instr_csv`, or `rolling_csv`) with window **W** (default 20, `(t−W,t]`); or wire `running_ipc_png` / known tip maps. Missing inputs → skip + `gaps.md` (no fake PNGs).
- **PR9 overlay gate FORCED (t774u):** SoftLib≻SIMD / both-arm compare / mix-split → PNG **must** be dual-pane `opcode_lanes_ipc` (`*_running_ipc_pr9.png`); IPC-only = FAIL. Recipe: `REGRESSION_salvage_2026-09-17/scripts/make_pr9_running_ipc_overlay.py` or tip `rolling_ipc.build_from_artifacts`. Self-check: `python -m scripts.st_report selftest-ipc`.
- `gaps` / `fill` / `check` own gap report, stub fill, and trail leak detection.
- See `benches/<st>/report/arms.yaml` for that ST's 3–4 arm spine (no dig numbers in this doc).
