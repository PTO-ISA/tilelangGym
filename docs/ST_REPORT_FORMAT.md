# tilelangGym ST report format

Write or rewrite one ST `ANALYSIS.md` (and supporting notes) so every ST follows the same SV1-style report spine. Keep numbers measured; never invent el/cyc, IPC, or rooftop %.

## When to use
- New ST ANALYSIS (SV*, CF*, SP*) on Ascend SimtVF / SIMD / SoftLib
- Rewriting an existing ST report to match this format
- Reviewing whether a report is complete before PR merge

## Confirm with the author first
Before rewriting a report, **ask and confirm Aims** (section 1). Propose a short draft Aims list for that ST; do not invent aims the author did not approve. Also confirm if any section should be skipped (e.g. no SoftLib yet).

## Report spine (required section order)
Use these top-level sections in order. Subsections may expand, but do not reorder or rename the spine.

### 1. Aims
What this ST proves or measures (author-confirmed).

### 2. Test-case design
Kernels, E/TILE/T (or VL/G), datatypes, pass criteria, case matrix.

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

### 7. Conclusion
SIMD best, SIMT Ascend default, SIMT SoftLib hand-craft; summary table and **performance ratios** (SIMD ÷ SIMT arms).

## Naming and metrics conventions
- Prefer `TILE8192` (not untiled / TILE8K) when E and TILE are aligned SoT cases; keep multi-TILE ladders when sensitivity needs them.
- el/cyc = elements / VF cycles (state which VF).
- Mid-avg = interior window (document window width, e.g. W=20); overall = whole-kernel.
- Rooftop % always vs the binding resource named in that section (EXU, UB, serial stage roof, etc.).
- Plots: embed running-IPC (and stage) PNGs inline in ANALYSIS when they exist under the ST folder.

## Repo placement
- Primary: `benches/<st>/ANALYSIS.md` (e.g. `benches/sv5/ANALYSIS.md`)
- Supporting notes may live beside it (`SIMD_RUNNING_IPC.md`, rooftop notes) but ANALYSIS must stand alone with the spine above.
- This file (`docs/ST_REPORT_FORMAT.md`) is the shared reviewer-facing copy of the format; keep it in sync with the bot skill recipe.

## Workflow
1. Confirm Aims (+ optional section skips) with the author.
2. Inventory existing numbers, plots, SoftLib sources for that ST; mark gaps (do not fabricate).
3. Rewrite ANALYSIS to the spine; move/keep plots under the ST; update theory roofs to match chosen layout/instr.
4. Land via the usual ST push path (pto-b10 peanut when CloudAgent SCM is blocked); tip SHA + PR URL back.
5. Ask before merge.

## Out of scope
- Changing kernel semantics or inventing new SoftLib numbers without a measured run
- Merging the PR without an explicit author yes
