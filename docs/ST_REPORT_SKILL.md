# tilelangGym ST report

**Contract only.** This file is the writing rules for one ST `ANALYSIS.md`. It carries no measured numbers, no host paths, no prior dig notes, and no agent memory. Inputs for a run are only: ST id, Aims, and an optional design prompt. Everything measured comes from the harness (`scripts.st_report`).

Open-code / peanut: `cat docs/ST_REPORT_SKILL.md` then run the harness CLI. Do not rely on chat history.

## Clean-run rule
- Load **this file** each time; do not rely on chat history or bot profile for ST lore.
- Do not invent el/cyc, IPC, or rooftop %.
- Trail / sensitivity probes stay out of the main ANALYSIS unless the author asks to keep them.
- Do not merge without an explicit author yes (process rule, not harness).

## Inputs (only)
1. **ST id** — e.g. `sv5`
2. **Aims** — author-confirmed list of what this ST proves
3. **Design prompt** (optional) — kernels, E/TILE/T or VL/G, datatype, arm hints

If Aims are missing or arms are ambiguous, **ask** before writing. Prefer asking over stuffing trail experiments.

## Conclusion arms (main report)
Keep **mainly 3 arms** (4 only when SoftLib SIMD is meaningfully distinct from PTO-DSL best):
1. **SIMD PTO-DSL best** — if DSL cannot reach the best schedule, also mention (or promote) **handcraft SIMD SoftLib / CCE** as the SIMD SoT
2. **SIMT Ascend default** lowering
3. **SIMT SoftLib hand-craft**
4. *(optional)* **SIMD SoftLib** when it is a meaningful fourth case vs DSL best

Drop FA / Fallback-A twins, legacy paths, bank-cliff stock, membar diagnostics, and other intermediate probes from headline tables and conclusion unless the author asks to keep them.

## Report spine (required order)
1. **Aims**
2. **Test-case design** — case matrix for conclusion arms only
3. **Layout and schedule (SIMT and SIMD)**
4. **SIMD PTO-DSL performance** — el/cyc, theory roof, sensitivity, instr mix, running IPC, overall/mid-avg rooftop %
5. **SIMT (Ascend default) lowering performance** — LDS/STS 32 vs 64, T-dim sensitivity
6. **SIMT hand-craft (SoftLib) analysis** — prefer LDS.32/STS.32 when SoT; stage-split LD/compute/ST; stage IPC plots
7. **Conclusion** — 3 (or 4) arms only; summary table and performance ratios

## Naming and metrics
- Prefer `TILE8192` when E and TILE are aligned SoT cases.
- el/cyc = elements / VF cycles (name which VF).
- Mid-avg = interior window (document W, e.g. W=20); overall = whole-kernel.
- Rooftop % vs the binding resource named in that section.
- Embed running-IPC / stage PNGs for kept arms only when they exist under the ST folder.

## Harness handoff (measure layer)
From repo root:

```bash
python -m scripts.st_report init    --st <id> --aims "..." [--design "..."]
python -m scripts.st_report harvest --st <id>
python -m scripts.st_report plot    --st <id>
python -m scripts.st_report gaps    --st <id>
python -m scripts.st_report fill    --st <id> [--dry-run]
python -m scripts.st_report check   --st <id>
```

- `init` writes `benches/<st>/report/arms.yaml` (+ ANALYSIS stub if missing) from Aims/design only.
- `harvest` / `plot` / `gaps` / `fill` / `check` own SUMMARY.tsv, plots, gap report, stub fill, and trail leak detection.
- This skill does **not** replace the harness; the harness does **not** replace this skill.

## Repo placement
- Primary: `benches/<st>/ANALYSIS.md`
- Supporting notes may sit beside it; ANALYSIS must stand alone with the spine above.
- Keep `docs/ST_REPORT_FORMAT.md` in sync with this file (contract + harness pointer only — no dig numbers).

## Workflow
1. Confirm Aims (+ optional section skips) and conclusion arms when ambiguous.
2. `init` → inventory harness gaps (do not fabricate).
3. Rewrite ANALYSIS to the spine around kept arms; roofs match chosen layout/instr.
4. Land via the usual ST push path; tip SHA + PR URL back; ask before merge.

## Out of scope
- Changing kernel semantics or inventing SoftLib numbers without a measured run
- Merging without explicit author yes
- Stuffing trail cases into main ANALYSIS without author confirm
- Putting host paths, ST dig el/cyc values, or agent prefs into this skill file
