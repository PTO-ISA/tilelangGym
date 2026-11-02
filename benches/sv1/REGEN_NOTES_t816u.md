# t816u — restore SoftLib LDS narrative lost in t815u PR #15 regen

**When:** 2026-11-02 HKT (Asia/Hong_Kong) · **PR:** PTO-ISA/tilelangGym#15 · **branch:** sv1-pr9-analysis-regen-t815u · **prior tip:** 6cfd9b9

## Root cause (audit)

1. t815u intent was only to add PR9 opcode_lanes_ipc overlays + SEMANTICS_GATE + ISA-mix/call_count on SV1.
2. The regen replaced benches/sv1/ANALYSIS.md wholesale with a new 154-line spine (~11 KB) instead of editing the main tip doc (314 lines / ~26 KB).
3. The new spine marked SoftLib as N/A for SV1 this run and treated LDS.32/64 as a separate case not measured, which dropped the entire main SoftLib rooftop story (sec 7.1-7.4) plus stock CBT / shared-vs-fragment / head-to-head narrative.
4. PR9 PNG/JSON files were actually present on the branch under benches/sv1/plots/; what looked gone in review was the SoftLib/stock ANALYSIS body (and its embedded legacy IPC narrative). Clean-regress PR9 section existed but SoftLib LDS.32 best + LDS32/LDS64 matrix did not.
5. Box pack glm53 sv1/ANALYSIS.md had already been overwritten with the same thin template, so the richest prior source was origin/main:benches/sv1/ANALYSIS.md.

## Fix (this commit)

- Rebuild ANALYSIS.md from main SoftLib + stock spine (secs 1-9 including SoftLib LDS.32 vs LDS.64, 1-in schedule, dual-input fence table, TILE ladder, head-to-head).
- Append clean-regress t813u SimtVF/SIMD twin tables + PR9 running_ipc_pr9 dual-pane refs + per-segment mix + ISA-mix bridge as secs 10-13 (era-2; no cross-era fold).
- Keep SEMANTICS_GATE one-liner in the header.
- Do not delete legacy IPC-only PNGs.
- Do not invent numbers — all KPI copied from main tip ANALYSIS / t815u regen / harvest JSON already on the branch.

## Section inventory

| source | lines (approx) | SoftLib LDS.32/64 | PR9 running-IPC | SEMANTICS_GATE / ISA-mix |
|--------|---------------:|:-----------------:|:---------------:|:------------------------:|
| main tip | 314 | yes (sec 7.1-7.4) | no (IPC-only only) | no |
| broken PR #15 6cfd9b9 | 154 | no (N/A stub) | yes (sec 7) | yes |
| restored (t816u) | ~474 | yes (sec 7.1-7.4) | yes (sec 11) | yes |

## Confirmation checklist

- SoftLib best A+B LDS.b32 T512 loadall_barrier 7.42 present
- LDS.32 vs LDS.64 BW / roof matrix present (sec 7.1)
- SoftLib TILE ladder / fence x source present (sec 7.2-7.4)
- PR9 sv1_simtvf_running_ipc_pr9.png + sv1_simdt_running_ipc_pr9.png cited (sec 11)
- SEMANTICS_GATE + ISA-mix / call_count present
- KPI SoT cited: SimtVF 2.4059 / SIMD twin 29.362 (era-2) alongside SoftLib era-1 SoT

## Skill gap (t817u)

Prior DONE bars forced PR9/ISA-mix/SEMANTICS_GATE but did NOT forbid dropping SoftLib LDS.32 SoT / LDS32-LDS64 matrix / full SoT narrative on ANALYSIS regen. Patched into tilelanggym-st-report + softlib-simt-schedule + st-explore-from-prompt (+ L6 note) as no-loss / proven-supersede / TODO regen rule.

## Policy correction (t821u)

Lok clarified: ANALYSIS **full rewrite is allowed**. The enforce bar is **no information loss** and **no silent contradiction** vs previously shipped ANALYSIS; replacements need **proven supersede** (dump/RESULT/harvest) else **TODO / open clarification**. Not merely merge-not-replace.
