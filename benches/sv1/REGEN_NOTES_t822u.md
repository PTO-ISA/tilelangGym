# REGEN_NOTES t822u — PR9 each-ISA-group lanes

**When:** 2026-11-02 HKT (Asia/Hong_Kong)
**Why:** Lok t822u — running IPC figures must use latest format with **each ISA group** as a lane.

## What was wrong
- Prior `*_running_ipc_pr9.png` used `LANE_HIDE` that omitted body address-math ISA (`IADD_I`, `LEA`, `SHFI`, …).
- SimtVF sidecar lanes were thin: `S2R, LDS, FADD, STS, END, MOV, LOP3` — missing SHFI×656 / IADD_I×192 / LEA (≈40% of named issues).
- `lane_coverage` was not gated; compute-only hides could still pass `plot_style=opcode_lanes_ipc`.

## Harness change
- `scripts/st_report/rolling_ipc.py`: default `LANE_HIDE = LANE_HIDE_NOISE` only; body address-math shown; `lane_mode=all_isa_groups`; `lane_coverage` in meta. Legacy `LANE_HIDE_COMPUTE_ONLY` retained.
- `scripts/make_pr9_running_ipc_overlay.py`: FAIL if `lane_coverage < 0.95` or empty `lane_opcodes`.
- L6 / PRC-IPC-001 / ST_REPORT_FORMAT / tilelanggym-st-report skill: t822u bar.

## Regen
- Same dumps as t815u (proved replace):
  - SimtVF: `/mnt/fluxdata/happybot/glm53_sv1567_sf_mr2_t813u/out/sv1/opsim_sv1_e8192_t512_frag/OPPROF_20261031170813_FSOMFUCLTGZSTNFZ/dump/core0.veccore0.instr_popped_log.dump → lanes=['S2R', 'IADD_I', 'LEA', 'SHFI', 'LDS', 'FADD', 'STS', 'END', 'LOP3', 'MOV'] cov=1.0`
  - SIMD: `/mnt/fluxdata/happybot/glm53_sv1567_sf_mr2_t813u/out/sv1/opsim_sv1dt_e8192_tile8192_vl64/OPPROF_20261031164454_RFRDZRUKZRTQKIED/dump/core0.veccore0.instr_popped_log.dump → lanes=['VLD', 'VADD', 'VST'] cov=0.9865125240847784`
- Also emitted `*_pr9_staged.png` (STAGE_TAXONOMY bands).

## Policy (t821u)
Full rewrite of plot artifacts OK — new PNG+JSON from same dumps prove supersede; SoftLib LDS SoT narrative untouched.
