# t815u SV1 report regen — NOTES

**When:** 2026-11-02 HKT (Asia/Hong_Kong) · **Pack:** `/mnt/fluxdata/happybot/glm53_sv1567_sf_mr2_t813u` · **Foundry tip:** `3c959fa` · **Plot tip repo:** `tilelangGym-pr8-t705u` @ `27e5540`

## What was wrong with the old plots

1. **tip `benches/sv1/plots/`** (and pack `ref/sv1`) still carry **IPC-only** figures:
   - `sv1_simt_e16k_*_ex_ipc_30cyc.png`
   - `sv1_cbt_vs_fused_rolling20_ipc.png`
   - `sv1d_e16k_*_ex_ipc_30cyc.png`
   These are single-axis / body4 rolling IPC — **not** PR9 (`opcode_lanes_ipc` dual-pane). Skill FORCED reject.
2. Pack `sv1/ANALYSIS.md` (t813u) already cited PR9 PNGs and ISA-mix, but:
   - **missing explicit `SEMANTICS_GATE: PASS — …` one-liner** (t814 residual)
   - `sv1/plots/` had PNGs **without** co-located JSON sidecars (JSON lived only under pack `plots/`)
3. This regen re-ran `scripts/make_pr9_running_ipc_overlay.py` against the same best dumps and co-locates `*_running_ipc_pr9.png` + JSON with top-level `lane_opcodes` + `plot_style=opcode_lanes_ipc`.

## Commands run (pto-b10)

```bash
PACK=/mnt/fluxdata/happybot/glm53_sv1567_sf_mr2_t813u
PY=/home/happybot/projects/venv-ptoas-vmi-0.1.3/bin/python
REPO=/mnt/fluxdata/happybot/tilelangGym-pr8-t705u
DUMP_SIMTVF=$PACK/out/sv1/opsim_sv1_e8192_t512_frag/OPPROF_20261031170813_FSOMFUCLTGZSTNFZ/dump/core0.veccore0.instr_popped_log.dump
DUMP_SIMDT=$PACK/out/sv1/opsim_sv1dt_e8192_tile8192_vl64/OPPROF_20261031164454_RFRDZRUKZRTQKIED/dump/core0.veccore0.instr_popped_log.dump

$PY $PACK/scripts/make_pr9_running_ipc_overlay.py \
  --dump "$DUMP_SIMTVF" \
  --out-png $PACK/t815u_sv1_regen/plots/sv1_simtvf_running_ipc_pr9.png \
  --title "SV1 / SimtVF E8192 T512 frag - running IPC (W=20)" --E 8192 --repo "$REPO"

$PY $PACK/scripts/make_pr9_running_ipc_overlay.py \
  --dump "$DUMP_SIMDT" \
  --out-png $PACK/t815u_sv1_regen/plots/sv1_simdt_running_ipc_pr9.png \
  --title "SV1 / handwritten SIMD twin E8192 TILE8192 VL64 - running IPC (W=20)" --E 8192 --repo "$REPO"
```

Also copied harvest slices: `sv1_isa_mix_perN.json`, `sv1_stage_mid_ipc.json`, `sv1_opcode_mix.json`, `sv1_harvest.json`.

## KPI cited (unchanged — from RESULT / harvest; not invented)

| arm | tag | el/cyc |
|-----|-----|------:|
| SimtVF best | `sv1_e8192_t512_frag` | **2.4059** |
| SIMD twin hub | `sv1dt_e8192_tile8192_vl64` | **29.3620** |
| SIMD twin E16K peak | `sv1dt_e16384_tile16384_vl64` | 31.5684 |
| Both-arm ratio (same E) | SIMD ≻ SimtVF | **~12.2×** |
| ISA-mix bridge | pred/meas | **1.0000** both arms |

## SEMANTICS_GATE line

**Present** in regenerated `ANALYSIS.md` header:

`SEMANTICS_GATE: PASS — S1 in-kernel / suite numeric check (C=A+B fp32); S2 SimtVF VF_SIMT body vs SIMD twin vf_real_execute_time body (era=fp32-stream-eltwise, reduce_axis=na-eltwise); S3 stream mix only (LDS/FADD/STS vs VLD/VADD/VST — no reduce family); S4 named counters (VF_SIMT / vf_real_execute_time) — A1 / A10.`

## PR9 proof

| file | plot_style | lane_opcodes |
|------|------------|--------------|
| `plots/sv1_simtvf_running_ipc_pr9.json` | opcode_lanes_ipc | S2R, LDS, FADD, STS, END, MOV, LOP3 |
| `plots/sv1_simdt_running_ipc_pr9.json` | opcode_lanes_ipc | VLD, VADD, VST |

mid-med IPC: SimtVF **0.50** · SIMD twin **2.20**

## File list (this deliverable dir)

- `ANALYSIS.md`
- `NOTES.md`
- `plots/sv1_simtvf_running_ipc_pr9.png`
- `plots/sv1_simtvf_running_ipc_pr9.json`
- `plots/sv1_simdt_running_ipc_pr9.png`
- `plots/sv1_simdt_running_ipc_pr9.json`
- `plots/sv1_isa_mix_perN.json`
- `plots/sv1_stage_mid_ipc.json`
- `plots/sv1_opcode_mix.json`
- `plots/sv1_harvest.json`

## Absolute pack paths (pto-b10)

- Regen root: `/mnt/fluxdata/happybot/glm53_sv1567_sf_mr2_t813u/t815u_sv1_regen/`
- Also mirrored plots under pack regen; original t813u ANALYSIS remains at `/mnt/fluxdata/happybot/glm53_sv1567_sf_mr2_t813u/sv1/ANALYSIS.md` (pre-SEMANTICS_GATE) — this deliverable is the review copy.

## PR handoff (parent / CloudAgent)

- Do **not** git-commit from this subagent.
- Suggested target tree for review PR (tilelangGym `benches/sv1/` pattern, historical PTO-ISA/tilelangGym PR #8 SoftLib+IPC style): replace/add `benches/sv1/ANALYSIS.md` + `benches/sv1/plots/*_running_ipc_pr9.{png,json}` (+ optional harvest JSON). Remove or demote IPC-only legacy PNGs from the PR diff if they would confuse review.
- Note: `peanutchan/tilelang-ai` is a **tilelang** language fork (no `benches/sv1` at HEAD); historical SV report PRs (#5–#8) live on **PTO-ISA/tilelangGym**. Parent should confirm which remote/branch before opening the CloudAgent PR.
