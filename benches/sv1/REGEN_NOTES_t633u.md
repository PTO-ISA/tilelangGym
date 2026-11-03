# REGEN_NOTES t633u — PR9 diagrams visible in ANALYSIS (PR #17)

**When:** 2026-11-03 HKT (Asia/Hong_Kong)
**Why:** Lok t633u on PR #17 — "diagram inside still not the latest PR9 style". Fence-only tip `2f9f658` already had correct `*_running_ipc_pr9.png` + `meta.lane_mode=all_isa_groups`, but ANALYSIS **only table-cited** those paths. GitHub rendered the older IPC-only embeds in §4/§5 (`*_ex_ipc_30cyc.png`, `*_rolling20_ipc.png`).

## Dumps (same proved set as t822u/t823u — available on pto-b10)
- SoftLib: `/tmp/st_out_cce_lds32_ab_20261019/.../core0.veccore0.instr_popped_log.dump`
- SimtVF: `/mnt/fluxdata/happybot/glm53_sv1567_sf_mr2_t813u/out/sv1/opsim_sv1_e8192_t512_frag/.../core0.veccore0.instr_popped_log.dump`
- SIMD twin: `/mnt/fluxdata/happybot/glm53_sv1567_sf_mr2_t813u/out/sv1/opsim_sv1dt_e8192_tile8192_vl64/.../core0.veccore0.instr_popped_log.dump`

## Regen
```bash
PY=/home/happybot/projects/venv-ptoas-vmi-0.1.3/bin/python
PYTHONPATH=. $PY scripts/make_pr9_running_ipc_overlay.py --dump <DUMP> --out-png benches/sv1/plots/<arm>_running_ipc_pr9.png --title "... · all ISA groups" --E 8192 --frontend softlib_simt|simd --staged-out ..._pr9_staged.png
```

## Gate
| arm | plot_style | lane_mode | lane_coverage | lanes |
|-----|------------|-----------|--------------:|-------|
| SoftLib LDS.32 best | opcode_lanes_ipc | all_isa_groups | 1.0 | S2R, IADD_I, LEA, SHFI, LDS, FADD, STS, END |
| SimtVF frag | opcode_lanes_ipc | all_isa_groups | 1.0 | S2R, IADD_I, LEA, SHFI, LDS, FADD, STS, END, LOP3, MOV |
| SIMD twin | opcode_lanes_ipc | all_isa_groups | 0.9865 | VLD, VADD, VST |

## ANALYSIS
- Add `![...](plots/*_running_ipc_pr9.png)` embeds under §11
- Demote §4/§5 IPC-only embeds as **Non-PR9 sidecar** captions
- Keep How-to-run fence closed (t631u); no KPI invention; SoftLib SoT retained

## DO NOT MERGE until Lok reviews
