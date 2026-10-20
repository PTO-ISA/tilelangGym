# CBT TILE ladder — moving-window IPC NOTES

Date: 2026-10-20 (Asia/Hong_Kong)

## Data source
- **Preferred:** `core0.veccore0.instr_popped_log.dump` (cycle-accurate). Non-empty for all arms.
- Other cores' popped dumps are 0 B; chrome `trace.json` used only for VF_SIMT ts/dur calibration.
- Window: W=20 and W=30, left-open right-closed `(t−W, t]`; IPC = (#Body4)/W; el/cyc = (#FADD·32)/W.
- Body split: silence gap > 80 cycles between Body4 issues ⇒ new VF body region.

## Per-arm summary

| arm | trips | Body4 regions | VF_chrome | VF_cyc | wall_us | wall_cyc≈ | non-VF gap | mean inter-VF gap (cyc) | peak mid IPC20 | whole Body4/VF | el/cyc |
|-----|------:|--------------:|----------:|-------:|--------:|----------:|-----------:|------------------------:|---------------:|---------------:|-------:|
| CBT_TILE2048 | 8 | 8 | 8 | 4203 | 2.34 | 4212 | 9 | 1096 | 1.44 | 0.49 | 3.898 |
| CBT_TILE16384 | 1 | 1 | 1 | 4433 | 2.46 | 4428 | -5 | — | 1.25 | 0.46 | 3.696 |
| FUSED_TILE2048 | 8 | 1 | 1 | 4356 | 2.42 | 4356 | -0 | — | 1.35 | 0.50 | 3.761 |
| CBT_TILE4096 | 4 | 4 | 4 | 4273 | 2.37 | 4266 | -7 | 2104 | 1.29 | 0.48 | 3.834 |
| CBT_TILE8192 | 2 | 2 | 2 | 4253 | 2.37 | 4266 | 13 | 3738 | 1.35 | 0.48 | 3.852 |

### CBT TILE2048 inter-VF gaps (cycles)
- gaps: [1105, 1112, 1054, 1180, 1043, 1111, 1064]
- mean: **1095.6** cyc  min=1043  max=1180
- body spans (cyc): [566, 448, 448, 448, 448, 448, 448, 448]

### Fused TILE2048
- Body4 regions: **1** (expect 1) — continuous VF body wrapping serial tile loop.
- inter-VF gaps: none

### CBT TILE16384
- Body4 regions: **1** (expect 1) — single VF, no inter-VF kickstart gaps.

## Kickstart / wait visibility

**One-liner for Lok:** YES — CBT TILE2048 has 8 Body4 bursts with mean inter-VF silence ≈1096 cyc (~2.4× a ~448-cyc tile body); fused TILE2048 is one continuous Body4. Kickstart/wait+MTE shows clearly (rolling IPC→0 in the red bands). el/cyc=E/ΣVF_cyc ignores those gaps by construction, so it stays ~3.9.

Notes:
- `el/cyc = E / ΣVF_cyc` hides gaps because gaps are **outside** VF_SIMT cycle accounting.
- wall_us − VF_time ≈ MTE + scalar glue; rolling IPC drops to ~0 in the red silence bands.
- Peak mid-body IPC is similar across CBT/fused; the difference is **how many times** you pay the per-VF fill/drain + MTE gap.

## PNG outputs
- `/tmp/sv1_float2_e16k_tile_cbt_mwipc_20261020/CBT_vs_FUSED_rolling20_ipc.png`
- `/tmp/sv1_float2_e16k_tile_cbt_mwipc_20261020/CBT_vs_FUSED_rolling20_elcyc.png`
- `/tmp/sv1_float2_e16k_tile_cbt_mwipc_20261020/CBT2048_vs_FUSED2048_overlay_W20.png`
- `/tmp/sv1_float2_e16k_tile_cbt_mwipc_20261020/CBT_TILE4096_8192_rolling20_ipc.png`
- `/tmp/sv1_float2_e16k_tile_cbt_mwipc_20261020/CBT_vs_FUSED_rolling30_ipc.png`

## Series CSV
- `/tmp/sv1_float2_e16k_tile_cbt_mwipc_20261020/series/*_W20.csv`

## Dump paths
- CBT_TILE2048: `/tmp/sv1_float2_e16k_tile_copy_between_20261020/opsim_sv1tt_e16384_tile2048_t512/OPPROF_20261020162844_KHNCGMQZCHVNBTIC/dump/core0.veccore0.instr_popped_log.dump`
- CBT_TILE16384: `/tmp/sv1_float2_e16k_tile_copy_between_20261020/opsim_sv1tt_e16384_tile16384_t512/OPPROF_20261020163111_JDIDUHBHXBMZNPUC/dump/core0.veccore0.instr_popped_log.dump`
- FUSED_TILE2048: `/tmp/sv1_float2_e16k_tile_t512_20261020/opsim_sv1tt_e16384_tile2048_t512/OPPROF_20261020154855_CCJUQJFOEADATVYD/dump/core0.veccore0.instr_popped_log.dump`
- CBT_TILE4096: `/tmp/sv1_float2_e16k_tile_copy_between_20261020/opsim_sv1tt_e16384_tile4096_t512/OPPROF_20261020162920_UYPACFFYJINNAZCO/dump/core0.veccore0.instr_popped_log.dump`
- CBT_TILE8192: `/tmp/sv1_float2_e16k_tile_copy_between_20261020/opsim_sv1tt_e16384_tile8192_t512/OPPROF_20261020163016_TOABNUYVCRIEDXUN/dump/core0.veccore0.instr_popped_log.dump`
