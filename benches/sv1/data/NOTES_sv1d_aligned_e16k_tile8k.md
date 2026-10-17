# SV1d tiled SIMD — aligned E=16K TILE sweep

- **Date:** 2026-10-17 HKT
- **SOC:** Ascend950PR_9599 · **CANN:** cann_92b2
- **Kernel:** fully-unrolled per-tile MTE (`scripts/gen_sv1dt_kernel.py`) — MTE load A/B tile → VL=64 vadd → MTE store C
- **Tag:** `sv1dt_e16384_tile{TILE}_vl64`
- **Axis:** fixed E=16384, TILE ∈ {2048,4096,8192,16384} (same as Simt §4.1)

| TILE | VF_cyc | el/cyc | IPC | us |
|-----:|-------:|-------:|----:|---:|
| 2048 | 824 | 19.8835 | 1.2913 | 7.69 |
| 4096 | 644 | 25.441 | 1.6211 | 5.87 |
| 8192 | 560 | 29.2571 | 1.8464 | 4.99 |
| 16384 | 520 | 31.5077 | 1.9788 | 4.55 |

Hub IPC plot: **TILE=8192** (`plots/sv1d_e16k_tile8k_ex_ipc_30cyc.png`).
Prior whole-E size ladder (`SUMMARY_sv1d_size_ladder.tsv`) kept as appendix only.
