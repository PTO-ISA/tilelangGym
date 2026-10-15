# 30-cycle sliding body-mix IPC plots — SV1d + SimtVF (aligned)

**Date:** 2026-10-14 HKT · Lok: same metric on both — full LD/ST/EX body mix (**no FADD-only / no EX-only**)

## Metric (aligned)

| Kernel | Body-mix ops counted in 30-cyc window |
|--------|----------------------------------------|
| **SV1d SIMD** | `RV_VLDI + RV_VADD + RV_VSTI` |
| **SimtVF** | `SIMT_LDS + SIMT_STS + SIMT_IADD + SIMT_IADD_I + SIMT_FADD` (**Body4**, pack **.64**) |

Both plots: window center vs body-mix IPC; mid-body median; peak excl fill/drain; matching axes/titles.

**Superseded:** `sv1_simt_e16k_fadd_ex_ipc_30cyc.*` (FADD-only) — do not use for compare.

## Plots

| Kernel | File |
|--------|------|
| SV1d E=16384 VL=64 | `plots/sv1d_e16k_ex_ipc_30cyc.png` (+`.pdf`) |
| SimtVF TILE2048@T128 E=16384 | `plots/sv1_simt_e16k_body4_ex_ipc_30cyc.png` (+`.pdf`) |
| Rooftop summary (TILE labeled) | `plots/sv1_rooftop_corrected_summary.png` |

Also: `C:\Users\l\tmp\sv1_plots\` · pto-b10 `benches/sv1/plots/`

## SV1d E=16K (SIMD) — body-mix VLDI+VADD+VSTI

- Trace: `data/sv1d_e16384_vl64/`
- VF_cyc=520 · body (1st VLDI→last VSTI issue)=473.2 · GHz_cal≈1.7993
- **mid-body median IPC = 2.133**
- **peak after fill (VSTI-active) = 2.767** @ c≈84
- mid-body peak (central 50%) = **2.267**
- early fill blip = 3.367 @ c≈48 (LD-heavy — ignore for steady)
- Roof note: add≈IPC/4 when mix≈2:1:1 → mid med ~34.1 el/cyc (~80% of 42.67)

## SimtVF TILE2048@T128 — Body4 LDS+STS+IADD+FADD

- Trace: `data/sv1tt_e16384_tile2048_t128/`
- VF_cyc=5088 · Body4=1860 (LDS=512 STS=256 IADD=580 FADD=512) · pack **.64**
- Body = 1st LDS → last STS issue ≈ 5010.6 cyc
- **mid-body median Body4 IPC = 0.367**
- **peak excl cold/drain Body4 IPC = 0.667** @ c≈1404
- mix-cap IPC≈4 (4 wraps) drawn on plot; measured ~9% of width
- el/cyc_VF = 3.22 (~10% of roof 32)
- **No FADD-only series** on this plot (Lok)

## Scripts

- `scripts/plot_sv1d_ex_ipc_30cyc.py`
- `scripts/plot_sv1_simt_body4_ex_ipc_30cyc.py`
- `scripts/plot_sv1_rooftop_corrected_summary.py` (T-sweep = **TILE=E=16384 untiled**, not 8192)

## Corrected rooftops

See `NOTE_sv1_rooftop_corrected.md`.
