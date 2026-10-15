# PASS / FAIL + perf (clean)
**Date:** 2026-10-15 HKT · **SOC:** Ascend950PR_9599 · **Pin:** TileLang remat tip `a3012088` + [tilelang#5](https://github.com/PTO-ISA/tilelang/pull/5)
Single living table. Historical design notes / raw dumps were removed from `reports/`.
Per-case highlight + verdict: `benches/<id>/VERDICT.md` (SV1: `benches/sv1/ANALYSIS.md`).

## SV1–SV9 simt-asc (gate / measured)

| Tag | Status | µs | IPC / note |
|-----|--------|---:|------------|
| `sv1_e256_t32` | **PASS** | 1.21 | 0.194 |
| `sv1_e2048_t32` | **PASS** | 2.58 | 0.200 |
| `sv3_m24_vl64_k16_t32_keep` | **PASS** | 6.79 | 0.324 |
| `sv3_m32_vl64_k16_t32_keep` | **PASS** | 8.63 | 0.340 |
| `sv3_m32_vl64_k16_t32_split_cm16` | **PASS** | 7.79 | 0.355 |
| `sv4_e256_b8_t32_keep2` | **PASS** | 3.79 | 0.118 |
| `sv4_e256_b16_t32_keep2` | **PASS** | 6.83 | 0.106 |
| `sv6_r64_c128_g32_t32` | **PASS** | 12.25 | 0.110 |
| `sv7_r64_c128_g64_t32` | **PASS** | 14.44 | 0.090 |
| `sv7_r64_c128_g128_t32` | **PASS** | 14.69 | 0.088 |
| `sv8_r64_c128_g16_t32_live` | **PASS** | 17.04 | 0.141 |
| `sv8_r64_c128_g16_t32_spill_dist` | **PASS** | 18.23 | 0.138 |
| `sv9_e256_k1_t32_keep` | **PASS** | 1.17 | 0.262 |
| `sv9_e256_k8_t32_keep` | **PASS** | 1.48 | 0.505 |
| `sv9_e256_k8_t32_remat_scores` | **PASS** | 2.02 | 0.301 |
| `sv9_e256_k8_t32_remat_idx` | **PASS** | 1.48 | 0.505 |
| `sv4_e256_b8_t32_keep_idx` | **PASS** | 1.67 | 0.367 |
| `sv4_e256_b8_t32_remat_idx_calc` | **PASS** | 1.62 | 0.417 |
| `sv4_e256_b16_t32_keep_idx` | **PASS** | 2.39 | 0.343 |
| `sv4_e256_b16_t32_remat_idx_calc` | **PASS** | 2.22 | 0.369 |
| `sv4_e512_b8_t32_keep_idx` | **PASS** | 2.83 | 0.290 |
| `sv4_e512_b8_t32_remat_idx_calc` | **PASS** | 2.71 | 0.288 |
| `sv4_e1024_b8_t32_keep_idx` | **PASS** | 7.16 | 0.174 |
| `sv4_e1024_b8_t32_remat_idx_calc` | **PASS** | 6.30 | 0.192 |
| `sv4_e1024_b16_t32_keep_idx` | **PASS** | 11.83 | 0.172 |
| `sv4_e1024_b16_t32_remat_idx_calc` | **PASS** | 10.53 | 0.190 |
| `sv4_e1536_b8_t32_keep_idx` | **PASS** | 12.09 | 0.130 |
| `sv4_e1536_b8_t32_remat_idx_calc` | **PASS** | 12.79 | 0.132 |
| `sv4_e2048_b8_t32_keep_idx` | **PASS** | 26.08 | 0.114 |
| `sv4_e2048_b8_t32_remat_idx_calc` | **PASS** | 26.01 | 0.107 |
| `sv4_e1024_b8_t16_keep_idx` | **PASS** | 84.44 | 0.033 |
| `sv4_e1024_b8_t16_remat_idx_calc` | **PASS** | 86.69 | 0.030 |
| `sv4d_e256_b8_t32_keep_idx` | **PASS** | 1.02 | - |
| `sv4d_e256_b8_t32_remat_idx` | **PASS** | 1.02 | - |
| `sv4d_e512_b8_t32_keep_idx` | **PASS** | 1.25 | - |
| `sv4d_e512_b8_t32_remat_idx` | **PASS** | 1.25 | - |
| `sv4d_e1024_b8_t32_keep_idx` | **PASS** | 2.32 | - |
| `sv4d_e1024_b8_t32_remat_idx` | **PASS** | 2.32 | - |
| `sv4d_e256_b16_t32_keep_idx` | **PASS** | 1.14 | - |
| `sv4d_e256_b16_t32_remat_idx` | **PASS** | 1.14 | - |
| `sv4d_e2048_b8_t32_keep_idx` | **FAIL** | NA | - |
| `sv4d_e2048_b8_t32_remat_idx` | **FAIL** | NA | - |
| `sv5_r64_c128_g8_t32_keep_reg` | **PASS** | 10.44 | 0.157 |
| `sv5_r64_c128_g8_t32_ub_reload` | **PASS** | 8.10 | 0.202 |
| `sv5_r64_c128_g16_t32_keep_reg` | **PASS** | 10.82 | 0.142 |
| `sv5_r64_c128_g16_t32_ub_reload` | **PASS** | 7.74 | 0.193 |
| `sv5_r64_c128_g32_t32_keep_reg` | **PASS** | 26.08 | 0.101 |
| `sv5_r64_c128_g32_t32_ub_reload` | **PASS** | 25.86 | 0.102 |
| `sv5_r128_c128_g16_t32_keep_reg` | **FAIL** | 24.49 | 0.133 |
| `sv5_r128_c128_g16_t32_ub_reload` | **PASS** | 14.81 | 0.190 |
| `sv5_r256_c128_g16_t32_keep_reg` | **FAIL** | 100.07 | 0.068 |
| `sv5_r256_c128_g16_t32_ub_reload` | **PASS** | 29.32 | 0.186 |
| `sv5_r64_c128_g16_t32_keep_in_rf` | **PASS** | 7.04 | 0.110 |
| `sv5_r64_c128_g16_t32_ub_stream` | **PASS** | 4.18 | 0.220 |
| `sv6_r64_c128_g32_t32_keep_in_warp` | **PASS** | 14.83 | 0.092 |
| `sv6_r64_c128_g32_t32_reload` | **PASS** | 12.58 | 0.107 |
| `sv7_r64_c128_g64_t32_multiwarp_ub` | **PASS** | 7.32 | 0.114 |
| `sv7_r64_c128_g64_t32_reload` | **PASS** | 14.54 | 0.090 |
| `sv7_r64_c128_g128_t32_multiwarp_ub` | **PASS** | 15.42 | 0.089 |
| `sv7_r64_c128_g128_t32_reload` | **PASS** | 14.58 | 0.089 |
| `sv2_r16_c32_t32_input_keep` | **PASS** | 1.16 | 0.362 |
| `sv2_r32_c64_t32_input_stream` | **PASS** | 3.60 | 0.149 |
| `sv2_r32_c128_t32_input_stream` | **PASS** | 8.49 | 0.117 |
| `sv2_r32_c64_t32_reload` | **PASS** | 4.42 | 0.130 |

## SV1 SIMD DSL highlights (roof ~44 / Lok 42.67)

| Tag | Status | µs | el/cyc |
|-----|--------|---:|-------:|
| `sv1d_e8192_vl64` | **PASS** | 2.59 | 29.3 (~69% of 42.67) |
| `sv1d_e16384_vl64` | **PASS** | 4.55 | 31.5 (~74% VF; body mid peak ~**85%**) |

**SV1 keynote:** SIMD = high-UB-BW load/st in a small loop — **no per-thread index, no instr schedule** — ~**85%** of ~**44** el/cyc roof. See [`benches/sv1/ANALYSIS.md`](../benches/sv1/ANALYSIS.md).

## How to read
- **PASS/FAIL** here is the suite rollup.
- **Highlight + verdict** for each bench lives under `benches/<id>/VERDICT.md`.
