# SV5/SV6/SV7 absmax 4-arm — Case-1 **column / Nx1** (t708u)

**Who:** Lok t708u · REGRESSION-2  
**When:** 2026-10-27 HKT (Asia/Hong_Kong)  
**Env:** pto-b10 · cann_92b2 · dav-3510 · Ascend950PR_9599 · `era=fp32-absmax`  
**Live pack:** `/mnt/fluxdata/happybot/glm53_softlib_simt_absmax_nx1_t708u/`  
**TSV:** `reports/RESULT_nx1_t708u.tsv` · box copy `/workspace/RESULT_nx1_t708u.tsv`  
**Axis:** **`axis=Nx1`** (Harness §21) — **do not** merge into 1×G SoftLib SIMT SoT.

---

## 1. Case family

| Field | Definition |
|-------|------------|
| **Gold** | View `A[E]` as shape **`(G, n_cols=E/G)`** row-major; `C[c] = max_r |A[r,c]|` |
| **Shapes** | **16×1, 32×1, 64×1, 128×1** column reduce (G rows × 1 col owned per active lane at stock T) |
| **E** | 8192 fp32 → fp32 |
| **SoftLib SIMT load** | `a[i] = A0[i*n_cols + tid]` (stride-`n_cols` column fold) — **no** `asc_reduce_max` on SoT path |
| **KPI** | `el/cyc = E / named_VF_cyc` |
| **vs 1×G** | Same Case-1 absmax program family; **different reduce axis / SoftLib access** (§21) |

**Not this table:** historical **warp-redux** (`wr_*`, sparse-2 STS) — different gold (warp-level max, not `E/G` column outputs). Reconfirmed as peer only (§5).

---

## 2. 4-arm table (Lok ask — all four arms shown)

KPI el/cyc. SoftLib CCE kept in table per t708u even when ≈ VMI (t706u prose footnote still applies).

| G (Nx1) | VMI | SoftLib CCE | SoftLib SIMT stock | SoftLib SIMT best |
|--:|----:|------------:|-------------------:|------------------:|
| 16 | **33.44** | **31.39** | **9.68** | **11.69** |
| 32 | **22.14** | **21.28** | **4.74** | **9.58** |
| 64 | **30.68** | **30.68** | **3.46** | **6.08** |
| 128 | **32.25** | **32.38** | **2.15** | **2.19** |

### Arm provenance

| Arm | Source | Notes |
|-----|--------|-------|
| **VMI** | t705u Gmap close (`vmi_absmax_gmap_t699u_20261027`) | SIMD **R×C / Gmap** of same Case-1 gold (Harness §19 mapping explore). G16 VL128/NG8 VAND; G32 dig **R32×C256** VL256/NG8; G64 VCMAX VL64 VABS; G128 VMAX→VCMAX / VL128 VCMAX |
| **SoftLib CCE** | CASE_DEF / t685u RESULT | MicroAPI absmax; G16/32 VMI best; G64 **tied**; G128 SoftLib **+0.4%** — footnote-only in prose (t706u), **numbers kept** here |
| **SoftLib SIMT stock** | this pack · `nx1_col` · **T = n_cols** · `lds32_fused` | One thread per column at max launch width |
| **SoftLib SIMT best** | this pack · `nx1_col` · **T = n_cols** · `lds32_membar` | Stage membar wins at G16/32/64; G128 ~flat |

### SoftLib SIMT tags (motif = **nx1-col**)

| G | stock tag | best tag | schedule |
|--:|-----------|----------|----------|
| 16 | `nx1_col_lds32_fused_g16_t512_e8192` (VF 846) | `nx1_col_lds32_membar_g16_t512_e8192` (VF **701**) | nx1-col · LDS.32 · fused → **membar** · T=512 (=n_cols) |
| 32 | `nx1_col_lds32_fused_g32_t256_e8192` (VF 1727) | `nx1_col_lds32_membar_g32_t256_e8192` (VF **855**) | nx1-col · LDS.32 · **membar** · T=256 |
| 64 | `nx1_col_lds32_fused_g64_t128_e8192` (VF 2365) | `nx1_col_lds32_membar_g64_t128_e8192` (VF **1348**) | nx1-col · LDS.32 · **membar** · T=128 |
| 128 | `nx1_col_lds32_fused_g128_t64_e8192` (VF 3812) | `nx1_col_lds32_membar_g128_t64_e8192` (VF **3734**) | nx1-col · LDS.32 · membar ~tied · T=64 |

`lds64_*` tags measured equal to `lds32_*` (generator still scalar-strided loads — **no true LDS.64 along column**; stop recorded).

---

## 3. vs axis=1×G SoftLib SIMT SoT (t697u / t689u)

| G | SIMT best **1×G** | SIMT best **Nx1 col** | Δ (Nx1 ÷ 1×G) |
|--:|------------------:|----------------------:|--------------:|
| 16 | 4.74 (wc halfmask loadall+absall T512) | **11.69** (nx1-col membar T512) | **~2.5×** |
| 32 | 8.22 (wc dual-drain T512) | **9.58** (nx1-col membar T256) | **~1.2×** |
| 64 | 5.54 (wc-multi T512) | **6.08** (nx1-col membar T128) | **~1.1×** |
| 128 | 3.72 (wc-multi T256) | **2.19** (nx1-col membar T64) | **~0.59×** (1×G wins) |

**Reading:** mid-G column SoftLib SIMT benefits from stride-`n_cols` + T=`n_cols` + membar; G128 column is launch-width bound (T≤64) and loses to 1×G multi-warp T256.

---

## 4. SEMANTICS_GATE

```
SoftLib SIMT nx1-col G16/32/64/128: SEMANTICS_GATE: PASS — S1 Case-1 column gold C[c]=max_r|A[r,c]| ok=True maxabs=0; S2 VF body G-loads stride n_cols; S3 SoftLib SIMT LDS32 abs (no VCMAX claim); S4 VF_cyc moves with G/T; axis=Nx1 labeled
VMI R×C / SoftLib CCE: SEMANTICS_GATE: PASS — re-cite t705u/t685u era=fp32-absmax; S1 same Case-1 gold G; mapping explore R×C (§19); not reopened this turn
wr peers G16: SEMANTICS_GATE: PASS for Nx1-peer gold only — NOT Case-1 E/G; do not merge into §2 SoftLib SIMT columns (§21 #24)
```

---

## 5. Appendix — wr peers (different gold)

Reconfirm t686u G16 wr on this machine:

| tag | T | el/cyc |
|-----|--:|-------:|
| `wr_abs_lds32_membar_sparse2_g16_t512_e8192` | 512 | **10.98** (VF 746) |
| `wr_abs_lds64_fused_g16_t512_e8192` | 512 | 10.18 |
| `wr_abs_lds64_fused_g16_t128_e8192` | 128 | 4.51 (stock wr) |

Case-1 **nx1-col** G16 best **11.69** > wr **10.98** (no warp-redux / sparse STS tax; full `E/G` outputs).

---

## 6. Schedule matrix tried (§20) + stop

| Axis | Measured | Stop / residual |
|------|----------|-----------------|
| LDS.32 vs LDS.64 tag | yes (equal) | **true float2 along column = no-path** (non-contig rows) |
| fused vs membar | yes | membar wins G16/32/64; G128 ~flat |
| T ladder | G16: 128/256/512; G32: 128/256; G64: 64/128; G128: 64 | T>`n_cols` idle — not used; half-T slower |
| misched on/off | **not** this pack | residual: SoT @ default bisheng |
| loadall / dual-drain | **not** this pack | residual optional (1×G motifs); Nx1 SoT = membar stage |
| warp-redux on Case-1 col | **excluded** from SoT | wrong gold — peer only |

---

## 7. No PR this turn

Lok asked for the **4-arm show** only. Numbers complete on fluxdata pack + this MD. Parent decides check-in / PR #8 tip update.
