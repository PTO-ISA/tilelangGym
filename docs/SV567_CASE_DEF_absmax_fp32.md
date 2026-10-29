# SV5 / SV6 / SV7 — Case definition (baseline)

**Locked (Lok t683u, 2026-10-27 HKT):** SoftLib/ST SV5–7 **baseline case** is **absmax**, **input fp32 → output fp32**.  
Do **not** treat Case-3 recip as the primary case for the 4-arm / SoT table.

**Applied (t697u axis=1xG; t698u peer):** SoftLib SIMT **main** stock/best = contiguous **1×G only** (`axis=1xG`, Harness §21). Historical G16 **wr** (stride-T per-lane fold → one warp redux = **16×1 / N×1**) → **Nx1 appendix** only. G16 main SoT = **wc halfmask** true **1×16**. Primary close: `/workspace/SOFTLIB_SIMT_1xG_AXIS_FIX_t697u.md` (G16 best **4.74** loadall+absall T512). Peer: `/workspace/SOFTLIB_ABSMAX_4ARM_1xG_t698u.md` (G16 **4.63** loadall+membar).

**Applied (t686u multi-warp):** continuous/contig load + warp-level reduce + multi-warp T ladder (§6).  
**Applied (t689u G32 full back-pressure):** G32 SoftLib SIMT best → dual-drain T512 el/cyc **8.22**.

---

## 1. Baseline case (Case-1)

| Field | Definition |
|-------|------------|
| **Name** | **absmax** (group abs-reduce-max) |
| **ST map** | **SV5** gold G=**16**; **SV6** G=**32**; **SV7** G=**64** and **128** |
| **E** | **8192** elements (fp32) |
| **Input** | `A[E]` **fp32** |
| **Output** | `C[E/G]` **fp32** — one absmax per gold group of size **G** |
| **Compute** | per group \(g\): \(C[g] = \max_{i \in g} |A[i]|\) with group = contiguous **1×G** cols \([gG, gG+G)\) |
| **Gold** | absmax-only — **no** clamp, **no** recip / `1/max` epilogue |
| **Reduce axis** | **`axis=1xG`** (Harness §21) — required for SoftLib SIMT main compare |
| **KPI** | `el/cyc = E / named_VF_cyc` |
| **era=** | `fp32-absmax` (datatype/cast + program family) |

**Shape / tiling:** R×C or 1D+group is a **mapping explore** (Harness §19), not a case change — E and gold G stay fixed.  
**ABS path:** i32 bitcast+VAND vs VABS_FP is an **instruction-mapping** explore (Harness §18), not a case change.

---

## 2. Not the baseline (appendix / foreign era)

| Variant | Why different | Use |
|---------|---------------|-----|
| **Case-3 recip** | Same fp32 dtype class, but gold = \(Y = 1/\max(\mathrm{absmax}, \varepsilon)\) — **different program / epilogue** (§13) | Appendix only; **do not** merge into absmax SoT table |
| **N×1 / warp-redux fold** (`axis=Nx1`) | Per-lane fold at stride=T then one `asc_reduce_max` (historical G16 **wr** 16×1-style) — **different reduce axis** (§21) | §3b appendix / separate Nx1 suite; **not** SoftLib SIMT main column |
| fp16→fp32 cast + reduce | Different datatype/cast era | Later / separate Aims |
| SoftLib A-only LDS load | No absmax product | Load SoT / ladder only |

**Answer to “is Case-3 different dtype?”:** **No** — Case-3 SoftLib SIMT digs here are still **fp32**. The split is **program family / gold** (absmax-only vs recip), not fp16 vs fp32.

---

## 3. Absmax 4-arm table (baseline era only · **axis=1xG**)

KPI el/cyc. SoftLib SIMT **stock** = 1×G warp-coop at legacy T; **best** = 1×G multi-warp / back-pressure (t697u G16; t689u G32).  
**G16 wr / N×1 is appendix only** — do not put in SoftLib SIMT columns below (Harness §21 / pollution #24).

| ST | G | SIMD VMI (absmax) | SoftLib CCE (absmax) | SoftLib SIMT stock **1×G** | SoftLib SIMT best **1×G** |
|----|---:|--:|--:|--:|--:|
| SV5 | 16 | **33.44** (VL128/NG8 VAND; VF 245) | **31.39** (VCGMAX-ref; VF 261) | **1.13** (wc halfmask LDS.32 T128; VF 7264) | **4.74** (wc halfmask loadall+absall drain **T512**; VF 1727; t697u) |
| SV6 | 32 | **22.14** (VL256/NG8 storeB; VF 370; best-sweep PARTIAL) | **21.28** (half-VL VCMAX; VF 385) | **1.57** (wc LDS.32 T128; VF 5225) | **8.22** (wc loadall+absall+dual-drain **T512**; VF 997; t689u) |
| SV7 | 64 | **OPEN** | **30.68** (whole-VL VCMAX; VF 267) | **2.39** (wc-multi per2 T128; VF 3434) | **5.54** (wc-multi LDS.32 **T512**; VF 1478) |
| SV7 | 128 | **OPEN** | **32.38** (VMAX→VCMAX; VF 253) | **1.67** (wc-multi per4 T64; VF 4916) | **3.72** (wc-multi LDS.32 **T256**; VF 2201) |

### SoftLib SIMT SoT tags (post-t697u · axis=1xG)

| G | stock tag (legacy T) | best tag | motif |
|--:|----------------------|----------|-------|
| 16 | `wc_abs_lds32_redux_g16_t128_e8192` | `wc_abs_back_loadall_absall_dual_drain_g16_t512_e8192` | warp-coop-halfmask · **1×16** |
| 32 | `wc_abs_lds32_redux_g32_t128_e8192` | `wc_abs_back_loadall_absall_dual_drain_g32_t512_e8192` | warp-coop · **1×32** |
| 64 | `wc_abs_lds32_redux_g64_t128_e8192` | `wc_abs_lds32_redux_g64_t512_e8192` | warp-coop-multi · **1×64** |
| 128 | `wc_abs_lds32_redux_g128_t64_e8192` | `wc_abs_lds32_redux_g128_t256_e8192` | warp-coop-multi · **1×128** |

**Appendix N×1 (G16 wr — not main SoftLib SIMT):** stock `wr_abs_lds64_fused_g16_t128` **4.51** · best `wr_abs_lds32_membar_sparse2_g16_t512` **10.98**. Detail: `/workspace/SOFTLIB_ABSMAX_4ARM_1xG_t698u.md` §5.


## 3b. Appendix — SoftLib SIMT **N×1** family (not main SoT)

`axis=Nx1` · per-lane stride-T fold + warp redux · **do not** merge into §3 SoftLib SIMT columns.

| G | tag (historical) | T | el/cyc | note |
|--:|------------------|---:|-------:|------|
| 16 | `wr_abs_lds32_membar_sparse2_g16_t512_e8192` | 512 | **10.98** | was t686u SoftLib SIMT “best”; **16×1-style**, not Case-1 1×16 |
| 16 | `wr_abs_lds64_fused_g16_t128_e8192` | 128 | 4.51 | stock wr |
| 32 / 64 / 128 | — | — | — | optional future **32×1 / 64×1 / 128×1** suite |


## 4. Case-3 numbers (appendix — do not merge)

| ST | G | SoftLib SIMT stock (Case-3) | SoftLib SIMT best (Case-3) |
|----|---:|--:|--:|
| SV5 | 16 | 1.647 | 1.881 (wg lds64_loadall_membar) |
| SV6 | 32 | 0.925 | 1.261 (warp-coop) |
| SV7 | 64 | 0.918 | 2.010 (warp-coop-multi) |
| SV7 | 128 | 0.913 | 1.443 (warp-coop-multi) |

Detail: `/mnt/fluxdata/happybot/glm53_softlib_simt_20261026/reports/SOFTLIB_SIMT_CLOSE_t676u.md`

---

## 5. Next measure (under this case def)

1. ~~SoftLib SIMT absmax G32/64/128~~ **done t685u**.  
2. ~~Multi-warp T ladder~~ **done t686u**.  
3. ~~Axis fix: SoftLib SIMT main = 1×G; demote wr Nx1~~ **done t697u** (G16 best **4.74**; t698u peer 4.63).  
4. Re-list **VMI absmax** SoT per G under `era=fp32-absmax` — G64/128 still OPEN.  
5. Optional: G16 1×G further §20 (true paired dual-drain / LDS.64 halfmask); optional **Nx1 suite** 16×1/32×1/64×1/128×1 as separate compare.  
6. Keep Case-3 recip as optional sensitivity appendix only.

---

## 6. SoftLib SIMT measurement-flow defaults (t697u · axis=1xG)

Write these into gen/run scripts **before** later check-cap / schedule digs:

| Knob | Default | Notes |
|------|---------|-------|
| **Axis** | **`axis=1xG`** contiguous Case-1 gold | Harness §21; Nx1 wr → appendix only |
| **Load** | continuous / contig `A0[i*T + tid]` (or float2 LDS.64 load-all) | |
| **Reduce** | warp-coop / halfmask / warp-coop-multi `asc_reduce_max` covering **1×G** | G16 = halfmask; G32 = full warp; G≥64 = multi with fold **inside** gold group |
| **T (threads)** | **multi-warp ladder** | G16/32/64 best **T=512**; G128 best **T=256** |
| **Stock T** | G16/32/64 → T128; G128 → T64 | Keep for stock column / regression |
| **G16 best schedule** | halfmask **loadall+absall drain** T512 | el/cyc **4.74** (t697u); t698u loadall+membar 4.63 peer; not wr |
| **G32 best schedule** | **loadall+absall+dual-drain** T512 | el/cyc **8.22** (t689u) |
| **Pack (1×G)** | `/mnt/fluxdata/happybot/glm53_softlib_simt_absmax_1xg_t697u/` (peer t698u) | t697u harvest / winners |
| **Pack (G32 back)** | `/mnt/fluxdata/happybot/glm53_softlib_simt_absmax_g32_back_t689u/` | G32 dual-drain 8.22 |
| **Pack (Nx1 wr)** | `/mnt/fluxdata/happybot/glm53_softlib_simt_absmax_multwarp_t686u/` | appendix only |

Generator: `/workspace/_gen_1xg_abs_t697u.py` · runner `_run_1xg_abs_t697u.sh` · harvest `_harvest_1xg_t697u.py` (peer `_gen_1xg_t698u.py`).
