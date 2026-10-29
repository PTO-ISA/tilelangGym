# SoftLib absmax Case-1 — 4-arm CLOSE on true contiguous **1×G** (t698u)

**何時：** 2026-10-27 14:33 HKT（Asia/Hong_Kong）  
**邊個叫：** Lok t697–t698u — fix SoftLib SIMT to **1×16/1×32/1×64/1×128**; harness axis rule; refresh 4-arm  
**Case-1 lock：** absmax · A **fp32** → C **fp32** · E=8192 · gold G ∈ {16,32,64,128} · **NO recip** · `era=fp32-absmax` · **`axis=1xG`**  
**Harness：** §21 reduce-axis mapping + pollution #24  
**Live pack：** `/mnt/fluxdata/happybot/glm53_softlib_simt_absmax_1xg_t698u/`  
**Env：** cann_92b2 / dav-3510 / `.venv-npu` / `sim_dsl.sh` Ascend950PR_9599（同 t685–t689）

---

## 1. Semantics fix

| Class | Meaning | Main SoftLib SIMT 4-arm? |
|-------|---------|--------------------------|
| **1×G** | Contiguous group `[gG, gG+G)` absmax; G16 halfmask / G32 full-warp / G64–128 warp-coop-multi (per-lane fold **inside** gold G) | **Yes** |
| **N×1** (G16 **wr**) | Per-lane fold at stride=T then one warp redux — **16×1-style**, different axis | **No** — appendix only |

Prior SoftLib SIMT G16 **wr 10.98** is **not** apples-to-apples with G32/64/128 **wc**. Main table below is **1×G only**.

---

## 2. Full 4-arm table（absmax · axis=1xG）

KPI el/cyc. SoftLib SIMT **stock** = 1×G fused at legacy T; **best** = 1×G multi-warp / back-pressure winner.

| ST | G | SIMD VMI (absmax) | SoftLib CCE (absmax) | SoftLib SIMT stock **1×G** | SoftLib SIMT best **1×G** |
|----|---:|--:|--:|--:|--:|
| SV5 | 16 | **33.44** (VL128/NG8 VAND; VF 245) | **31.39** (VCGMAX-ref; VF 261) | **1.13** (wc halfmask LDS.32 T128; VF 7264) | **4.63** (wc halfmask **loadall+membar** T512; VF 1771) |
| SV6 | 32 | **22.14** (VL256/NG8 storeB; VF 370; best-sweep PARTIAL) | **21.28** (half-VL VCMAX; VF 385) | **1.57** (wc LDS.32 T128; VF 5225) | **8.22** (wc **loadall+absall+dual-drain** T512; VF 997; t689u / reconfirm t698u 8.22) |
| SV7 | 64 | **OPEN** | **30.68** (whole-VL VCMAX; VF 267) | **2.39** (wc-multi per2 T128; VF 3434) | **5.54** (wc-multi LDS.32 T512; VF 1478) |
| SV7 | 128 | **OPEN** | **32.38** (VMAX→VCMAX; VF 253) | **1.67** (wc-multi per4 T64; VF 4916) | **3.72** (wc-multi LDS.32 T256; VF 2201) |

**VMI / SoftLib CCE：** re-cited from t685u / `SV567_CASE_DEF_absmax_fp32.md` / RESULT.tsv — **not** reopened this turn (already 1×G MicroAPI/SIMD Case-1).

### SoftLib SIMT 1×G SoT tags (t698u)

| G | stock tag | best tag | motif |
|--:|-----------|----------|-------|
| 16 | `wc_abs_lds32_redux_g16_t128_e8192` | `wc_abs_back_loadall_membar_g16_t512_e8192` | warp-coop-halfmask · axis=1xG |
| 32 | `wc_abs_lds32_redux_g32_t128_e8192` | `wc_abs_back_loadall_absall_dual_drain_g32_t512_e8192` | warp-coop · axis=1xG |
| 64 | `wc_abs_lds32_redux_g64_t128_e8192` | `wc_abs_lds32_redux_g64_t512_e8192` | warp-coop-multi · axis=1xG |
| 128 | `wc_abs_lds32_redux_g128_t64_e8192` | `wc_abs_lds32_redux_g128_t256_e8192` | warp-coop-multi · axis=1xG |

---

## 3. SoftLib SIMT 1×G tried (t698u live · all ok=True)

### G=16 (true 1×16)

| tag | motif | T | el/cyc | note |
|-----|-------|--:|-------:|------|
| wc_abs_lds32_redux_g16_t128 | halfmask | 128 | **1.1278** | **stock** |
| wc_abs_lds32_redux_g16_t256 | halfmask | 256 | 1.9176 | |
| wc_abs_lds32_redux_g16_t512 | halfmask | 512 | 3.1520 | bare multi-warp |
| wc_abs_back_loadall_membar_g16_t256 | halfmask+back | 256 | 2.7911 | |
| wc_abs_back_loadall_membar_g16_t512 | halfmask+back | 512 | **4.6256** | **best 1×16** |
| wc_abs_back_loadall_absall_dual_drain_g16_t256 | halfmask+dual | 256 | 1.8789 | |
| wc_abs_back_loadall_absall_dual_drain_g16_t512 | halfmask+dual | 512 | 3.3978 | dual < loadall_membar |
| wg_abs_lds64_loadall_membar_g16_t128 | within-group | 128 | 1.8889 | 1×G peer |
| wg_abs_lds64_loadall_membar_g16_t256 | within-group | 256 | 1.9033 | |
| wg_abs_lds64_loadall_membar_g16_t512 | within-group | 512 | 1.9154 | weak T-scale |

### G=32 / 64 / 128 (confirm)

| G | stock el/cyc | best el/cyc | best tag |
|--:|-------------:|------------:|----------|
| 32 | 1.5678 | **8.2166** | dual_drain T512 |
| 64 | 2.3856 | **5.5426** | wc T512 |
| 128 | 1.6664 | **3.7219** | wc T256 |

Bit-match prior t686/t689 within noise (G32 dual 8.22 vs 8.2166).

---

## 4. Check-cap（1×G SoftLib SIMT best）

| G | best | % of SoftLib A-only ~16 | SoftLib CCE | % of CCE |
|--:|-----:|------------------------:|------------:|---------:|
| 16 | 4.63 | **28.9%** | 31.39 | **14.7%** |
| 32 | 8.22 | **51.4%** | 21.28 | **38.6%** |
| 64 | 5.54 | **34.6%** | 30.68 | **18.1%** |
| 128 | 3.72 | **23.3%** | 32.38 | **11.5%** |

---

## 5. Appendix — N×1 / G16 wr（do **not** merge into main SoftLib SIMT）

Historical warp-redux fold (`axis=Nx1` / 16×1-style). From t685u / t686u packs.

| schedule | T128 | T256 | T512 | note |
|----------|-----:|-----:|-----:|------|
| wr lds64 fused | **4.51** (old “stock”) | 7.15 | 10.18 | |
| wr lds64 membar+sparse2 | 8.33 | 10.69 | 9.39 | |
| wr lds32 membar+sparse2 | 6.35 | 8.97 | **10.98** | old “best” — **not** 1×16 |

Optional future suite: measure **32×1 / 64×1 / 128×1** under `axis=Nx1` — separate from this table.

---

## 6. SEMANTICS_GATE

```
G16 SoftLib SIMT stock/best wc halfmask (±back): SEMANTICS_GATE: PASS — S1 Case-1 absmax gold G=16 E/G outs axis=1xG; S2 halfmask 16-of-32 + contig LDS.32; S4 VF named
G32 SoftLib SIMT stock/best wc (±dual_drain): SEMANTICS_GATE: PASS — S1 gold G=32 axis=1xG; S2 full-warp redux; S4 VF
G64/G128 SoftLib SIMT stock/best wc-multi: SEMANTICS_GATE: PASS — S1 gold G axis=1xG; S2 per-lane fold inside gold G + warp redux; S4 VF
SoftLib CCE / VMI: SEMANTICS_GATE: PASS — re-cite t685u RESULT.tsv; absmax-only; not reopened
G16 wr (appendix): SEMANTICS_GATE: PASS for Nx1 family only — NOT main 1×G SoftLib SIMT (§21 / pollution #24)
```

---

## 7. Paths

| Artifact | Path |
|----------|------|
| Live pack | `/mnt/fluxdata/happybot/glm53_softlib_simt_absmax_1xg_t698u/` |
| RESULT TSV | `.../reports/RESULT_1xg_t698u.tsv` · `/workspace/RESULT_1xg_t698u.tsv` |
| Case def | `/workspace/SV567_CASE_DEF_absmax_fp32.md` |
| 4-arm table | `/workspace/SOFTLIB_SIMT_absmax_4arm_table.md` |
| Harness §21 | `/workspace/REGRESSION_salvage_2026-09-17/HARNESS_REBUILD.md` (+ glm53 inputs copy) |
| Gen/run | `/workspace/_gen_1xg_t698u.py` · `_run_1xg_t698u.sh` · `_harvest_1xg_t698u.py` |

**Do not merge：** G16 wr 10.98 · Case-3 recip · foreign era VMI.
