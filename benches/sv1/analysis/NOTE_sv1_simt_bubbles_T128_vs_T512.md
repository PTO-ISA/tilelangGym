# SimtVF untiled E=16K — issue bubbles T=128 vs T=512

**Date:** 2026-10-14 HKT · pack **.64** · Body4 = LDS+STS+IADD+FADD · issue-cap **4**  
**Tags:** `sv1t_e16384_t128` / `sv1t_e16384_t512` (TILE=E=16384, **NOT** 8192)  
**Keeps:** `/workspace/sv1_sv_analysis/data/sv1t_e16384_t{128,512}/` (re-run OPPROF 2026-10-14)

## Verdict

1. **YES — issue bubbles are dependency / scoreboard waits.**  
   - FADD `waitBitMask≠0` = **50%** both T (hist mostly `3`, also `5`/`7`) → RAW on prior LDS (and sometimes IADD).  
   - LDS also waits **~44%** (`waitBitMask=2/4`) — addr / prior IADD chain.  
   - STS is clean (`wait=0`); IADD_I waits **~32–34%**.  
   - Chrome-trace `stallCyc` on FADD ≈1 (issue penalty once dep clears).

2. **YES — T=512 reduces bubbles, but modestly on average idle; the big win is cutting pathological long bubbles + raising multi-issue.**  
   - Idle (width=0): **69.4% → 67.6%** (−1.8 pp) — still mostly idle.  
   - Mean idle-bubble: **2.61 → 2.46** cyc.  
   - **Max idle-bubble: 244 → 32** (−87%) — T128 has a long fill/drain hole; T512 hides it.  
   - Width&lt;2 cycles: **93.2% → 88.6%**; width=3: **4 → 100** cycles; width=4: **2 → 13**.  
   - Body4 IPC: **0.376 → 0.468** (+24%, still only **11.7% of cap=4**).  
   - el/cyc VF: **2.87 → 3.71**.

So T↑ helps latency hiding (fewer long stalls, slightly more co-issue) but does **not** approach the 4-wide roof — dep chain + IADD tax still dominate.

## Quantified (body = 1st LDS → last STS issue)

| Metric | T128 | T512 | Δ |
|--------|-----:|-----:|--:|
| VF_cyc | 5700 | 4415 | -1285 |
| body_span_cyc | 5373 | 4342 | -1031 |
| Body4 count | 2020 | 2032 | +12 |
| Body4 IPC | 0.376 | 0.468 | +0.092 |
| % of cap=4 | 9.40 | 11.70 | +2.30 |
| % cycles width=0 (idle) | 69.37 | 67.57 | -1.79 |
| % cycles width&lt;2 | 93.23 | 88.58 | -4.65 |
| % cycles width&lt;4 | 99.96 | 99.70 | -0.26 |
| mean issue width | 0.376 | 0.468 | +0.092 |
| idle bubbles n | 1429 | 1195 | -234 |
| idle bubble mean len | 2.608 | 2.455 | -0.153 |
| idle bubble max | 244 | 32 | -212 |
| idle bubble p50 / p90 | 1 / 7 | 1 / 6 | — |
| LDS→FADD gap mean (same sch+warp) | 2.50 | 2.83 | +0.32 |
| LDS→FADD gap p50 (sch+warp) | 1.80 | 3.60 | — |
| FADD→STS gap mean (sch+warp) | 4.87 | 55.50 | — |
| el/cyc VF | 2.874 | 3.711 | +0.837 |
| FADD % waitBitMask≠0 | 50.0 | 50.0 | 0 |
| LDS % waitBitMask≠0 | 43.8 | 43.8 | 0 |
| IADD_I % waitBitMask≠0 | 34.1 | 31.9 | -2.1 |

### Issue-width hist (Body4 issues / cycle)

| width | T128 cycles | T512 cycles |
|------:|------------:|------------:|
| 0 | 3727 | 2934 |
| 1 | 1282 | 912 |
| 2 | 358 | 383 |
| 3 | 4 | 100 |
| 4 | 2 | 13 |

## waitBitMask / stallCyc (dependency evidence)

### T128

| Class | n | % wait≠0 | wait hist | mean stallCyc |
|-------|--:|--------:|-----------|--------------:|
| SIMT_LDS | 512 | 43.8 | `{'0': 288, '2': 220, '4': 4}` | 2.01 |
| SIMT_FADD | 512 | 50.0 | `{'3': 224, '0': 256, '5': 28, '7': 4}` | 1.05 |
| SIMT_STS | 256 | 0.0 | `{'0': 256}` | 2.00 |
| SIMT_IADD_I | 740 | 34.1 | `{'0': 488, '4': 224, '2': 28}` | 2.76 |

### T512

| Class | n | % wait≠0 | wait hist | mean stallCyc |
|-------|--:|--------:|-----------|--------------:|
| SIMT_LDS | 512 | 43.8 | `{'0': 288, '2': 208, '4': 16}` | 2.03 |
| SIMT_FADD | 512 | 50.0 | `{'3': 224, '0': 256, '5': 16, '7': 16}` | 1.22 |
| SIMT_STS | 256 | 0.0 | `{'0': 256}` | 2.00 |
| SIMT_IADD_I | 752 | 31.9 | `{'0': 512, '4': 224, '2': 16}` | 2.70 |

**Reading:** Chain is **IADD (addr) → LDS.64 → FADD → STS.64**. FADD wait bits `(3, 5, 7)` are scoreboard RAW on LDS (and sometimes IADD). LDS wait `(2, 4)` is upstream addr/IADD. STS has no wait — it issues when FADD result is free, but the **idle holes** between producer complete and next Body4 issue are the visible bubbles. Same-warp LDS→FADD gap stays ~few cycles; T512 wins by overlapping **other warps** into those holes (max bubble 244→32, more width=2/3/4).

## Plots

- `plots/sv1_simt_bubble_T128_vs_T512.png` — quant compare  
- `plots/sv1_simt_bubble_dep_chain_T128.png` — timeline slice (sch0) with wait labels  
- `plots/sv1_simt_bubble_dep_schematic.png` — LDS→bubble→FADD→bubble→STS schematic  

## Method

Re-ran `sv1t_e16384_t{128,512}` on pto-b10 with `sim_dsl` + `PTOAS_KEEP_MSPROF_STAGING=1` (SO already present). Kept `core0.veccore0/trace.json` + `instr_exe.csv`. Calibrated µs→cyc from VF cycles/dur. Issue width = Body4 issues on the same rounded cycle. Idle bubble = consecutive width=0 runs. Dep gaps = consecutive LDS→FADD / FADD→STS on the **same (schId, warpId)**.


## Caveat: max idle 244 is late-drain, not typical

Lok flagged that **250 idle for T128 seems wrong** — correct instinct for *typical* idle.

- Distribution: **mean 2.61 / p50=1 / p90=7 / p99=8**; **99.6% of bubbles ≤8**.
- The **244** is **one** bubble at body **~91%** (cyc [4934,5177]) = **late DRAIN** with only ~4% Body4 left — **not** VF prologue, **not** mid-body.
- Mid-body max idle (exclude first/last 5% or 10%): **8**.
- Detail + hist: `NOTE_sv1_simt_T128_idle_bubble_location.md`, plot `plots/sv1_simt_T128_idle_bubble_location.png`.
