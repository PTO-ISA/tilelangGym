# SV1 SimtVF — `alloc_shared` temps vs `alloc_fragment` temps A/B

> **Tower / staging only — do not cite as SDS.** This file is the old `alloc_shared` t1/t2/t3 staging arm (−37%/−50% vs frag). Correct SDS (direct GM→shared, frag≡gmshared tie; peak T1024=3.846) is `SHARED_VS_FRAGMENT_AB_GM_SHARED_FLOAT2.md` / ANALYSIS §8.


**Date:** 2026-10-16 HKT · **Host:** pto-b10 (`edgexpert-59a6`) · **SOC:** Ascend950PR_9599 opsim  
**Mode:** `ST_VF_MODE=simt-asc` · **CANN:** cann_92b2 (`/usr/local/Ascend/cann_92b2/cann-9.2.0-beta.2`)  
**TileLang deps:** `/mnt/fluxdata/happybot/projects/tilelang-pto-isa-st` · **CAMO:** `.camodel_deps_vmi018`  
**Remote OUT:** `/tmp/sv1_shared_frag_ab_20261016` · **Box mirror:** `/workspace/st_out_sv1_shared_frag/`

## Aim

For this short-live stream eltwise, Lok expected **register/layout differs** but **performance ≈ no diff** between temps in UB (`alloc_shared`) vs RF (`alloc_fragment`). Measure both.

## Kernel override

`sv1_stream_eltwise.py` now honors **`ST_SV1_TEMPS`**:

| value | temps | tag suffix |
|-------|-------|------------|
| `fragment` / `frag` / `rf` | `alloc_fragment` | `_frag` |
| `shared` / `ub` / `ubuf` | `alloc_shared` | `_shared` |
| unset | simt→fragment, simd→shared (legacy) | `_frag` / `_shared` |

Default simt path unchanged when env unset.

## Results (SimtVF)

| tag | temps | compile | opsim | VF_cyc | el/cyc | wall µs | LDS | STS | FADD | IADD | layout |
|-----|-------|---------|-------|-------:|-------:|--------:|----:|----:|-----:|-----:|--------|
| sv1_e8192_t128_frag | frag | OK | PASS maxabs=0 | **4680** | **1.750** | 2.60 | 128 | 64 | 256 | 192 | RF `t1/t2/t3[64]` |
| sv1_e8192_t128_shared | shared | OK | PASS maxabs=0 | **7450** | **1.100** | 4.14 | 128 | **256** | 256 | 384 | no RF arrays; UB ptr temps |
| sv1_e8192_t512_frag | frag | OK | PASS maxabs=0 | **3405** | **2.406** | 1.89 | 128 | 64 | 256 | 192 | RF `t1/t2/t3[16]` |
| sv1_e8192_t512_shared | shared | OK | PASS maxabs=0 | **6840** | **1.198** | 3.80 | 128 | **256** | 256 | 384 | no RF arrays; UB ptr temps |
| sv1_e16384_t128_frag | frag | OK | PASS maxabs=0 | **9031** | **1.814** | 5.02 | 256 | 128 | 512 | 384 | RF `t1/t2/t3[128]` |
| sv1_e16384_t128_shared | shared | OK | **FAIL** maxabs≈492 | — | — | — | — | — | — | — | UB temps; likely UB OOB |

### Perf delta (shared vs frag, same E×T)

| Case | frag el/cyc | shared el/cyc | Δ el/cyc | VF_cyc shared/frag |
|------|------------:|--------------:|---------:|-------------------:|
| E8192 T128 | 1.750 | 1.100 | **−37%** | **+59%** cycles |
| E8192 T512 | 2.406 | 1.198 | **−50%** | **+101%** cycles |

**Verdict vs pre-measure expectation (“≈ no perf diff”):** **Not confirmed.** Shared temps are **materially slower** on every PASS pair (≈ **37–50%** lower el/cyc). Layout differs as expected.

## Layout evidence (lowered CCE)

### Fragment (E8192 T128) — per-thread RF arrays

```c
float t1[64];
float t2[64];
float t3[64];
// LDS A/B → RF, FADD in RF, STS result → C UB
```

A/B/C still live in `__ubuf__` via `buf_dyn_shmem` (3×32 KB = 96 KB). Temps are **not** UB.

### Shared (E8192 T128) — temps as UB pointer traffic

No `float tN[…]`. Body is pure `__ubuf__ float4*` traffic through `buf_dyn_shmem` with extra offsets for t1/t2/t3 (e.g. +24576 / +32768 / +40960 floats). That produces **4× STS** vs frag (256 vs 64 at E8192) — each temp write hits UB.

### E16384 shared FAIL

A+B+C+t1+t2+t3 ≈ **6 × 64 KB = 384 KB** ≫ ~128 KB UB/MRF. Frag arm keeps temps in RF so UB stays at A+B+C ≈ 192 KB (still tight) but opsim PASS; shared arm fails numerically (`maxabs≈492`).

## Opcode note

Counts are `SIMT_LDS` / `SIMT_STS` / `SIMT_FADD` / `SIMT_IADD*` from `core0.veccore0_instr_exe.csv`. Shared’s extra STS matches UB temp stores; FADD count matches (same arithmetic work).

## Artifacts

- Box report: this file · `SHARED_VS_FRAGMENT_AB.tsv` · `SHARED_VS_FRAGMENT_AB.json`
- Sources: `/workspace/st_out_sv1_shared_frag/sources/`
- Remote keep: `/tmp/sv1_shared_frag_ab_20261016/` (so/, opsim_*, logs/, sources/)
- Kernel patch: `benches/sv1/sv1_stream_eltwise.py` (`ST_SV1_TEMPS`) — **not committed**

## Bottom line

| Expectation (Lok) | Measured |
|-------------------|----------|
| Reg / layout differs | **Yes** — frag=`float tN[E/T]`; shared=UB ptr temps |
| Perf ≈ no diff | **No** — shared **−37% / −50%** el/cyc @ E8192 T128 / T512; E16K shared **COMPILE_OK but opsim FAIL** |
