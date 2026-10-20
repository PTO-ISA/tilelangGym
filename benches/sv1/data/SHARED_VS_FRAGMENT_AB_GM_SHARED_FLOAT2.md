# SV1 SimtVF: GM→shared add-in-shared vs fragment RF temps — **float2** A/B

**Date:** 2026-10-16 HKT (T128/T512) · **Redo T-dim:** 2026-10-20 HKT (T64–T2048) · **Host:** pto-b10 (edgexpert-59a6) · **SOC:** Ascend950PR_9599 opsim  
**Mode:** `ST_VF_MODE=simt-asc` · **Knob:** `ST_SV1_TEMPS=frag|shared` → tags `frag` / `gmshared`  
**Vectorization:** stock **float2** (`max_vector_bits=64` / `threadIdx.x*2`)  
**Remote:** `/tmp/sv1_gm_shared_ab_float2_20261016` · redo `/tmp/sv1_float2_tdim_E8192_20261020`  
**Box:** `/workspace/st_out_sv1_gm_shared_ab_float2/` · redo `/workspace/st_out_sv1_float2_tdim_E8192_20261020/`

## Lok correction (voice)

Prior shared A/B used **shared→shared staging** (`t1/t2/t3 = alloc_shared` then copy/add) — that was the wrong pattern.  
We **no longer have** shared→fragment `T.copy`. New shared pattern:

1. **Direct GM→shared** `T.copy(A/B → a_ub/b_ub)`
2. **Add IN shared:** `c_ub[i] = a_ub[i] + b_ub[i]` (no t1/t2/t3)
3. `T.copy(c_ub → C)`

Fragment arm kept as **RF temps baseline** (`alloc_fragment` t1/t2/t3) for comparison.

## Results (E=8192 TILE8192 float2)

| Case | temps | VF_cyc | el/cyc | wall µs | LDS | STS | FADD | IADD | opsim |
|------|-------|-------:|-------:|--------:|----:|----:|-----:|-----:|-------|
| T64 | **frag** | **4497** | **1.822** | 2.50 | 256 | **128** | 256 | 384 | PASS |
| T128 | **frag** | **3160** | **2.592** | 1.76 | 256 | **128** | 256 | 384 | PASS |
| T128 | **gmshared** | **3160** | **2.592** | 1.76 | 256 | **128** | 256 | 384 | PASS |
| T256 | **frag** | **2389** | **3.429** | 1.33 | 256 | **128** | 256 | 384 | PASS |
| T256 | **gmshared** | **2389** | **3.429** | 1.33 | 256 | **128** | 256 | 384 | PASS |
| T512 | **frag** | **2205** | **3.715** | 1.23 | 256 | **128** | 256 | 384 | PASS |
| T512 | **gmshared** | **2205** | **3.715** | 1.23 | 256 | **128** | 256 | 384 | PASS |
| **T1024** | **frag** | **2130** | **3.846** | **1.18** | 256 | **128** | 256 | 384 | **PASS** |
| **T1024** | **gmshared** | **2130** | **3.846** | **1.18** | 256 | **128** | 256 | 384 | **PASS** |
| T2048 | **frag** | **2163** | **3.787** | 1.20 | 256 | **128** | 256 | 384 | PASS |
| T2048 | **gmshared** | **2163** | **3.787** | 1.20 | 256 | **128** | 256 | 384 | PASS |

| Case | Δ el/cyc (gmshared vs frag) |
|------|----------------------------:|
| T128 / T256 / T512 / T1024 / T2048 | **0%** (tied) |

**Peak:** TILE8192 stock float2 **T1024 = 3.846** el/cyc (~**48%** of LD.64 A+B roof ~8). T2048 = 3.787 (above T512 3.715).  
**Verdict:** Under GM→shared / add-in-shared, **shared matches fragment**. Prefer either; no shared penalty.

## Instr mix (opcodes `SIMT_*`)

| | SIMT_LDS | SIMT_STS | SIMT_FADD | SIMT_IADD_I |
|--|--------:|---------:|----------:|------------:|
| frag / gmshared | 256 | 128 | 256 | 384 |

## vs prior (wrong) shared staging — tower only

Prior shared arm had `t1/t2/t3 = alloc_shared` → **STS 128→512** (−37%/−50% el/cyc). See **`SHARED_VS_FRAGMENT_AB.md`** — **tower / staging only; do not cite as SDS**.

## Artifacts

- `SHARED_VS_FRAGMENT_AB_GM_SHARED_FLOAT2.{md,tsv,json}`
- Reconcile note: `/workspace/pr7_tdim_reconcile_E8192.md`
- Legacy tower: `SHARED_VS_FRAGMENT_AB.md` (**staging only — not SDS**)
