# PASS + perf — Simt vs SIMD-DSL (one case / row)

**Date:** 2026-10-15 HKT · **SOC:** Ascend950PR_9599 · **Pin:** TileLang remat tip `a3012088` + [tilelang#5](https://github.com/PTO-ISA/tilelang/pull/5)

**All rows below are PASS** on this pin (simt-asc gate + handwritten SIMD DSL). Historical red / compile-error tags are not listed.

| Column | Meaning |
|--------|---------|
| **Simt µs** | `core0.veccore0` wall (best simt-asc arm at the shape) |
| **SIMD-DSL µs** | handwritten `*d` PTO-DSL twin (best arm) |
| **Ratio** | Simt µs ÷ SIMD-DSL µs (>1 ⇒ DSL faster wall) |
| **VF ratio** | Simt VF_cycles ÷ DSL VF_cycles (>1 ⇒ DSL fewer VF cycles) |

Per-case highlight: `benches/<id>/VERDICT.md` (SV1: [`ANALYSIS.md`](../benches/sv1/ANALYSIS.md)).

---

## SV1 — SIMD keynote

SIMD DSL = small UB ld/st VL loop — **no per-thread index, no unroll, no instr schedule** — body mid ~**85%** of ~**44** el/cyc roof (Lok **42.67**). Simt roof **16** el/cyc; best ~**4.08** (ILP16 @ T128).

| Case | Shape / arm | Status | Simt µs | SIMD-DSL µs | Ratio | Simt el/cyc | DSL el/cyc | Note |
|------|-------------|--------|--------:|------------:|------:|------------:|-----------:|------|
| SV1 | E256 stream | **PASS** | 1.21 | 0.89 | **1.36** | — | 4.27 | gate |
| SV1 | E8192 VL64 vs Simt T128 ILP16 | **PASS** | — | 2.59 | — | **4.08** | **29.3** | DSL ~7× el/cyc vs Simt mid |
| SV1 | E16384 untiled T512 vs DSL VL64 | **PASS** | 6.73 | 4.55 | **1.48** | **3.71** | **31.5** | DSL ~74% of 42.67 roof |

---

## SV2–SV9 — best arm / matched shape

Sources: opsim harvest 2026-10-07/08 (`VF_CYCLES_SIMT_VS_DSL_BEST`); remat-tip recheck stamp on pin above.

| Case | Shape / arm | Status | Simt µs | SIMD-DSL µs | Ratio | VF ratio | Note |
|------|-------------|--------|--------:|------------:|------:|---------:|------|
| SV2 | R32×C64 input_stream | **PASS** | 3.60 | 1.01 | **3.56** | 28.7 | stream primary |
| SV2 | R32×C128 input_stream | **PASS** | 8.49 | 1.34 | **6.34** | 53.4 | wide-C |
| SV2 | R16×C32 input_keep | **PASS** | 1.16 | — | — | — | ABI fallback; no *d at C32 |
| SV3 | M24 VL64 K16 keep | **PASS** | 6.79 | 3.14 | **2.16** | 11.1 | Acc KEEP |
| SV3 | M32 VL64 K16 split_cm16 | **PASS** | 7.79 | 6.88 | **1.13** | 1.61 | chunk flush |
| SV4 | E256 B8 remat_idx_calc / keep_idx | **PASS** | 1.62 | 1.02 | **1.59** | 9.63 | remat tip clears remat_idx |
| SV4 | E512 B8 keep_idx | **PASS** | 2.83 | 1.25 | **2.26** | — | ladder |
| SV4 | E1024 B8 remat_idx_calc / keep_idx | **PASS** | 6.30 | 2.32 | **2.72** | — | remat wins −12% vs keep |
| SV5 | R64 C128 G16 ub_stream | **PASS** | 4.18 | 13.06 | **0.32** | 0.33 | Simt wins (Case-3) |
| SV6 | R64 C128 G32 reload | **PASS** | 12.58 | 11.66 | **1.08** | 1.26 | Case-3 |
| SV7 | R64 C128 G64 multiwarp_ub | **PASS** | 7.32 | 11.10 | **0.66** | 0.67 | Simt multiwarp wins |
| SV7 | R64 C128 G128 reload | **PASS** | 14.58 | 10.79 | **1.35** | 1.47 | DSL reload wins |
| SV8 | R64 C128 G16 live | **PASS** | 17.04 | 19.56 | **0.87** | 0.89 | near parity |
| SV9 | E256 K8 keep / remat_scores | **PASS** | 1.48 | 1.00 | **1.48** | 4.14 | DSL remat_scores best |

---

## CF — control-flow (primary)

| Case | Shape / arm | Status | Simt µs | SIMD-DSL µs | Ratio | VF ratio |
|------|-------------|--------|--------:|------------:|------:|---------:|
| CF1 | E256 K8 keep | **PASS** | 0.97 | 0.90 | **1.08** | 2.94 |
| CF2 | E256 K8 remat | **PASS** | 1.95 | 0.90 | **2.17** | 18.1 |
| CF3 | E256 K8 remat_idx | **PASS** | 1.33 | 0.83 | **1.60** | 8.10 |
| CF6 | E256 pnear5 newton | **PASS** | 1.37 | 0.93 | **1.47** | 7.45 |

---

## SP — sparse / packing (where *d twin exists)

| Case | Shape / arm | Status | Simt µs | SIMD-DSL µs | Ratio | VF ratio | Note |
|------|-------------|--------|--------:|------------:|------:|---------:|------|
| SP1 | T32 K2 H128 G32 remat_pos / keep_pos | **PASS** | 11.19 | 19.04 | **0.59** | 0.55 | Simt wins |
| SP2 | T32 K2 H128 G32 sf0_w0+keep_acc | **PASS** | 9.19 | 2.02 | **4.55** | 16.4 | |
| SP3 | M32 H128 G32 fp32 | **PASS** | 7.45 | 1.44 | **5.17** | 32.0 | |
| SP3 | M32 H128 G32 e8m0 | **PASS** | 7.45 | 1.39 | **5.36** | 35.9 | |
| SP6 | N32 H128 G32 unpack / vselr | **PASS** | 5.35 | 1.22 | **4.39** | 19.9 | |

SP4 / SP5: simt-asc **PASS**; no handwritten `*d` twin in-tree — omitted from ratio columns.

---

## Verdict (rollup)

- **SV1:** prefer handwritten SIMD DSL for stream eltwise — high UB BW, no index/schedule tax (~**85%** of ~44 el/cyc).
- **SV2–SV4 / CF / SP2–SP3 / SP6:** DSL usually fewer VF cycles (ratio ≫1); wall often follows.
- **SV5–SV8 Case-3:** mixed — Simt multiwarp / ub_stream can beat DSL wall; report both arms.

---

## SV1 detail (IPC / mix / bubbles)

Full instr-mix, running IPC, overhead/bubbles, and Simt T128 TODOs live in the case doc:

[`benches/sv1/ANALYSIS.md`](../benches/sv1/ANALYSIS.md) — keynote up front; **Detail** section at the end.

## How to re-measure

```bash
bash scripts/regression/oneshot_sv1_e8192_t128_t512.sh   # weekly SV1 gate
bash scripts/oneshot/oneshot_sv1_sv9.sh                  # simt-asc SV suite
bash scripts/oneshot/oneshot_ptodsl_sv1_sv9d.sh          # SIMD-DSL twins
```
