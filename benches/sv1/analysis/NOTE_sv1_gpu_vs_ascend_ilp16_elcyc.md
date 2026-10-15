# GPU T=128 float4/.128 reschedule el/cyc vs Ascend ILP16

**Date:** 2026-10-15 HKT · **GPU:** GB10 sm_121 (pto-b10) · **Ascend:** camodel SimtVF ILP16@T128

## Method

- GPU: `ncu` `sm__cycles_active.max` / `sm__cycles_elapsed.max` (same as TileLang SUMMARY).
- `elems_per_cyc = 16384 / cycles_*`
- Hand kernels: float4 → SASS **LDG.E.128 / STG.E.128 only** (census: LDG.E.128=98, LDG.E.64=0).
- Ascend ILP16: keep-trace VF_cyc=4015 → **4.081 el/cyc**, roof **16** (Lok 1R+1W@128B).

## Results

| Case | Pack | Scope | cycles_active | **el/cyc (active)** | el/cyc (elapsed) | vs Ascend 4.08 | notes |
|------|------|-------|--------------:|--------------------:|-----------------:|---------------:|-------|
| **Ascend ILP16 @ T128** | **.64** | 1 SimtVF | 4015 | **4.081** | — | 1.00× | **25.5% of roof 16** |
| Ascend roof | .64@128B R+W | UB | — | **16** | — | 3.92× | Lok |
| GPU TL tcopy TILE2048@T128 | float4 CUDA (.64 SASS dump) | 1 CTA | 15827 | **1.035** | 0.849 | 0.25× | prior SUMMARY |
| GPU hand **unroll** float4 | **.128** | multi-CTA | 7617 | **2.151** | 1.476 | 0.53× | grid-stride |
| GPU hand **ILP4** float4 | **.128** | multi-CTA | 8457 | **1.937** | 1.366 | 0.47× | 8×LDG before FADD |
| GPU hand ILP16 float4 | **.128** | multi-CTA | 17408 | 0.941 | 0.779 | 0.23× | under-subscribed grid |
| GPU hand **ILP4 1CTA** | **.128** | **1×T=128** | 18673 | **0.877** | 0.739 | 0.21× | fairest vs Ascend T |
| GPU hand ILP16 1CTA | **.128** | 1×T=128 | 21456 | 0.764 | 0.657 | 0.19× | |
| GPU recreate tcopy2048 | **.128** | 1 CTA | 27355 | 0.599 | 0.532 | 0.15× | worse than TL dump |

## Reading

1. **Ascend ILP16 4.08 ≫ GPU 1CTA float4/.128 (~0.88)** and ≫ TileLang GPU **1.04** on the same E=16K T=128 shape — different ISA/roofs; Ascend is **~4×** the GPU 1CTA el/cyc and **~25% of its own roof 16**, GPU TL is **~1 el/cyc**.
2. Best GPU float4/.128 **schedule** win vs TL baseline: multi-CTA hand unroll **2.15** (active) / **1.48** (elapsed) — still **≪ Ascend 4.08**, and multi-CTA is not 1:1 with one SimtVF.
3. SASS confirms reschedule path is **.128**; TileLang shipped dump can still be **.64** despite float4 source.
4. Do **not** claim Ascend is “faster silicon” from this — Ascend number is **camodel VF**, GPU is **real GB10**, roofs differ (Ascend UB 16 el/cyc vs GPU DRAM/L2).

## Roofs (apples ≠ oranges)

| Platform | Roof used here | Meaning |
|----------|----------------|---------|
| Ascend SimtVF | **16 el/cyc** | 1R+1W @ 128B/cyc on UB (Lok) |
| GPU | no single matched roof | Chip DRAM BW ≫ measured; 1CTA limited by latency/issue, not peak HBM |

Rough GPU chip BW sketch (order-of-magnitude only): if ~273 GB/s @ ~2.4 GHz and 12 B/el (2R+1W float) → ~**9–10 el/cyc** chip-wide BW roof; measured multi-CTA ~1.5–2.2 elapsed is still a small fraction — not the binding story for 1CTA.

## Artifacts

```text
pto-b10:/tmp/sv1_gpu_f4_bench/
  bench_f4_ilp_E16k.cu  bench_f4  ncu_cycles.csv  bench_log.txt  bench.sass
/workspace/sv1_sv_analysis/data/
  NOTE_sv1_gpu_vs_ascend_ilp16_elcyc.md
  SUMMARY_gpu_f4_ilp_vs_ascend.json
  NOTE_sv1_gpu_sass_unroll_T128.md  (SASS qualitative)
```
