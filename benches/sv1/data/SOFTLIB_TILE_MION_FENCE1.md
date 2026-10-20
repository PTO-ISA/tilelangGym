# SoftLib MI-on + one-barrier tile ladder (E=16384, T=256, A+B LDS.b32 fence1)

- **OUT:** `/tmp/st_out_cce_softlib_tile_mion_fence1_20261020`
- **Date:** 2026-10-20 HKT · cann_92b2 · Ascend950PR_9599
- **Recipe:** per-tile GM→UB A/B → SimtVF loadall A+B → fence1 (`asm volatile("" ::: "memory")`) → FADD → STS → UB→GM C
- **Sync:** SoftLib `SetFlag`/`WaitFlag` with CBT-equivalent MTE3↔V priming (v4)
- **MI:** ON · **T=256** · **A+B** (not A-only)
- **Skip TILE16K:** A+B+C = 192 KB > UB/MRF 128 KB
- **Serial roof A+B LDS.b32:** 10.667

| TILE | trips | VF_SIMT calls | VF_cyc | el/cyc | %ser 10.667 | UB A+B+C (KB / %128) | PASS |
|-----:|------:|--------------:|-------:|-------:|------------:|---------------------:|:----:|
| 2048 | 8 | 8 | 3268 | **5.014** | 47.0% | 24.0 / 18.8% | PASS |
| 4096 | 4 | 4 | 2555 | **6.413** | 60.1% | 48.0 / 37.5% | PASS |
| **8192** | **2** | **2** | **2155** | **7.603** | **71.3%** | **96.0 / 75.0%** | **PASS** |

## Per-VF body cycles (fill asymmetry)

| TILE | VF cycle list | note |
|-----:|---------------|------|
| 2048 | 1119 + 7×307 | first VF warm |
| 4096 | 968 + 3×529 | first VF warm |
| 8192 | 1120 + 1035 | near untiled E8192 fence1 VF=1119 |

## vs stock CBT ~3.9

SoftLib TILE8K **7.603** ≈ **1.95×** CBT TILE2K **3.898** on E=16K; approaches SoftLib untiled T256 fence1 **7.32** / T512 **7.42**.
