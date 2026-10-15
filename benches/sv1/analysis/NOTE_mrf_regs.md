# MRF / live registers after SimtVF reschedule (T=128)

**Date:** 2026-10-15 HKT · A5 **MRF = 128 KB**

## Lok UB footprint (data buffers)

| Scope | 1 input | A+B | A+B+C |
|-------|--------:|----:|------:|
| **E=8K** | 32 KB | 64 KB | **96 KB** |
| E=16K (this bench) | 64 KB | 128 KB | **192 KB** |

- **E=8K full resident A+B+C = 96 KB → fits in 128 KB MRF** with **32 KB headroom** for live regs / other.
- **E=16K A+B+C = 192 KB > 128 KB** — cannot hold full inout in MRF; must tile/UB-stage. Reg pressure is *on top of* whatever slice is resident.

## Software ILP live regs (theoretical peak)

Classic schedule keeps `a[0..K)`, `b[0..K)`, `c[0..K)` → **~3K float2 / thread** (8B each).

| ILP K | peak float2/thr | bytes/thr | × T=128 | % of 128KB MRF |
|------:|----------------:|----------:|--------:|---------------:|
| 4 | 12 | 96 | 12 KB | 9.4% |
| 8 | 24 | 192 | 24 KB | 18.8% |
| **16** | **48** | **384** | **48 KB** | **37.5%** |
| 32 | 96 | 768 | 96 KB | 75.0% |
| 64 | 192 | 1536 | 192 KB | **150% — exceeds MRF if all threads' temps resident** |

### Fit with E=8K data (96 KB buffers + regs)?

| ILP | buffers+regs (if both count in MRF) | fit 128KB? |
|-----|-------------------------------------:|:----------:|
| 16 | 96 + 48 = **144 KB** | **NO** if regs fully extra on top of 96KB data |
| 16 | if temps overlay / not all 128 thr simultaneous | maybe |
| 32 | 96 + 96 = **192 KB** | **NO** |
| 8 | 96 + 24 = **120 KB** | **YES** (tight) |

**Reading:** Full-unroll **ILP64** reg footprint alone (192 KB @ T128) **does not fit** A5 128KB MRF under the all-threads-live assumption. **ILP16** regs (48 KB) fit *alone* but **96+48 > 128** if E=8K inout and temps are both charged to MRF. Prefer **ILP≤8** for comfortable E=8K+regs, or confirm HW shares/overlays MRF banks so not all threads' temps are co-resident.

## Measured (opsim)

See `SUMMARY_mrf_regs.json` — `simt.mrf` dump string hits + `vf_sym_size` from `aicore_binary.o`.
