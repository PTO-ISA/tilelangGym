# SV1 analysis — SIMD DSL keynote + Simt note

**Date:** 2026-10-15 HKT · **SOC:** Ascend950PR_9599 opsim

## Keynote — handwritten SIMD DSL (`sv1d_stream_eltwise.py`)

SIMD DSL maps to **UB vector load / store in a small VL loop**. No per-thread index math, **no unroll**, and **no instruction scheduling** are required for good throughput.

| Item | Value |
|------|-------|
| Pattern | VL=64 fp32 · **2R+1W** VADD-style body (VLDI / VADD / VSTI) |
| **Roof (Lok)** | UB R+W ≈ **512 B/cyc** → 768 B/chunk ÷ 1.5 cyc = **~44 el/cyc** (exact **42.67**) |
| Body mid peak | **~36 el/cyc ≈ 85%** of roof |
| Body sustained | **~34 el/cyc ≈ 80%** of roof |
| VF (E=16K SUMMARY) | **31.5 el/cyc** (~74% of 42.67) |

**Takeaway:** keep the kernel as a short UB ld/st loop; performance already sits near the mem roof without schedule work.

Kernels: `benches/sv1/sv1d_stream_eltwise.py` (DSL) · `benches/sv1/sv1_stream_eltwise.py` (`ST_VF_MODE` TileLang).

## SimtVF (brief)

| Item | Value |
|------|-------|
| **Simt roof** | **16 el/cyc** (UB **1R+1W @ 128B each**) |
| Pack | **LDS/STS.64 only** (no `.128` from `T.Parallel`) |
| Stock best | untiled T512 ≈ **3.71** el/cyc (**23%** of 16) |
| ILP16 @ T=128 | **4.08** el/cyc (**25.5%** of 16); mid ~**4.5** — still mid-body stall / IADD tax, not launch |

Simt needs schedule work at fixed T=128 to climb toward roof 16; SIMD already does not.

## Todos — Simt T=128 scheduling (not SIMD)

- [ ] Deeper ILP at fixed T=128 (keep `.64`)
- [ ] Cut IADD tax; raise mid-median Body4 IPC toward roof **16**
- [ ] I-cache / MRF checks on deeper unroll

## Weekly gate

`scripts/regression/oneshot_sv1_e8192_t128_t512.sh` — E=8192, T=128 + T=512, `ST_VF_MODE=simt-asc`.
