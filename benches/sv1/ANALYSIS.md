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

---

## Detail — instr mix, running IPC, overhead / bubbles

Scope: E=16384 fp32 **2R+1W** stream on Ascend950PR_9599 opsim unless noted. Host launch is **outside** VF; SUMMARY el/cyc = E / VF_cyc.

### Instr mix

| Path | Useful mem / ALU | Addr / tax | Pack | Pattern |
|------|------------------|------------|------|---------|
| **SIMD DSL** (`sv1d`, VL=64) | VLDI ×2 + VADD + VSTI per VL chunk | **none** (no per-thread index) | VL=64 | small UB ld/st loop |
| **Simt** stock / ILP16 | LDS:FADD:STS = **512:512:256** (.64) | IADD(+I) ≈ **740–1504** (~0.9–1.5× memF) | **.64 only** (no `.128` from `T.Parallel`) | per-thread index + IADD tax |

**SIMD (E16K):** body instr **1029** / VF **520** ⇒ IPC ≈ **1.98** (whole-VF). Mid-window vec IPC ≈ **2.13** (peak ≈ **2.27**) → add ≈ **0.53–0.57** → body **~34–36 el/cyc**.

**Simt ILP16 body4 rows:** LDS 512 + FADD 512 + STS 256 + IADD_I 740 (FADD share of body rows ≈ **25%**; stock mix roughly LDS~28% / STS~14% / IADD~31% / FADD~28%).

### Running IPC / el/cyc

| Arm | VF_cyc | el/cyc | Body4 / mid IPC | % of path roof |
|-----|-------:|-------:|-----------------:|---------------:|
| SIMD DSL E16K VL64 | 520 | **31.5** | mid vec IPC ≈ **2.13** (peak ≈ **2.27**); whole-VF ≈ **1.98** | **~74%** of 42.67 (body mid ~**85%**) |
| Simt untiled T512 | 4415 | **3.71** | Body4 mean ≈ **0.47**; mid-med ≈ **0.37–0.47** | **23%** of 16 |
| Simt ILP16 @ T128 | 4015 | **4.08** | mean **0.54** / mid-med **0.47** / peak **1.87**; mid sustained ~**4.5** el/cyc | **25.5%** of 16 |

Peak Body4 IPC (**1.87**) is **short bursts** — mid-median (**~0.47**) and ~**65–69%** idle set wall time, not the spikes.

### Overhead / bubbles

**SIMD DSL (E16K):** VF wrap **520** vs body **~482** ⇒ ~**38c** pro+epi (~7% of VF). Body is near-roof; gap to 42.67 is mostly that wrap + small mid slack, **not** schedule.

**Simt ILP16 @ T128 (VF-scoped):**

| Piece | Share of VF | Note |
|-------|------------:|------|
| Prologue | **~0.8–1%** | S2R / SHFI / LEA |
| Mid body (70% span) | **~65%** | rate that matters |
| Head+tail ramp | **~28%** | schedule ramp inside body |
| Epilogue | **~6.8–7%** | SET_FLAG / MOV |
| Host launch | **0% of VF** | not in 4.08 el/cyc |

Idle / issue width (stock untiled, Body4 window):

| Arm | % idle (width 0) | mean issue width | idle bubble p50 / p90 / max |
|-----|-----------------:|-----------------:|----------------------------|
| T128 | **~69%** | **0.38** | 1 / 7 / 244 |
| T512 | **~68%** | **0.47** | 1 / 6 / 32 |

Wait tax: ~**35–36%** of ops have non-zero wait; LDS wait_nz ≈ **44%**, FADD wait_nz ≈ **50%** (ILP16 can cut FADD wait share toward ~**13%**). Gap to roof **16** is **mid-body stall / IADD / issue**, not launch/head–tail (those only cost ~**0.43** el/cyc vs mid ~4.5 → VF 4.08).

### TODO / remarks

**Simt T=128 schedule (open):**

- [ ] Deeper ILP at fixed T=128 (keep `.64`; no warp-yield)
- [ ] Cut IADD tax; raise **mid-median** Body4 IPC toward roof **16** (not peak spikes)
- [ ] I-cache / code-size check on deeper unroll
- [ ] MRF live-set check if full-unroll E=8K (~96 KB vs 128 KB MRF)

**Remarks:**

- SIMD already near mem roof without schedule work — keep as the SV1 mapping keynote.
- Simt pack stays `.64` from `T.Parallel`; `.128` is a GPU-side / hand path, not current Ascend lowering.
- Do not treat deferred simt-pto SoftLib history as the work queue; gate is **simt-asc** + handwritten SIMD DSL on remat tip.
- Weekly: `scripts/regression/oneshot_sv1_e8192_t128_t512.sh`.

### How to re-measure

```bash
bash scripts/regression/oneshot_sv1_e8192_t128_t512.sh   # weekly SV1 gate
bash scripts/oneshot/oneshot_sv1_sv9.sh                  # simt-asc SV suite
bash scripts/oneshot/oneshot_ptodsl_sv1_sv9d.sh          # SIMD-DSL twins
```
