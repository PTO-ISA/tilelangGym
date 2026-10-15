# SV1d (handwritten VMI PTO-DSL) size ladder — asymptotic α / el/cyc

**Date:** 2026-10-14 HKT · **SOC:** Ascend950PR_9599 opsim (pto-b10) · **VL=64**
**Env note:** must NOT source `~/projects/env.sh` for DSL camodel — it pollutes `PTOAS_BIN`/`LD_LIBRARY_PATH` and causes GetKernelAddr→zero-output FAIL. Use cann_91b3 + `.camodel_deps_vmi018` only.

## Results

| E | VF_cyc | wall µs | body_instr | IPC | cyc/vec | **el/cyc** |
|--:|-------:|--------:|-----------:|----:|--------:|-----------:|
| 256 | 60 | 0.89 | 21 | 0.3500 | 15.0 | **4.2667** |
| 4096 | 161 | 1.62 | 261 | 1.6211 | 2.5156 | **25.441** |
| 8192 | 280 | 2.59 | 517 | 1.8464 | 2.1875 | **29.2571** |
| 16384 | 520 | 4.55 | 1029 | 1.9788 | 2.0312 | **31.5077** |

## Slope α = ΔC / (ΔE/64)

- pairwise **4096→8192**: α=**1.8594** cyc/vec ⇒ asym el/cyc ≈ **34.42** (raw @8192: 29.26)
- pairwise **8192→16384**: α=**1.8750** cyc/vec ⇒ asym el/cyc ≈ **34.13** (raw @16384: 31.51)
- span **4096→16384**: α=**1.8698** ⇒ asym el/cyc ≈ **34.23**
- lstsq E≥4096: α=**1.8705** cyc/vec, C0=41.0 ⇒ asym el/cyc ≈ **34.21**

## Verdict vs theory α≈1 → 64 el/cyc

At largest measured **E=16384**: **31.5 el/cyc** (520 VF cyc). Large-E slope α≈**1.87** cyc/vec ⇒ asymptotic ≈**34 el/cyc**.

**~64 el/cyc is NOT approached** on this handwritten VMI PTO-DSL SV1 (full-resident A+B+C in UB). Gap is ~2× vs theory (α≈1.87 vs 1) — consistent with ~2 cyc/vec for the 2×vload+vadd+vstore chunk when LD/ST/EX do not fully collapse to 1 issue/cyc.

E=32768 skipped: 3·E·4 = 384 KiB exceeds ~192–256 KiB UB for full-resident; needs tiled GM streaming (not measured here).

Raw TSV: `data/SUMMARY_sv1d_size_ladder.tsv`.
