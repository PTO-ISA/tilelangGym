# VERDICT — SV1: Stream Parallel eltwise baseline

**Focus paths:** simt-asc (gate), simd-pto, handwritten SIMD DSL. **Deferred:** simt-pto SoftLib.

**Pins:** TileLang remat tip `a3012088` ([tilelang#5](https://github.com/PTO-ISA/tilelang/pull/5)); full-table stamp `31483438` (2026-10-10); CANN cann_92b2; PTOAS 0.67.

**Full exploration + plots:** see [`ANALYSIS.md`](ANALYSIS.md) (tile vs VF/el/instr, thread vs IPC/el, large-tile instr breakdown, LDS/STS.64 rooftop **16**, ILP@T128, bubble/T, GPU compare).

## Per-case (simt-asc)

| Tag | Suite | simt-asc verdict | Notes |
|-----|-------|------------------|-------|
| `sv1_e256_t32` | gate | PASS |  |
| `sv1_e2048_t32` | gate | PASS |  |
| `sv1_e{512…32768}_t32` | explore | EXPLORE | size ladder |
| `sv1_e8192_t{64…512}` | explore | EXPLORE | thread sweep @ 8K |
| `sv1t_e16384_t{32…512}` | explore-measured | MEASURED PASS | best stock el/cyc **3.71 @ T512** |
| `sv1tt_e16384_tile*_t128` | explore-measured | MEASURED PASS | best tiled **TILE2048: 5088 cyc, 3.22 el/cyc**; pack **.64 only** |
| `sv1tt_e16384_tile{256,1024,8192}_t32` | explore-measured | MEASURED PASS | T32 foil |

## Handwritten SIMD DSL / simd-pto

| Path | Status |
|------|--------|
| **Handwritten SIMD DSL** (`sv1d_e*_vl64`) | Active — size ladder listed in `config.toml`; fold VF/α when remeasured |
| **simd-pto** | Active focus — suite-wide Oct-10 still largely red; per-bench remeasure TBD |
| **simt-pto SoftLib** | **Deferred** |

## Rooftop (Lok-corrected)

- **SimtVF roof = 16 el/cyc** (UB **1R+1W @ 128B each**). Pack stays **LDS/STS.64 only** (no `.128` on Ascend Simt `T.Parallel`).
- Stock Simt SV1 tops out ~**3.2–3.7 el/cyc** (~**20–23% of roof 16**); Body4 mid-median IPC ~**0.47** ≪ mix-cap **4**.
- **ILP16 @ fixed T=128** (CCE rewrite, `.64` only): **4.08 el/cyc (25.5% of 16)** — beats untiled T=2048 (**3.83**). Mid sustained ~**4.5**; gap to 16 is mid-body idle/IADD, not launch/head–tail.
- SIMD/SV1d (separate roof **42.67** @ 2R+1W): body ~**34 el/cyc (~80%)**.

## Bubble / T findings (untiled E=16K)

- T=128 issue bubbles are **scoreboard RAW** (FADD `waitBitMask≠0` **50%**): IADD→LDS.64→FADD→STS.64.
- T↑ fills holes with other warps: Body4 IPC **0.376→0.468** (T128→T512); max idle bubble **244→32** (mostly **drain** outlier — mid-body max idle only **8**). Mid-median stays ~**0.47** even at T=2048; mid-peak rises (0.50→1.50) but is not sustained.
- Lever: **schedule at fixed T=128**, not pile warps.

## GPU compare (same E=16K T=128, order-of-magnitude)

| Case | Pack | el/cyc |
|------|------|------:|
| Ascend ILP16 @ T128 | .64 | **4.08** |
| GPU TileLang TILE2048@T128 | float4 CUDA | ~1.04 |
| GPU hand ILP4 1×CTA .128 | .128 | ~0.88 |
| GPU hand unroll multi-CTA .128 | .128 | ~2.15 |

Fairest 1×T=128: Ascend looks higher on camodel VF vs ncu GB10 — different roofs/ISAs; not a fair shootout. GPU SASS does `.128` + ILP reschedule; Ascend Simt path still `.64`.

## Weekly / CI regression (stock SimtVF)

- **Script:** [`oneshot_sv1_e8192_t128_t512.sh`](../../oneshot_sv1_e8192_t128_t512.sh) — **SV1 only**, **E=8192**, **T=128** and **T=512**, `simt-asc`.
- **Docs:** [`docs/regression/REGRESS_SV1_E8192_T128_T512.md`](../../docs/regression/REGRESS_SV1_E8192_T128_T512.md).
- Expect **PASS** without schedule rewrites / AI fixes. Exit 0 iff both PASS.

## Todos — T=128 instruction scheduling (next)

- [ ] Deeper software ILP at **fixed T=128** (ILP32/64), keep **LDS/STS.64**
- [ ] Cut **IADD** tax (imm / float2 base indexing); interleave / soft-pipe schedules
- [ ] Raise **mid-median** Body4 IPC / sustained el/cyc toward roof **16** (today mid ~4.5 / VF 4.08)
- [ ] Verify **I-cache**: aicore `.text` ~**6 KB** @ ILP16 (comfortable vs typical 16–32 KB I$); re-check deeper unroll
- [ ] Verify **MRF 128 KB** budget for E=8K (A+B+C=96 KB + live regs); ILP16 theory ~48 KB temps @ T128
- [ ] Optional later: emit **`.128`** (separate from schedule)

## Artifacts

- `ANALYSIS.md`, `plots/*.png`, `data/SUMMARY_*.tsv`, `data/MIX_*.txt`, `data/IADD_IPC_REPORT_tile16k.txt`
- `analysis/NOTE_sv1_rooftop_corrected.md`, `analysis/NOTE_sv1_simt_ilp64_T128.md`, `analysis/NOTE_sv1_simt_bubbles_T128_vs_T512.md`, `analysis/NOTE_sv1_gpu_vs_ascend_ilp16_elcyc.md`
