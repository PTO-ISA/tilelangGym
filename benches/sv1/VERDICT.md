# VERDICT — SV1: Stream Parallel eltwise baseline

**Focus paths:** simt-asc (gate), simd-pto, handwritten SIMD DSL. **Deferred:** simt-pto SoftLib.

**Pins:** TileLang remat tip `a3012088` ([tilelang#5](https://github.com/PTO-ISA/tilelang/pull/5)); full-table stamp `31483438` (2026-10-10); CANN cann_92b2; PTOAS 0.67.

**Full exploration + plots:** see [`ANALYSIS.md`](ANALYSIS.md) (tile vs VF/el/instr, thread vs IPC/el, large-tile instr breakdown, LDS/STS.64/.128 rooftop).

## Per-case (simt-asc)

| Tag | Suite | simt-asc verdict | Notes |
|-----|-------|------------------|-------|
| `sv1_e256_t32` | gate | PASS |  |
| `sv1_e2048_t32` | gate | PASS |  |
| `sv1_e{512…32768}_t32` | explore | EXPLORE | size ladder |
| `sv1_e8192_t{64…512}` | explore | EXPLORE | thread sweep @ 8K |
| `sv1t_e16384_t{32…512}` | explore-measured | MEASURED PASS | best el/cyc **3.71 @ T512** |
| `sv1tt_e16384_tile*_t128` | explore-measured | MEASURED PASS | best tiled **TILE2048: 5088 cyc, 3.22 el/cyc**; pack **.64 only** |
| `sv1tt_e16384_tile{256,1024,8192}_t32` | explore-measured | MEASURED PASS | T32 foil |

## Handwritten SIMD DSL / simd-pto

| Path | Status |
|------|--------|
| **Handwritten SIMD DSL** (`sv1d_e*_vl64`) | Active — size ladder listed in `config.toml`; fold VF/α when remeasured |
| **simd-pto** | Active focus — suite-wide Oct-10 still largely red; per-bench remeasure TBD |
| **simt-pto SoftLib** | **Deferred** |

## Rooftop (one-liner)

Stock Simt SV1 tops out ~**3.2–3.7 el/cyc** with **LDS/STS.64**; **32 el/cyc FADD roof not reached** (IPC ~0.4 vs ~4 needed). **LDS/STS.128 never emitted** on Ascend Simt `T.Parallel`.

## Artifacts

- `ANALYSIS.md`, `plots/*.png`, `data/SUMMARY_sv1_simt_tile16k.tsv`, `data/MIX_*.txt`, `data/IADD_IPC_REPORT_tile16k.txt`
