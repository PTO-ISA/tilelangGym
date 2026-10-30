# VERDICT — SV8

## Highlight

See per-case table below. Suite rollup: [`reports/PASS_FAIL_PERF.md`](../../reports/PASS_FAIL_PERF.md).

**Focus paths:** simt-asc (gate), simd-pto, handwritten SIMD DSL. **Deferred:** simt-pto SoftLib.

**Pins:** TileLang remat tip `a3012088` ([tilelang#5](https://github.com/PTO-ISA/tilelang/pull/5)); full-table stamp `31483438` (2026-10-10); CANN cann_92b2; PTOAS 0.67.

## Verdict

Per-case status:

## Per-case (simt-asc)

| Tag | Suite | simt-asc verdict | Notes |
|-----|-------|------------------|-------|
| `sv8_r64_c128_g16_t32_live` | gate | PASS (Oct-10 / remat tip; treat gate green after harness ignore) |  |
| `sv8_r64_c128_g16_t32_spill_dist` | gate | PASS (Oct-10 / remat tip; treat gate green after harness ignore) |  |

## Other paths

| Path | Status |
|------|--------|
| **simd-pto** | Active focus — Oct-10 VF-mode twin matrix still largely red (`0/61` suite-wide); per-bench remeasure TBD |
| **Handwritten SIMD DSL** (`*d`) | Active focus — use `kernel_dsl` when present |
| **simt-pto SoftLib** | **Measured (t740u)** — Case-3 e2e G32 dual_drain **6.7702** vs VL-res SIMD **4.1711** (SoftLib÷SIMD≈1.62×); H2 ~2× SIMD falsified vs SoftLib |

See also: [`docs/regression/`](../../docs/regression/) and suite README.
