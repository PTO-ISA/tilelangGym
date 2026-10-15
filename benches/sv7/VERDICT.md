# VERDICT — SV7

## Highlight

See per-case table below. Suite rollup: [`reports/PASS_FAIL_PERF.md`](../../reports/PASS_FAIL_PERF.md).

**Focus paths:** simt-asc (gate), simd-pto, handwritten SIMD DSL. **Deferred:** simt-pto SoftLib.

**Pins:** TileLang remat tip `a3012088` ([tilelang#5](https://github.com/PTO-ISA/tilelang/pull/5)); full-table stamp `31483438` (2026-10-10); CANN cann_92b2; PTOAS 0.67.

## Verdict

Per-case status:

## Per-case (simt-asc)

| Tag | Suite | simt-asc verdict | Notes |
|-----|-------|------------------|-------|
| `sv7_r64_c128_g64_t32` | gate | HARNESS_BARE (ignore; use multiwarp_ub) | harness leftover |
| `sv7_r64_c128_g128_t32` | gate | HARNESS_BARE (ignore; use multiwarp_ub) | harness leftover |
| `sv7_r64_c128_g64_t32_multiwarp_ub` | gate | PASS |  |
| `sv7_r64_c128_g128_t32_multiwarp_ub` | gate | PASS |  |

## Other paths

| Path | Status |
|------|--------|
| **simd-pto** | Active focus — Oct-10 VF-mode twin matrix still largely red (`0/61` suite-wide); per-bench remeasure TBD |
| **Handwritten SIMD DSL** (`*d`) | Active focus — use `kernel_dsl` when present |
| **simt-pto SoftLib** | **Deferred** — do not drive fixes from historical 29-FAIL list |

**Harness ignore:** `sv7_r64_c128_g64_t32`, `sv7_r64_c128_g128_t32`

See also: [`docs/regression/`](../../docs/regression/) and suite README.
