# VERDICT — SV6: Case-3 mid-G reduce eltwise

**Focus paths:** simt-asc (gate), simd-pto, handwritten SIMD DSL. **Deferred:** simt-pto SoftLib.

**Pins:** TileLang remat tip `a3012088` ([tilelang#5](https://github.com/PTO-ISA/tilelang/pull/5)); full-table stamp `31483438` (2026-10-10); CANN cann_92b2; PTOAS 0.67.

## Per-case (simt-asc)

| Tag | Suite | simt-asc verdict | Notes |
|-----|-------|------------------|-------|
| `sv6_r64_c128_g32_t32` | gate | HARNESS_FALSE_FAIL (ignore bare tag; use keep_in_warp) | harness false FAIL — kernel defaults keep_in_warp |
| `sv6_r64_c128_g32_t32_keep_in_warp` | gate | PASS |  |

## Other paths

| Path | Status |
|------|--------|
| **simd-pto** | Active focus — Oct-10 VF-mode twin matrix still largely red (`0/61` suite-wide); per-bench remeasure TBD |
| **Handwritten SIMD DSL** (`*d`) | Active focus — use `kernel_dsl` when present |
| **simt-pto SoftLib** | **Deferred** — do not drive fixes from historical 29-FAIL list |

**Harness ignore:** `sv6_r64_c128_g32_t32`

See also: [`docs/regression/`](../../docs/regression/) and suite README.
