# VERDICT — SV4

## Highlight

See per-case table below. Suite rollup: [`reports/PASS_FAIL_PERF.md`](../../reports/PASS_FAIL_PERF.md).

**Focus paths:** simt-asc (gate), simd-pto, handwritten SIMD DSL. **Deferred:** simt-pto SoftLib.

**Pins:** TileLang remat tip `a3012088` ([tilelang#5](https://github.com/PTO-ISA/tilelang/pull/5)); full-table stamp `31483438` (2026-10-10); CANN cann_92b2; PTOAS 0.67.

## Verdict

Per-case status:

## Per-case (simt-asc)

| Tag | Suite | simt-asc verdict | Notes |
|-----|-------|------------------|-------|
| `sv4_e256_b8_t32_keep_idx` | gate | PASS (Oct-10 / remat tip; treat gate green after harness ignore) |  |
| `sv4_e256_b8_t32_remat_idx` | gate | PASS (cleared on tilelang#5 @ a3012088) |  |
| `sv4_e256_b8_t32_remat_idx_calc` | gate | PASS (Oct-10 / remat tip; treat gate green after harness ignore) |  |
| `sv4_e256_b16_t32_keep_idx` | gate | PASS (Oct-10 / remat tip; treat gate green after harness ignore) |  |
| `sv4_e256_b16_t32_remat_idx` | gate | PENDING re-check on tilelang#5 @ a3012088 (same fix path as b8) |  |
| `sv4_e256_b16_t32_remat_idx_calc` | gate | PASS (Oct-10 / remat tip; treat gate green after harness ignore) |  |
| `sv4_e512_b8_t32_keep_idx` | gate | PASS (Oct-10 / remat tip; treat gate green after harness ignore) |  |
| `sv4_e512_b8_t32_remat_idx_calc` | gate | PASS (Oct-10 / remat tip; treat gate green after harness ignore) |  |
| `sv4_e1024_b8_t32_keep_idx` | gate | PASS (Oct-10 / remat tip; treat gate green after harness ignore) |  |
| `sv4_e1024_b8_t32_remat_idx_calc` | gate | PASS (Oct-10 / remat tip; treat gate green after harness ignore) |  |
| `sv4_e1024_b16_t32_keep_idx` | gate | PASS (Oct-10 / remat tip; treat gate green after harness ignore) |  |
| `sv4_e1024_b16_t32_remat_idx_calc` | gate | PASS (Oct-10 / remat tip; treat gate green after harness ignore) |  |
| `sv4_e1536_b8_t32_keep_idx` | gate | PASS (Oct-10 / remat tip; treat gate green after harness ignore) |  |
| `sv4_e1536_b8_t32_remat_idx_calc` | gate | PASS (Oct-10 / remat tip; treat gate green after harness ignore) |  |
| `sv4_e2048_b8_t32_keep_idx` | gate | PASS (Oct-10 / remat tip; treat gate green after harness ignore) |  |
| `sv4_e2048_b8_t32_remat_idx_calc` | gate | PASS (Oct-10 / remat tip; treat gate green after harness ignore) |  |
| `sv4_e1024_b8_t16_keep_idx` | gate | PASS (Oct-10 / remat tip; treat gate green after harness ignore) |  |
| `sv4_e1024_b8_t16_remat_idx_calc` | gate | PASS (Oct-10 / remat tip; treat gate green after harness ignore) |  |

## Other paths

| Path | Status |
|------|--------|
| **simd-pto** | Active focus — Oct-10 VF-mode twin matrix still largely red (`0/61` suite-wide); per-bench remeasure TBD |
| **Handwritten SIMD DSL** (`*d`) | Active focus — use `kernel_dsl` when present |
| **simt-pto SoftLib** | **Deferred** — do not drive fixes from historical 29-FAIL list |

**Pending:** `sv4_e256_b16_t32_remat_idx`

### SV4 remat note

- `sv4_e256_b8_t32_remat_idx` **PASS** on tip `a3012088`.
- `sv4_e256_b16_t32_remat_idx` same TileLang indirect-addr / EvalSet fix — **re-check pending**.

See also: [`docs/regression/`](../../docs/regression/) and suite README.
