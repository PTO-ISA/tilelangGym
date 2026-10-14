# VERDICT — SV1: Stream Parallel eltwise baseline

**Focus paths:** simt-asc (gate), simd-pto, handwritten SIMD DSL. **Deferred:** simt-pto SoftLib.

**Pins:** TileLang remat tip `a3012088` ([tilelang#5](https://github.com/PTO-ISA/tilelang/pull/5)); full-table stamp `31483438` (2026-10-10); CANN cann_92b2; PTOAS 0.67.

## Per-case (simt-asc)

| Tag | Suite | simt-asc verdict | Notes |
|-----|-------|------------------|-------|
| `sv1_e256_t32` | gate | PASS (Oct-10 / remat tip; treat gate green after harness ignore) |  |
| `sv1_e2048_t32` | gate | PASS (Oct-10 / remat tip; treat gate green after harness ignore) |  |
| `sv1_e512_t32` | explore | EXPLORE (not in Oct-10 gate table) | SV1-style ladder / optional |
| `sv1_e1024_t32` | explore | EXPLORE (not in Oct-10 gate table) | SV1-style ladder / optional |
| `sv1_e4096_t32` | explore | EXPLORE (not in Oct-10 gate table) | SV1-style ladder / optional |
| `sv1_e8192_t32` | explore | EXPLORE (not in Oct-10 gate table) | SV1-style ladder / optional |
| `sv1_e16384_t32` | explore | EXPLORE (not in Oct-10 gate table) | SV1-style ladder / optional |
| `sv1_e32768_t32` | explore | EXPLORE (not in Oct-10 gate table) | SV1-style ladder / optional |
| `sv1_e8192_t64` | explore | EXPLORE (not in Oct-10 gate table) | SV1-style ladder / optional |
| `sv1_e8192_t128` | explore | EXPLORE (not in Oct-10 gate table) | SV1-style ladder / optional |
| `sv1_e8192_t256` | explore | EXPLORE (not in Oct-10 gate table) | SV1-style ladder / optional |
| `sv1_e8192_t512` | explore | EXPLORE (not in Oct-10 gate table) | SV1-style ladder / optional |

## Other paths

| Path | Status |
|------|--------|
| **simd-pto** | Active focus — Oct-10 VF-mode twin matrix still largely red (`0/61` suite-wide); per-bench remeasure TBD |
| **Handwritten SIMD DSL** (`*d`) | Active focus — use `kernel_dsl` when present |
| **simt-pto SoftLib** | **Deferred** — do not drive fixes from historical 29-FAIL list |

### SV1 exploration

`suite = "explore"` cases mirror the SV1 cost / size / thread sweep (E up to 32k, T sweep at large E).
Gate oneshot still runs E∈{256,2048}×T=32 only.

See also: [`docs/regression/`](../../docs/regression/) and suite README.
