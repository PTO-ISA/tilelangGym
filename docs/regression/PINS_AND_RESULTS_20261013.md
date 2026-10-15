# Testing pins and regression snapshot (2026-10-14 HKT)

Caveat: newest **full** oneshot tables are still tip `31483438` (2026-10-10). Remat tip `a3012088` ([tilelang#5](https://github.com/PTO-ISA/tilelang/pull/5)) was not full-suite re-run. Do **not** pin Ascend `max_vector_bits=128` yet (rebuild/regress in flight).

**Focus:** simt-asc, simd-pto, handwritten SIMD DSL. **Defer** simt-pto SoftLib lowering.

## Recommended pins

| Item | Pin |
|------|-----|
| **CANN** | `cann_92b2` **9.2.0-beta.2** |
| **SOC / camodel** | Ascend950PR_9599 |
| **PTOAS** | nightly **0.67** (no YYYYMMDD in package name). Build `generated_at_utc=2026-10-09T23:22:18Z` (HKT 2026-10-10 ~07:22), `source_commit=1b36b6f58e6a9cf4db6e76b98206411a8f9d1f80`, GH Actions run `37994539624`, wheel `ptoas-0.67-cp312-cp312-manylinux_2_34_aarch64.whl` → deps `.camodel_deps_nightly067` |
| **TileLang (live remat tip)** | branch `fix/simtvf-remat-idx-evalset-vec` @ **`a3012088cc61a7fac36c50ae19caf2524d5bf804`** |
| **TileLang (full-table stamp)** | tip **`31483438`** (2026-10-10) — raw PASS tables below |
| **Suite history (PR #2)** | `feat/st-simtvf-parallel-sv1-sv9` @ `593a8f0c` |

Note: simt-asc full OUT used `.camodel_deps_vmi018`; PTOAS 0.67 deps live under `.camodel_deps_nightly067`.

## How to run (standalone gym)

```bash
export TILELANG_DEPS=/mnt/fluxdata/happybot/projects/tilelang-pto-isa-st
export CAMODEL_DEPS=/mnt/fluxdata/happybot/projects/.camodel_deps_vmi018
export ASCEND_HOME_PATH=/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2

# Primary: simt-asc (and simd-pto when ready)
export ST_VF_MODES=simt-asc
bash oneshot_vf_mode_regress.sh

# Handwritten SIMD DSL
bash oneshot_ptodsl_sv1_sv9d.sh
```

Primary metric: total VF cycles (plus wall µs / PASS line from opsim harvest).

## Raw full oneshot tables (tip `31483438`)

| Mode | OUT | Raw result | How to read |
|------|-----|------------|-------------|
| simt-asc | `st_out_cann92b2_20261010_104824` | 59 PASS / 5 FAIL | **Misleading residual** — see below |
| simd-pto | `st_out_simd_pto_20261010_163238` | 0/61 | **Active focus** |
| simt-pto | `st_out_nightly067_simt_pto_20261010_170837` | 33 PASS / 29 FAIL | **Deferred** SoftLib path |

Tables live as `VF_MODE_PASS_TABLE.txt` under each OUT on the lab host.

### simt-asc residual (5 on raw table) → ~all PASS after harness + remat

| Case | Class | Status |
|------|-------|--------|
| `sv4_e256_b8_t32_remat_idx` | remat / indirect Idx vector gather | **PASS** on `a3012088` (#5) |
| `sv4_e256_b16_t32_remat_idx` | same as b8 | **Pending re-check** on #5 @ `a3012088` |
| bare `sv6_r64_c128_g32_t32` | harness false FAIL | real arm `keep_in_warp` PASS |
| bare `sv7_r64_c128_g64_t32` | harness bare leftover | real arm `multiwarp_ub` PASS |
| bare `sv7_r64_c128_g128_t32` | harness bare leftover | same |

### simt-pto SoftLib — deferred

Historical classification (for later): mostly TileLang→PTO codegen, 0 pure PTOAS. See `ST_SUITE_simt_pto_FAIL_CLASSIFY_20261013.md`. Not a current gym priority.
