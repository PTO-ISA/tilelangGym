# Testing pins and regression snapshot (2026-10-14 HKT)

Caveat: newest **full** PASS tables are still tip `31483438` (2026-10-10). Remat tip `a3012088` was not full-re-run. Do **not** pin Ascend `max_vector_bits=128` yet (rebuild/regress in flight).

## Recommended pins

| Item | Pin |
|------|-----|
| **CANN** | `cann_92b2` **9.2.0-beta.2** |
| **SOC / camodel** | Ascend950PR_9599 |
| **PTOAS (simt-pto)** | nightly **0.67** (no YYYYMMDD in package name). Build `generated_at_utc=2026-10-09T23:22:18Z` (HKT 2026-10-10 ~07:22), `source_commit=1b36b6f58e6a9cf4db6e76b98206411a8f9d1f80`, GH Actions run `37994539624`, wheel `ptoas-0.67-cp312-cp312-manylinux_2_34_aarch64.whl` → deps `.camodel_deps_nightly067` |
| **TileLang (live remat tip)** | branch `fix/simtvf-remat-idx-evalset-vec` @ **`a3012088cc61a7fac36c50ae19caf2524d5bf804`** |
| **TileLang (full-table stamp)** | tip **`31483438`** (2026-10-10) — PASS tables below |
| **Suite history (PR #2)** | `feat/st-simtvf-parallel-sv1-sv9` @ `593a8f0c` |

Note: simt-asc full OUT used `.camodel_deps_vmi018`; simt-pto used `.camodel_deps_nightly067`.

## How to run (standalone gym)

```bash
# After installing TileLang tip + CANN + (for simt-pto) PTOAS 0.67 into CAMODEL_DEPS
export ST_VF_MODES=simt-asc   # or simt-pto
bash oneshot_sv1_sv9.sh
# CF / SP families: see oneshot_*.sh in repo root
```

Primary metric: total VF cycles (plus wall µs / PASS line from opsim harvest).

## Latest full PASS tables (tip `31483438`)

| Mode | OUT | Result |
|------|-----|--------|
| simt-asc | `st_out_cann92b2_20261010_104824` | **59 PASS / 5 FAIL** |
| simt-pto | `st_out_nightly067_simt_pto_20261010_170837` | **33 PASS / 29 FAIL** |
| simd-pto | `st_out_simd_pto_20261010_163238` | **0/61** |

Tables live as `VF_MODE_PASS_TABLE.txt` under each OUT on the lab host.

### simt-asc remaining FAILs (5 on full table)
`sv4_e256_b8_t32_remat_idx` (cleared on `a3012088` for e256_b8), `sv4_e256_b16_t32_remat_idx` (not re-proven), bare `sv6_r64_c128_g32_t32`, `sv7_r64_c128_g64_t32`, `sv7_r64_c128_g128_t32` (arm tags keep_in_warp/multiwarp already PASS).

### simt-pto
29 non-PASS — mostly TileLang→PTO codegen (27), 2 harness. See sibling docs in this folder.

