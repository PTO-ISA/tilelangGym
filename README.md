# tilelangGym

Standalone sensitivity / regression gym for **TileLang SimtVF + `T.Parallel`** on Ascend (and PTO / PTO-DSL twins).

Moved out of `PTO-ISA/tilelang` `examples/ascend/t_parallel_st_suite` so the suite can pin and track **TileLang** and **PTOAS** versions independently.

## Suite layout

| Path | Contents |
|------|----------|
| `kernels/` | SV1–SV9, CF1–CF6 (+ VMI `*v` twins), SP1–SP6 |
| `kernels_ptodsl/` | PTO-DSL `*d` twins (`@pto.jit`) |
| `oneshot_*.sh` | Family runners + `oneshot_vf_mode_regress.sh` (simt-asc / simt-pto / simd-pto) |
| `common_asc_harness.py` / `common_pto_harness.py` / `vf_mode.py` | Shared compile + opsim harness |
| `docs/regression/` | Pins, PASS/FAIL classification, probe notes |
| `reports/` | Design docs and historical PASS tables |
| `SUMMARY.md` | Suite history / renumber notes |

## Recommended testing pins (2026-10-14 HKT)

| Item | Pin |
|------|-----|
| **CANN** | `cann_92b2` **9.2.0-beta.2** (`…/cann_92b2/cann-9.2.0-beta.2`) |
| **SOC / camodel** | Ascend950PR_9599 |
| **PTOAS (simt-pto)** | nightly **0.67** (CLI reports `ptoas 0.67`; no `0.67.YYYYMMDD` suffix). **Datecode / build:** generated **20261009** UTC (`2026-10-09T23:22:18Z` → HKT 2026-10-10 ~07:22), `source_commit=`**`1b36b6f58e6a9cf4db6e76b98206411a8f9d1f80`**, GH Actions run [`37994539624`](https://github.com/hw-native-sys/PTOAS/actions/runs/37994539624), wheel `ptoas-0.67-cp312-cp312-manylinux_2_34_aarch64.whl` → deps dir `.camodel_deps_nightly067` |
| **TileLang (live remat tip)** | branch `fix/simtvf-remat-idx-evalset-vec` @ **`a3012088cc61a7fac36c50ae19caf2524d5bf804`** |
| **TileLang (full-table stamp)** | tip **`31483438`** (2026-10-10) — what the PASS tables below were measured on |
| **Suite history** | also lived on tilelang `feat/st-simtvf-parallel-sv1-sv9` @ `593a8f0c` (PR #2) |

> Do **not** pin Ascend `max_vector_bits=128` / float4 emit yet (rebuild/regress still in flight). Stock Simt-asc still caps at **float2 / 64-bit**.

Detail: [`docs/regression/PINS_AND_RESULTS_20261013.md`](docs/regression/PINS_AND_RESULTS_20261013.md).

## How to run

Host: Ascend lab box (e.g. pto-b10) with TileLang tip built, CANN `cann_92b2` sourced, and (for **simt-pto**) PTOAS 0.67 installed into `CAMODEL_DEPS` / `.camodel_deps_nightly067`.

```bash
git clone https://github.com/PTO-ISA/tilelangGym.git
cd tilelangGym

# Required env (example paths on pto-b10 / fluxdata lab layout)
export TILELANG_DEPS=/mnt/fluxdata/happybot/projects/tilelang-pto-isa-st   # tree with build/lib/libtilelang.so
export CAMODEL_DEPS=/mnt/fluxdata/happybot/projects/.camodel_deps_nightly067  # or .camodel_deps_vmi018 for simt-asc
export ASCEND_HOME_PATH=/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2
export PY=/path/to/.venv-npu/bin/python   # optional; defaults to python3
# optional opsim helper: RUN_CF_MB_OPSIM or $TILELANG_DEPS/examples/ascend/run_cf_mb_opsim.py

# Full VF-mode regress (primary)
export ST_VF_MODES=simt-asc          # or: simt-pto | simd-pto | simt-asc,simt-pto
bash oneshot_vf_mode_regress.sh

# Family oneshots (SimtSV / CF / SP / PTO-DSL)
bash oneshot_sv1_sv9.sh
bash oneshot_cf1_cf6.sh
bash oneshot_sp1_sp6.sh
bash oneshot_ptodsl_sv1_sv9d.sh
```

**Primary metric:** total VF cycles (plus wall µs / PASS line from opsim harvest). Tables land as `VF_MODE_PASS_TABLE.txt` under the OUT dir.

Laptop relays (optional): `run_via_laptop_*.ps1` SSH to the lab host.

Env notes:
- **simt-asc:** `target=ascend` + cython backend (not pto); deps-native `libtilelang.so`
- **simt-pto:** SoftLib EmitInlineSimtVF → PTOAS nightly 0.67
- **simd-pto:** `VectorizeParallelToPTO` / VL `vload` (not CUDA float4)
- Do not mix overlay and deps-native libs in one run

## Current regression snapshot (tip `31483438`, 2026-10-10)

| Mode | OUT (lab) | Result |
|------|-----------|--------|
| **simt-asc** | `st_out_cann92b2_20261010_104824` | **59 PASS / 5 FAIL** |
| **simt-pto** | `st_out_nightly067_simt_pto_20261010_170837` | **33 PASS / 29 FAIL** |
| **simd-pto** | `st_out_simd_pto_20261010_163238` | **0 PASS / 61 FAIL** |

Caveat: remat tip `a3012088` has **not** had a full-suite re-run yet. SV4 `e256_b8` remat_idx is cleared on that tip; bare sv6/sv7 arms and most simt-pto FAILs are still TileLang-side.

### simt-asc remaining FAILs (5 on full table)
1. `sv4_e256_b8_t32_remat_idx` — cleared on `a3012088` (e256_b8 only)
2. `sv4_e256_b16_t32_remat_idx` — not re-proven post-remat
3. bare `sv6_r64_c128_g32_t32` — harness false FAIL (kernel defaults `keep_in_warp`; that arm PASS)
4. bare `sv7_r64_c128_g64_t32` / `sv7_r64_c128_g128_t32` — leftover bare tags (multiwarp arms already PASS)

### simt-pto FAIL classification (29) — mainly TileLang→PTO codegen, **0 pure PTOAS**

| Bucket | Count | Notes |
|--------|------:|-------|
| Unsupported PTO cast f16→f32 (`codegen_pto.cc`) | 11 | sv2/sv3/sv5/sv7/sv8 |
| Illegal `llvm.store` on keep_idx (emitted PTO) | 8 | all `sv4_*_keep_idx`; sibling `remat_idx_calc` PASS → TileLang emit, not PTOAS |
| AscendThreadSync `eval_vec_` remat_idx | 2 | addressed on remat tip for e256_b8 |
| PTODSL `store()` arity | 3 | sv9 keep / remat_scores |
| IndentationError in emitted py | 2 | sp5 interleave / sideband |
| Vec init got `!pto.vreg` (e8m0) | 1 | sp3 e8m0 |
| Harness only | 2 | sv6 bare false FAIL; keep_in_warp PASS on remat tip probe |

Writeups: [`docs/regression/`](docs/regression/) (`ST_SUITE_simt_asc_pto_STATUS_20261013.md`, `ST_SUITE_simt_pto_FAIL_CLASSIFY_20261013.md`, `ST_SUITE_sv6_sv4_probe_20261013.md`).

## Design docs

- [`reports/ST_LIST_DESIGN.md`](reports/ST_LIST_DESIGN.md) — SV1–SV9 purpose
- [`reports/ST_CASE3_DESIGN.md`](reports/ST_CASE3_DESIGN.md) — Case-3 RF ladder
- [`reports/ST_SP1_TO_SP6_ACCESS.md`](reports/ST_SP1_TO_SP6_ACCESS.md) — SP access sensitivities
- [`reports/SIMT_LDST_WIDTH_20261008.md`](reports/SIMT_LDST_WIDTH_20261008.md) — elementwise ld/st width (float2 cap)

## License / upstream

Suite sources originated in [PTO-ISA/tilelang](https://github.com/PTO-ISA/tilelang) PR #2 / remat work. This repo is the standalone gym home going forward.
