# tilelangGym

Standalone sensitivity / regression gym for **TileLang SimtVF + `T.Parallel`** on Ascend, plus **SIMD Parallel → PTO** and **handwritten SIMD / PTO-DSL** twins.

Moved out of `PTO-ISA/tilelang` `examples/ascend/t_parallel_st_suite` so the suite can pin and track **TileLang** and **PTOAS** versions independently.

## Focus (current)

| Priority | Path | Notes |
|----------|------|-------|
| **Primary** | **simt-asc** | Ascend SoftLib SimtVF (`target=ascend`, cython). This is the main regression gate. |
| **Primary** | **simd-pto** | `VectorizeParallelToPTO` / VL `vload`/`vstore` (not CUDA float4). |
| **Primary** | **Handwritten SIMD DSL** | PTO-DSL `*d` twins under `kernels_ptodsl/` (`@pto.jit`). |
| **Deferred** | **simt-pto** SoftLib lowering | Kernels / `ST_VF_MODES=simt-pto` remain in-tree for later; **do not** treat the historical 29 FAIL table as the current work queue. |

Per-case shapes and verdicts: `benches/<id>/config.toml` + `benches/<id>/VERDICT.md`. SV1 explore analysis: [`benches/sv1/ANALYSIS.md`](benches/sv1/ANALYSIS.md).

## Suite layout

| Path | Contents |
|------|----------|
| **`benches/<id>/`** | **Per-benchmark home:** kernel(s) + `config.toml` (shapes/arms) + `VERDICT.md` (SV1…SV9, CF*, SP*) |
| `kernels/` / `kernels_ptodsl/` | Symlinks into `benches/` for legacy `oneshot_*.sh` compatibility |
| `oneshot_*.sh` | Family runners + `oneshot_vf_mode_regress.sh` |
| `common_asc_harness.py` / `common_pto_harness.py` / `vf_mode.py` | Shared compile + opsim harness |
| `docs/regression/` | Pins, PASS/FAIL notes, probes |
| `reports/` | Design docs and historical PASS tables |
| `SUMMARY.md` | Suite history / renumber notes |

See [`benches/README.md`](benches/README.md). Each bench `config.toml` lists gate cases (oneshot) and, where useful (e.g. SV1), `suite = "explore"` size/thread ladders.

## Recommended testing pins (2026-10-14 HKT)

| Item | Pin |
|------|-----|
| **CANN** | `cann_92b2` **9.2.0-beta.2** (`…/cann_92b2/cann-9.2.0-beta.2`) |
| **SOC / camodel** | Ascend950PR_9599 |
| **PTOAS** | nightly **0.67** (CLI `ptoas 0.67`; no `0.67.YYYYMMDD` suffix). Build datecode **20261009** UTC (`2026-10-09T23:22:18Z` → HKT 2026-10-10 ~07:22), `source_commit=`**`1b36b6f58e6a9cf4db6e76b98206411a8f9d1f80`**, GH Actions run [`37994539624`](https://github.com/hw-native-sys/PTOAS/actions/runs/37994539624), wheel `ptoas-0.67-cp312-cp312-manylinux_2_34_aarch64.whl` → deps `.camodel_deps_nightly067` |
| **TileLang (live remat tip)** | branch `fix/simtvf-remat-idx-evalset-vec` @ **`a3012088cc61a7fac36c50ae19caf2524d5bf804`** — [tilelang#5](https://github.com/PTO-ISA/tilelang/pull/5) |
| **TileLang (full-table stamp)** | tip **`31483438`** (2026-10-10) — raw oneshot tables below were measured on this tip |
| **Suite history** | also lived on tilelang `feat/st-simtvf-parallel-sv1-sv9` @ `593a8f0c` (PR #2) |

> Do **not** pin Ascend `max_vector_bits=128` / float4 emit yet (rebuild/regress still in flight). Stock Simt-asc still caps at **float2 / 64-bit**.

Detail: [`docs/regression/PINS_AND_RESULTS_20261013.md`](docs/regression/PINS_AND_RESULTS_20261013.md).

## How to run

### Weekly SV1 gate (E=8192, T=128 + T=512)

Single regression script — stock **simt-asc**, no schedule rewrites. Expect both cases **PASS**:

```bash
# after setting TILELANG_DEPS / CAMODEL_DEPS / ASCEND_HOME_PATH / PY / SIM_DSL (see below)
bash oneshot_sv1_e8192_t128_t512.sh
```

Details: [`docs/regression/REGRESS_SV1_E8192_T128_T512.md`](docs/regression/REGRESS_SV1_E8192_T128_T512.md).

## How to run (full suite)


Host: Ascend lab box (e.g. pto-b10) with TileLang tip built and CANN `cann_92b2` sourced. For **simd-pto** / PTO-DSL, also install PTOAS 0.67 into `CAMODEL_DEPS` (`.camodel_deps_nightly067` or lab equivalent).

```bash
git clone https://github.com/PTO-ISA/tilelangGym.git
cd tilelangGym

# Required env (example paths on pto-b10 / fluxdata lab layout)
export TILELANG_DEPS=/mnt/fluxdata/happybot/projects/tilelang-pto-isa-st   # tree with build/lib/libtilelang.so
export CAMODEL_DEPS=/mnt/fluxdata/happybot/projects/.camodel_deps_vmi018   # simt-asc; use .camodel_deps_nightly067 for PTOAS 0.67
export ASCEND_HOME_PATH=/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2
export PY=/path/to/.venv-npu/bin/python   # optional; defaults to python3

# Primary regress paths
export ST_VF_MODES=simt-asc               # Ascend SoftLib SimtVF (gate)
# export ST_VF_MODES=simd-pto             # Parallel → PTO / VL
bash oneshot_vf_mode_regress.sh

# Family oneshots (Simt SV / CF / SP) + handwritten SIMD DSL
bash oneshot_sv1_sv9.sh
bash oneshot_cf1_cf6.sh
bash oneshot_sp1_sp6.sh
bash oneshot_ptodsl_sv1_sv9d.sh
```

**Primary metric:** total VF cycles (plus wall µs / PASS line from opsim harvest). Tables land as `VF_MODE_PASS_TABLE.txt` under the OUT dir.

Laptop relays (optional): `run_via_laptop_*.ps1` SSH to the lab host.

Env notes:
- **simt-asc:** `target=ascend` + cython backend (not pto); deps-native `libtilelang.so`
- **simd-pto:** `VectorizeParallelToPTO` / VL `vload` (not CUDA float4)
- **Handwritten SIMD DSL:** `kernels_ptodsl/` `@pto.jit` twins
- **simt-pto SoftLib:** deferred — still runnable via `ST_VF_MODES=simt-pto`, not a current fix focus
- Do not mix overlay and deps-native libs in one run

## Current regression snapshot

### simt-asc — treat as ~all PASS (Oct-10 `59P/5F` is misleading)

Raw oneshot on tip **`31483438`** reported **59 PASS / 5 FAIL** (`st_out_cann92b2_20261010_104824`). That residual set is **not** five live product bugs:

| Case | Class | Status |
|------|-------|--------|
| `sv4_e256_b8_t32_remat_idx` | TileLang remat / indirect Idx vector gather | **PASS** on remat tip `a3012088` ([tilelang#5](https://github.com/PTO-ISA/tilelang/pull/5)) |
| `sv4_e256_b16_t32_remat_idx` | Same bug class as b8 | **Pending re-check** on #5 @ `a3012088` (same fix path; not re-opsim’d yet) |
| bare `sv6_r64_c128_g32_t32` | Harness false FAIL | Kernel defaults `arm=keep_in_warp` → builds `…_keep_in_warp.so`; that arm **PASS** |
| bare `sv7_r64_c128_g64_t32` | Harness bare leftover | Real arm `…_multiwarp_ub` already **PASS** |
| bare `sv7_r64_c128_g128_t32` | Harness bare leftover | Same |

**After** ignoring bare harness tags + remat PR #5, simt-asc should be **all PASS** except confirming `sv4_e256_b16_t32_remat_idx` on `a3012088`.

### simd-pto / handwritten SIMD DSL

| Mode | OUT (lab, tip `31483438`) | Snapshot |
|------|---------------------------|----------|
| **simd-pto** | `st_out_simd_pto_20261010_163238` | **0 PASS / 61** — active focus area |
| **PTO-DSL `*d`** | family oneshots / `kernels_ptodsl/` | Handwritten SIMD DSL twins — active focus |

### simt-pto SoftLib — deferred (do not prioritize)

Historical table on tip `31483438`: **33 PASS / 29 FAIL** (`st_out_nightly067_simt_pto_20261010_170837`). Failures were mainly TileLang→PTO SoftLib codegen (**0 pure PTOAS**). **Ignore / defer** this path for now; keep kernels in-tree but do not drive gym work off the 29-FAIL list. Optional detail: [`docs/regression/ST_SUITE_simt_pto_FAIL_CLASSIFY_20261013.md`](docs/regression/ST_SUITE_simt_pto_FAIL_CLASSIFY_20261013.md).

Writeups: [`docs/regression/`](docs/regression/).

## Design docs

- [`reports/ST_LIST_DESIGN.md`](reports/ST_LIST_DESIGN.md) — SV1–SV9 purpose
- [`reports/ST_CASE3_DESIGN.md`](reports/ST_CASE3_DESIGN.md) — Case-3 RF ladder
- [`reports/ST_SP1_TO_SP6_ACCESS.md`](reports/ST_SP1_TO_SP6_ACCESS.md) — SP access sensitivities
- [`reports/SIMT_LDST_WIDTH_20261008.md`](reports/SIMT_LDST_WIDTH_20261008.md) — elementwise ld/st width (float2 cap)

## License / upstream

Suite sources originated in [PTO-ISA/tilelang](https://github.com/PTO-ISA/tilelang) PR #2 / remat work ([tilelang#5](https://github.com/PTO-ISA/tilelang/pull/5)). This repo is the standalone gym home going forward.
