# tilelangGym

Standalone sensitivity / regression gym for **TileLang SimtVF + `T.Parallel`** on Ascend, plus **SIMD Parallel → PTO** and **handwritten SIMD / PTO-DSL** twins.

Moved out of `PTO-ISA/tilelang` `examples/ascend/t_parallel_st_suite` so the suite can pin and track **TileLang** and **PTOAS** versions independently.

## Focus (current)

| Priority | Path | Notes |
|----------|------|-------|
| **Primary** | **simt-asc** | Ascend SoftLib SimtVF (`target=ascend`, cython). Main regression gate — **all PASS** on remat tip below. |
| **Primary** | **simd-pto** | `VectorizeParallelToPTO` / VL `vload`/`vstore` (not CUDA float4). |
| **Primary** | **Handwritten SIMD DSL** | PTO-DSL `*d` twins under `kernels_ptodsl/` (`@pto.jit`). |

Per-case shapes and verdicts: `benches/<id>/config.toml` + `benches/<id>/VERDICT.md`. SV1 explore analysis: [`benches/sv1/ANALYSIS.md`](benches/sv1/ANALYSIS.md).

## Suite layout

| Path | Contents |
|------|----------|
| **`benches/<id>/`** | **Per-benchmark home:** kernel(s) + `config.toml` (shapes/arms) + `VERDICT.md` (SV1…SV9, CF*, SP*) |
| `kernels/` / `kernels_ptodsl/` | Symlinks into `benches/` for legacy oneshot compatibility |
| `scripts/oneshot/` | Family oneshot runners |
| `scripts/regression/` | Weekly / CI gates (SV1 E=8192 T128+T512) |
| `scripts/laptop/` | Optional `run_via_laptop_*.ps1` relays |
| `scripts/tools/` | harvest / opsim helpers |
| `oneshot_sv1_e8192_t128_t512.sh` | **Compat wrapper** → `scripts/regression/oneshot_sv1_e8192_t128_t512.sh` |
| `common_asc_harness.py` / `common_pto_harness.py` / `vf_mode.py` | Shared compile + opsim harness |
| `docs/regression/` | Pins, PASS notes, probes |
| `reports/` | Design docs and PASS tables |

See [`benches/README.md`](benches/README.md). Each bench `config.toml` lists gate cases (oneshot) and, where useful (e.g. SV1), `suite = "explore"` size/thread ladders.

## Recommended testing pins (2026-10-14 HKT)

| Item | Pin |
|------|-----|
| **CANN** | `cann_92b2` **9.2.0-beta.2** (`…/cann_92b2/cann-9.2.0-beta.2`) |
| **SOC / camodel** | Ascend950PR_9599 |
| **PTOAS** | nightly **0.67** (CLI `ptoas 0.67`; no `0.67.YYYYMMDD` suffix). Build datecode **20261009** UTC (`2026-10-09T23:22:18Z` → HKT 2026-10-10 ~07:22), `source_commit=`**`1b36b6f58e6a9cf4db6e76b98206411a8f9d1f80`**, GH Actions run [`37994539624`](https://github.com/hw-native-sys/PTOAS/actions/runs/37994539624), wheel `ptoas-0.67-cp312-cp312-manylinux_2_34_aarch64.whl` → deps `.camodel_deps_nightly067` |
| **TileLang (simt-asc all-PASS)** | branch `fix/simtvf-remat-idx-evalset-vec` @ **`a3012088cc61a7fac36c50ae19caf2524d5bf804`** — needs remat / **pointer-vectorize** patch [tilelang#5](https://github.com/PTO-ISA/tilelang/pull/5) |
| **TileLang (historical stamp)** | tip **`31483438`** (2026-10-10) — older raw tables only; **not** live simt-asc status |
| **Suite history** | also lived on tilelang `feat/st-simtvf-parallel-sv1-sv9` @ `593a8f0c` (PR #2) |

> Do **not** pin Ascend `max_vector_bits=128` / float4 emit yet (rebuild/regress still in flight). Stock Simt-asc still caps at **float2 / 64-bit**.

Detail: [`docs/regression/PINS_AND_RESULTS_20261013.md`](docs/regression/PINS_AND_RESULTS_20261013.md).

## How to run

### Weekly SV1 gate (E=8192, T=128 + T=512)

Single regression script — stock **simt-asc**, no schedule rewrites. Expect both cases **PASS**:

```bash
# after setting TILELANG_DEPS / CAMODEL_DEPS / ASCEND_HOME_PATH / PY / SIM_DSL (see below)
bash oneshot_sv1_e8192_t128_t512.sh   # or: bash scripts/regression/oneshot_sv1_e8192_t128_t512.sh
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
bash scripts/oneshot/oneshot_vf_mode_regress.sh

# Family oneshots (Simt SV / CF / SP) + handwritten SIMD DSL
bash scripts/oneshot/oneshot_sv1_sv9.sh
bash scripts/oneshot/oneshot_cf1_cf6.sh
bash scripts/oneshot/oneshot_sp1_sp6.sh
bash scripts/oneshot/oneshot_ptodsl_sv1_sv9d.sh
```

**Primary metric:** total VF cycles (plus wall µs / PASS line from opsim harvest). Tables land as `VF_MODE_PASS_TABLE.txt` under the OUT dir.

Laptop relays (optional): `scripts/laptop/run_via_laptop_*.ps1` SSH to the lab host.

Env notes:
- **simt-asc:** `target=ascend` + cython backend (not pto); deps-native `libtilelang.so`
- **simd-pto:** `VectorizeParallelToPTO` / VL `vload` (not CUDA float4)
- **Handwritten SIMD DSL:** `kernels_ptodsl/` `@pto.jit` twins
- Do not mix overlay and deps-native libs in one run

## Current regression snapshot

### simt-asc — **all PASS**

**TileLang:** branch `fix/simtvf-remat-idx-evalset-vec` @ **`a3012088cc61a7fac36c50ae19caf2524d5bf804`** — requires the remat / **pointer-vectorize** (EvalSet vector-gather pack) fix in [tilelang#5](https://github.com/PTO-ISA/tilelang/pull/5).

With that tip + cann_92b2, the simt-asc gate matrix (including SV4 remat_idx arms) is **all PASS**. Do not use the older full-table stamp `31483438` (2026-10-10) as the live simt-asc status.

### simd-pto / handwritten SIMD DSL

| Mode | Notes |
|------|-------|
| **simd-pto** | Active focus (`VectorizeParallelToPTO` / VL). |
| **PTO-DSL `*d`** | Handwritten SIMD DSL twins under `kernels_ptodsl/` — active focus. |

## Design docs

- [`reports/ST_LIST_DESIGN.md`](reports/ST_LIST_DESIGN.md) — SV1–SV9 purpose
- [`reports/ST_CASE3_DESIGN.md`](reports/ST_CASE3_DESIGN.md) — Case-3 RF ladder
- [`reports/ST_SP1_TO_SP6_ACCESS.md`](reports/ST_SP1_TO_SP6_ACCESS.md) — SP access sensitivities
- [`reports/SIMT_LDST_WIDTH_20261008.md`](reports/SIMT_LDST_WIDTH_20261008.md) — elementwise ld/st width (float2 cap)

## License / upstream

Suite sources originated in [PTO-ISA/tilelang](https://github.com/PTO-ISA/tilelang) PR #2 / remat work ([tilelang#5](https://github.com/PTO-ISA/tilelang/pull/5)). This repo is the standalone gym home going forward.
