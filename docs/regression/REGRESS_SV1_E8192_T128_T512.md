# SV1 weekly gate — E=8192 × T=128 + T=512

**Script:** [`scripts/regression/oneshot_sv1_e8192_t128_t512.sh`](../scripts/regression/oneshot_sv1_e8192_t128_t512.sh) (repo root)  
**Mode:** `ST_VF_MODE=simt-asc` (default)  
**Cases:** `sv1_e8192_t128`, `sv1_e8192_t512` — stock TileLang SimtVF, **no AI / schedule rewrites**.  
**Pass rule:** both `COMPILE_OK` and opsim line starts with `PASS`. Exit **0** only if both PASS.

## Why these two

Lok weekly smoke for Ascend SimtVF stream-eltwise at fixed large E with the two thread counts used in bubble / schedule discussion (T128 baseline vs T512 warp-fill). Full SV1–SV9 oneshot stays separate (`oneshot_sv1_sv9.sh`).

## How to run (pto-b10)

```bash
git clone https://github.com/PTO-ISA/tilelangGym.git
cd tilelangGym
git checkout feat/seed-st-suite-sv-cf-sp   # or main after merge

export TILELANG_DEPS=/mnt/fluxdata/happybot/projects/tilelang-pto-isa-st
export CAMODEL_DEPS=/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018
export ASCEND_HOME_PATH=/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2
export PY=/home/happybot/projects/tilelang-deepseek/.venv-npu/bin/python
export SIM_DSL=/home/happybot/PTOAS-vmi/scripts/sim_dsl.sh
# optional overrides:
# export ST_SIMTVF_OUT=/tmp/sv1_e8192_gate
# export SOC=Ascend950PR_9599

bash scripts/regression/oneshot_sv1_e8192_t128_t512.sh
echo exit=$?
```

Artifacts: `$ST_SIMTVF_OUT` (default `/tmp/sv1_e8192_t128_t512_gate`) — `SUMMARY_sv1_e8192_t128_t512.txt`, `logs/`, `opsim_*.log`, `so/`.

## Pins

See repo README / `benches/sv1/config.toml` `[pins]`. Prefer remat tip `a3012088` + CANN cann_92b2 when available; lab may still use cann_91b3 for older keeps.

## Related analysis (not required for gate PASS)

`benches/sv1/ANALYSIS.md` — roof **16**, ILP16@T128, T128 schedule todos. Gate script does **not** run ILP rewrites.
