#!/usr/bin/env bash
# Single SV1 regression gate for weekly/CI: E=8192 × T∈{128,512}, simt-asc only.
# Expect both cases COMPILE_OK + opsim PASS with no AI / manual fixes.
#
# How to run (pto-b10 example):
#   cd /path/to/tilelangGym
#   export TILELANG_DEPS=...   # tree with build/lib/libtilelang.so
#   export CAMODEL_DEPS=...
#   export ASCEND_HOME_PATH=.../cann-9.2.0-beta.2   # or lab CANN root with set_env.sh
#   export PY=/path/to/.venv-npu/bin/python
#   export SIM_DSL=/path/to/PTOAS/scripts/sim_dsl.sh   # or PTOAS_ROOT=...
#   export ST_VF_MODE=simt-asc
#   bash oneshot_sv1_e8192_t128_t512.sh
#
# Optional:
#   ST_SIMTVF_OUT=/tmp/sv1_e8192_gate   # output root (default /tmp/sv1_e8192_t128_t512_gate)
#   SOC=Ascend950PR_9599
#
# Exit 0 iff both tags PASS; non-zero otherwise.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

OUT="${ST_SIMTVF_OUT:-/tmp/sv1_e8192_t128_t512_gate}"
LOG=$OUT/oneshot_sv1_e8192_t128_t512.log
mkdir -p "$OUT/so" "$OUT/logs" "$OUT/sources" "$OUT/reports"
exec > >(tee -a "$LOG") 2>&1
echo "=== SV1 E=8192 T128+T512 gate $(date -Is) host=$(hostname) user=$(whoami) ==="

if [[ -z "${PY:-}" && -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
fi
: "${PY:?Set PY to the NPU venv interpreter (PYTHON_BIN is also accepted)}"
: "${ASCEND_HOME_PATH:?Set ASCEND_HOME_PATH to the CANN toolkit root (directory containing set_env.sh)}"
: "${TILELANG_DEPS:?Set TILELANG_DEPS to the TileLang tree that contains build/lib/libtilelang.so}"
: "${CAMODEL_DEPS:?Set CAMODEL_DEPS to the camodel dependency prefix (its bin/ is prepended to PATH)}"
if [[ -z "${SIM_DSL:-}" && -n "${PTOAS_ROOT:-}" ]]; then
  SIM_DSL="$PTOAS_ROOT/scripts/sim_dsl.sh"
fi
: "${SIM_DSL:?Set SIM_DSL to sim_dsl.sh, or set PTOAS_ROOT to use \$PTOAS_ROOT/scripts/sim_dsl.sh}"

DEPS=$TILELANG_DEPS
ASC=$ASCEND_HOME_PATH
CAMO=$CAMODEL_DEPS
SIM=$SIM_DSL
SOC="${SOC:-${SOC_VERSION:-Ascend950PR_9599}}"
export PY SIM_DSL SOC ST_VF_MODE="${ST_VF_MODE:-simt-asc}"

HERE=$(cd "$(dirname "$0")" && pwd)
SUITE=${SUITE:-$HERE}
if [[ ! -f "$SUITE/common_asc_harness.py" ]]; then
  echo "error: common_asc_harness.py not found under SUITE=$SUITE" >&2
  exit 1
fi
export ST_SIMTVF_OUT=$OUT
export PYTHONPATH="$SUITE:${PYTHONPATH:-}"

KERNEL="$SUITE/kernels/sv1_stream_eltwise.py"
if [[ ! -f "$KERNEL" ]]; then
  KERNEL="$SUITE/benches/sv1/sv1_stream_eltwise.py"
fi
if [[ ! -f "$KERNEL" ]]; then
  echo "error: sv1_stream_eltwise.py not found" >&2
  exit 1
fi

if [[ ! -f /tmp/run_cf_mb_opsim.py ]]; then
  _opsim_src=""
  for c in \
    "${RUN_CF_MB_OPSIM:-}" \
    "$DEPS/examples/ascend/run_cf_mb_opsim.py" \
    "$HERE/run_cf_mb_opsim.py"
  do
    if [[ -n "$c" && -f "$c" ]]; then
      _opsim_src="$c"
      break
    fi
  done
  if [[ -z "$_opsim_src" ]]; then
    echo "error: run_cf_mb_opsim.py not found. Set RUN_CF_MB_OPSIM or place under \$TILELANG_DEPS/examples/ascend/" >&2
    exit 1
  fi
  cp -f "$_opsim_src" /tmp/run_cf_mb_opsim.py
fi
cp -f "$SUITE"/run_opsim_*.py /tmp/ 2>/dev/null || true
sed -i 's/\r$//' "$SUITE"/*.py "$KERNEL" /tmp/run_opsim_*.py 2>/dev/null || true

BK=/tmp/libtilelang.so.deps_backup_sv1_e8192_gate
if [[ ! -f "$BK" ]]; then
  cp -a "$DEPS/build/lib/libtilelang.so" "$BK"
fi
cp -a "$BK" "$DEPS/build/lib/libtilelang.so"
trap 'cp -a "$BK" "$DEPS/build/lib/libtilelang.so" 2>/dev/null || true; echo "restored deps lib"' EXIT

set +u; source "$ASC/set_env.sh"; set -u
export ASCEND_HOME_PATH=$ASC
export PATH="$CAMO/bin:$PATH"
export PYTHONPATH="$SUITE:$DEPS:$DEPS/build:$CAMO:${ASC}/python/site-packages:${PYTHONPATH:-}"
export TILELANG_DISABLE_CACHE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CPLUS_INCLUDE_PATH="/usr/include/c++/12:/usr/include/aarch64-linux-gnu/c++/12${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export TILELANG_DISABLE_DATA_RACE_CHECK=1

echo "PY=$PY ASC=$ASC DEPS=$DEPS CAMO=$CAMO SIM=$SIM SOC=$SOC ST_VF_MODE=$ST_VF_MODE"
echo "KERNEL=$KERNEL"

SUMMARY="$OUT/SUMMARY_sv1_e8192_t128_t512.txt"
: > "$SUMMARY"
FAILS=0

run_one() {
  local TAG=$1 E=$2 T=$3
  echo "==== COMPILE $TAG E=$E T=$T ====" | tee -a "$SUMMARY"
  set +e
  "$PY" "$KERNEL" "$E" "$T" 2>&1 | tee "$OUT/logs/compile_${TAG}.log"
  local RC=${PIPESTATUS[0]}
  set -e
  if [[ "$RC" -ne 0 || ! -f "$OUT/so/${TAG}.so" ]]; then
    echo "COMPILE_FAIL $TAG rc=$RC" | tee -a "$SUMMARY"
    FAILS=$((FAILS + 1))
    return 0
  fi
  echo "COMPILE_OK $TAG" | tee -a "$SUMMARY"
  set +e
  "$SIM" --soc-version "$SOC" --output "$OUT/opsim_${TAG}" \
    /tmp/run_opsim_generic.py -- "$TAG" "$OUT" 2>&1 | tee "$OUT/opsim_${TAG}.log"
  set -e
  local PASSLINE
  PASSLINE=$(grep -E '^PASS|^FAIL' "$OUT/opsim_${TAG}.log" | tail -1 || true)
  echo "$TAG $PASSLINE" | tee -a "$SUMMARY"
  if [[ "$PASSLINE" != PASS* ]]; then
    FAILS=$((FAILS + 1))
  fi
}

run_one sv1_e8192_t128 8192 128
run_one sv1_e8192_t512 8192 512

echo "==== SUMMARY ===="
cat "$SUMMARY"
echo "FAILS=$FAILS"
if [[ "$FAILS" -eq 0 ]]; then
  echo "GATE_PASS sv1_e8192_t128 sv1_e8192_t512"
  exit 0
fi
echo "GATE_FAIL count=$FAILS"
exit 1
