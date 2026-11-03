#!/usr/bin/env bash
# Dual-era CF7d oneshot: hoisted + reduce-in-b, B/C only.
# Run on pto-b10. NO board .39. Do NOT source ~/projects/env.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${ST_SIMTVF_OUT:-/tmp/st_simtvf_parallel/ptodsl_cf7d_rev}"
mkdir -p "$OUT"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$HOME/projects/tilelang-deepseek/.venv-npu/bin/python" ]]; then
    PY="$HOME/projects/tilelang-deepseek/.venv-npu/bin/python"
  else
    PY="$HOME/projects/venv-ptoas-vmi-0.1.3/bin/python3"
  fi
else
  PY="$PYTHON_BIN"
fi
ASC="${ASCEND_HOME_PATH:-/mnt/fluxdata/Ascend/cann_91b3/cann-9.1.0-beta.3}"
PTOAS_ROOT="${PTOAS_ROOT:-$HOME/PTOAS-vmi}"
SIM_DSL="${SIM_DSL:-$PTOAS_ROOT/scripts/sim_dsl.sh}"
SOC="${SOC_VERSION:-Ascend950PR_9599}"

set +u
[[ -f "$ASC/set_env.sh" ]] && source "$ASC/set_env.sh" || true
set -u
export ASCEND_HOME_PATH="$ASC"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHON_BIN="$PY"
DEPS="${PTODSL_DEPS:-/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018}"
MLIR_PY="${MLIR_PYTHON_ROOT:-/mnt/fluxdata/happybot/llvm-vpto/build-llvm21/tools/mlir/python_packages/mlir_core}"
export PYTHONPATH="${DEPS}:${MLIR_PY}"
export PATH="${DEPS}/bin:${PATH}"
export PTOAS_MSPROF_PRIVATE_ROOT="${PTOAS_MSPROF_PRIVATE_ROOT:-/tmp/ptoas-msprof}"
mkdir -p "$PTOAS_MSPROF_PRIVATE_ROOT"

echo "[cf7d-rev] OUT=$OUT PY=$PY SOC=$SOC ROOT=$ROOT $(date -Is)"
if ! "$PY" -c 'from ptodsl import pto; print("ptodsl OK")'; then
  echo "[cf7d-rev] BLOCKED: cannot import ptodsl"; exit 2
fi
if [[ ! -x "$SIM_DSL" ]]; then
  echo "[cf7d-rev] BLOCKED: SIM_DSL missing $SIM_DSL"; exit 2
fi

SUMMARY="$OUT/SUMMARY.tsv"
echo -e "era\ttag\tstatus\twall_us\tvf_cycles\tbody_instr\tmaxabs\tnote" > "$SUMMARY"

harvest_vf() {
  local odir="$1"
  local vf="" body=""
  local csv
  csv=$(find "$odir" -name 'core0.veccore0_instr_exe.csv' 2>/dev/null | head -n 1 || true)
  if [[ -n "$csv" && -f "$csv" ]]; then
    vf=$(python3 -c '
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
s=0
for r in rows:
  if r.get("instr","").strip() in ("VF","VF_SIMT"):
    s += int(r["cycles"]) * int(r.get("call_count") or 1)
print(s if s else "NA")
' "$csv")
    body=$(python3 -c '
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
ctrl={"set_flag","wait_flag","end","end_label","nop","push_pb","dcci"}
s=0
for r in rows:
  instr=r.get("instr","").strip()
  if instr in ("VF","VF_SIMT") or instr in ctrl: continue
  s += int(r.get("call_count") or 0)
print(s)
' "$csv")
  fi
  echo "${vf:-NA} ${body:-NA}"
}


run_one() {
  local era="$1"; shift
  local tag="$1"; shift
  local pyfile="$1"; shift
  local odir="$OUT/opsim_${tag}"
  mkdir -p "$odir"
  echo "==== [$era] $tag ===="
  local rc=0
  "$SIM_DSL" --soc-version "$SOC" --output "$odir" \
    "$ROOT/$pyfile" -- "$@" > "$OUT/${tag}.log" 2>&1 || rc=$?
  tail -n 15 "$OUT/${tag}.log" || true

  local status="FAIL_OR_BLOCKED"
  local maxabs=""
  local wall=""
  if grep -q "PASS ${tag}" "$OUT/${tag}.log" 2>/dev/null; then
    status="PASS"
    maxabs=$(grep -oE 'maxabs=[0-9.eE+-]+' "$OUT/${tag}.log" | head -n 1 | cut -d= -f2 || true)
  elif grep -qiE 'soft link is not supported' "$OUT/${tag}.log" 2>/dev/null; then
    status="MSPROF_SOFTLINK_BLOCK"
  elif grep -qiE 'exceeded vf stack|COMPILE_FAIL|Traceback|spill|AttributeError|SyntaxError|NameError' "$OUT/${tag}.log" 2>/dev/null; then
    status="COMPILE_FAIL"
  fi
  wall=$(grep -E 'core0\.veccore0[[:space:]]+[0-9]+\.[0-9]+' "$OUT/${tag}.log" 2>/dev/null \
    | head -n 1 | grep -oE '[0-9]+\.[0-9]+' | head -n 1 || true)
  if [[ -z "$wall" ]]; then
    wall=$(grep -RhoE 'core0\.veccore0[[:space:]]+[0-9]+\.[0-9]+' "$odir" 2>/dev/null \
      | head -n 1 | grep -oE '[0-9]+\.[0-9]+' | head -n 1 || true)
  fi
  local vf_body vf_cyc body_instr
  vf_body=$(harvest_vf "$odir")
  vf_cyc=$(echo "$vf_body" | awk '{print $1}')
  body_instr=$(echo "$vf_body" | awk '{print $2}')
  echo -e "${era}\t${tag}\t${status}\t${wall:-NA}\t${vf_cyc}\t${body_instr}\t${maxabs:-NA}\trc=${rc}" | tee -a "$SUMMARY"
  echo "RESULT $tag $status wall_us=${wall:-NA} vf_cyc=${vf_cyc} body=${body_instr}"
}

GS="${CF7D_GS:-0 8 16 32 64}"
PGS="${CF7D_PGS:-25 50}"
ARMS="${CF7D_ARMS:-B C}"

for G in $GS; do
  for PG in $PGS; do
    for ARM in $ARMS; do
      TAG="cf7d_e256_b8_g${G}_vl64_pgate${PG}_arm${ARM}"
      run_one "hoisted" "$TAG" "cf7d_outer_inner_if.py" 256 8 "$G" "$PG" "$ARM"
      TAGR="cf7d_rib_e256_b8_g${G}_vl64_pgate${PG}_arm${ARM}"
      run_one "rib" "$TAGR" "cf7d_reduce_in_b.py" 256 8 "$G" "$PG" "$ARM"
    done
  done
done

echo "[cf7d-rev] done $(date -Is). SUMMARY:"
cat "$SUMMARY"
