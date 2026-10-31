#!/usr/bin/env bash
# PTO-DSL CF7d — outer gate A/B/C × G × p_gate first-cut opsim (Ascend950PR_9599).
# Run on pto-b10 login node. NO board .39.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${ST_SIMTVF_OUT:-/tmp/st_simtvf_parallel/ptodsl_cf7d}"
mkdir -p "$OUT"
PY="${PYTHON_BIN:-$HOME/projects/tilelang-deepseek/.venv-npu/bin/python}"
ASC="${ASCEND_HOME_PATH:-/mnt/fluxdata/Ascend/cann_91b3/cann-9.1.0-beta.3}"
PTOAS_ROOT="${PTOAS_ROOT:-$HOME/PTOAS-vmi}"
SIM_DSL="${SIM_DSL:-$PTOAS_ROOT/scripts/sim_dsl.sh}"
SOC="${SOC_VERSION:-Ascend950PR_9599}"

# Do NOT source ~/projects/env.sh — it pollutes LD/PYTHONPATH and kills AIV
# block_start (CF5d regress 2026-10-30). Match Oct-8 PASS recipe.
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
# ~/.cache is symlink -> fluxdata; msprof rejects soft-linked staging dirs
export PTOAS_MSPROF_PRIVATE_ROOT="${PTOAS_MSPROF_PRIVATE_ROOT:-/tmp/ptoas-msprof}"
mkdir -p "$PTOAS_MSPROF_PRIVATE_ROOT"

echo "[ptodsl-cf7d] OUT=$OUT PY=$PY SOC=$SOC ROOT=$ROOT"
if ! "$PY" -c 'from ptodsl import pto; print("ptodsl OK")'; then
  echo "[ptodsl-cf7d] BLOCKED: cannot import ptodsl"
  exit 2
fi

SUMMARY="$OUT/SUMMARY.tsv"
echo -e "tag\tstatus\twall_us\tvf_cycles\tbody_instr\tmaxabs\tnote" > "$SUMMARY"

harvest_vf() {
  local odir="$1"
  local vf="" body=""
  local csv
  csv=$(find "$odir" -name 'core0.veccore0_instr_exe.csv' 2>/dev/null | head -1 || true)
  if [[ -n "$csv" && -f "$csv" ]]; then
    # Prefer VF / VF_SIMT row cycles
    vf=$(awk -F',' 'NR==1{for(i=1;i<=NF;i++){h[$i]=i}}
      NR>1{
        instr=$h["instr"]; gsub(/^[ \t]+|[ \t]+$/,"",instr);
        cyc=$h["cycles"]; cc=$h["call_count"];
        if(instr=="VF" || instr=="VF_SIMT"){print cyc; exit}
      }' "$csv" 2>/dev/null || true)
    body=$(awk -F',' 'NR==1{for(i=1;i<=NF;i++){h[$i]=i}}
      NR>1{
        instr=$h["instr"]; gsub(/^[ \t]+|[ \t]+$/,"",instr);
        cc=$h["call_count"]+0;
        if(instr=="VF"||instr=="VF_SIMT") next;
        if(instr ~ /^(set_flag|wait_flag|end|end_label|nop|push_pb|dcci)$/) next;
        sum+=cc
      } END{print sum+0}' "$csv" 2>/dev/null || true)
  fi
  echo "${vf:-NA} ${body:-NA}"
}

run_one() {
  local tag="$1"; shift
  local pyfile="$1"; shift
  local odir="$OUT/opsim_${tag}"
  mkdir -p "$odir"
  echo "==== $tag ===="
  local rc=0
  if [[ -x "$SIM_DSL" ]]; then
    "$SIM_DSL" --soc-version "$SOC" --output "$odir" \
      "$ROOT/$pyfile" -- "$@" > "$OUT/${tag}.log" 2>&1 || rc=$?
  else
    (cd "$ROOT" && "$PY" "$pyfile" "$@") > "$OUT/${tag}.log" 2>&1 || rc=$?
  fi
  tail -20 "$OUT/${tag}.log" || true

  local status="FAIL_OR_BLOCKED"
  local maxabs=""
  local wall=""
  if grep -q "PASS ${tag}" "$OUT/${tag}.log" 2>/dev/null; then
    status="PASS"
    maxabs=$(grep -oE 'maxabs=[0-9.eE+-]+' "$OUT/${tag}.log" | head -1 | cut -d= -f2 || true)
  elif grep -qiE 'soft link is not supported' "$OUT/${tag}.log" 2>/dev/null; then
    status="MSPROF_SOFTLINK_BLOCK"
  elif grep -qiE 'exceeded vf stack|COMPILE_FAIL|Traceback|spill|AttributeError|SyntaxError|NameError' "$OUT/${tag}.log" 2>/dev/null; then
    status="COMPILE_FAIL"
  fi
  wall=$(grep -E 'core0\.veccore0[[:space:]]+[0-9]+\.[0-9]+' "$OUT/${tag}.log" 2>/dev/null \
    | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
  if [[ -z "$wall" ]]; then
    wall=$(grep -RhoE 'core0\.veccore0[[:space:]]+[0-9]+\.[0-9]+' "$odir" 2>/dev/null \
      | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
  fi
  local vf_body
  vf_body=$(harvest_vf "$odir")
  local vf_cyc body_instr
  vf_cyc=$(echo "$vf_body" | awk '{print $1}')
  body_instr=$(echo "$vf_body" | awk '{print $2}')
  echo -e "${tag}\t${status}\t${wall:-NA}\t${vf_cyc}\t${body_instr}\t${maxabs:-NA}\trc=${rc}" | tee -a "$SUMMARY"
  echo "RESULT $tag $status wall_us=${wall:-NA} vf_cyc=${vf_cyc} body=${body_instr}"
}

# First-cut: G ∈ {0,16,64} × p_gate ∈ {25,50,95} × A/B/C
for G in 0 16 64; do
  for PG in 25 50 95; do
    for ARM in A B C; do
      TAG="cf7d_e256_b8_g${G}_vl64_pgate${PG}_arm${ARM}"
      run_one "$TAG" "../../benches/cf7d/cf7d_outer_inner_if.py" 256 8 "$G" "$PG" "$ARM"
    done
  done
done

echo "[ptodsl-cf7d] done. SUMMARY:"
cat "$SUMMARY"
echo "[ptodsl-cf7d] Logs under $OUT"
