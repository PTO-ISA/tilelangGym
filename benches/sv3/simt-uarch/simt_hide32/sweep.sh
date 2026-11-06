#!/bin/bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
source /mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/set_env.sh
SIM_LIB="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib"
export TL_CAMODEL_LIB="${SIM_LIB}"
export LD_LIBRARY_PATH="${SIM_LIB}:${ASCEND_HOME_PATH}/runtime/lib64/stub:${ASCEND_HOME_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
PY="/home/onasismelchiorjeyakumar/workspace/tilelang/.venv/bin/python"
OUT="${ROOT}/results.txt"
mkdir -p build
: > "${OUT}"

run_one() {
  local op="$1" opid="$2" threads="$3"
  local so="build/${op}_t${threads}.so"
  echo "BUILD ${op} t=${threads}"
  bisheng -O2 -fPIC -std=c++20 -shared --npu-arch=dav-3510 \
    -I /home/onasismelchiorjeyakumar/workspace/tilelang/src \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -DSIMT_THREADS="${threads}" -DSIMT_OP="${opid}" \
    -Wl,--no-mmap-output-file \
    hide32.asc -o "${so}" || { echo "COMPILE_FAIL ${op} ${threads}" | tee -a "${OUT}"; return; }
  rm -rf "build/npusim_${op}_t${threads}"
  npusim record -s Ascend950 -o "build/npusim_${op}_t${threads}" -g -n 0 \
    "${PY} ${ROOT}/launch.py ${ROOT}/${so} ${op} ${threads}" > "build/${op}_t${threads}.log" 2>&1 || true
  local line
  line=$(rg -o "PASS maxabs=[0-9.e+-]+|FAIL maxabs=[0-9.e+-]+" "build/${op}_t${threads}.log" | tail -1 || true)
  "${PY}" - << PY >> "${OUT}"
import re
from pathlib import Path
from cannsim.core.public.instr_decoder import decode_instr_logs_from_file
root = Path("${ROOT}/build/npusim_${op}_t${threads}")
bins = sorted(root.glob("**/instr.bin"))
opname = {"lds":"SIMT_LDS","ldg":"SIMT_LDG","sts":"SIMT_STS","fadd":"SIMT_FADD","ffma":"SIMT_FFMA"}["${op}"]
print("${op} t=${threads} ${line} bin", "yes" if bins else "no", flush=True)
if not bins:
    raise SystemExit
pat = re.compile(r"\\(ID:\\s*(\\d+)\\)\\s+([A-Z0-9_]+)")
issues, dones = {}, {}
for e in decode_instr_logs_from_file(bins[-1]):
    if e.core != 0 or e.sub != 1:
        continue
    m = pat.search(e.dec or "")
    if not m:
        continue
    key = (e.pc, int(m.group(1)), m.group(2))
    if e.is_popped == 1 and key not in issues:
        issues[key] = e.time
    elif e.is_popped == 0:
        dones[key] = e.time
wins = sorted((issues[k], dones[k]) for k in issues if k[2]=="VF_SIMT" and k in dones)
vf = sum(b-a for a,b in wins)
rows = sorted((issues[k], dones[k]) for k in issues if k[2]==opname and k in dones)
if not rows:
    print("  no", opname, "vf", vf)
else:
    act = sorted(d-s for s,d in rows)
    span = rows[-1][1] - rows[0][0]
    width = 4 if "${op}" != "ldg" else 4
    nbytes = len(rows) * 32 * width
    print(f"  vf={vf} n={len(rows)} span={span} act_med={act[len(act)//2]} bytes={nbytes} B/cyc={nbytes/span:.2f} at128={nbytes/128:.0f}")
PY
}

run_one lds 0 32
run_one lds 0 128
run_one lds 0 256
run_one lds 0 512
run_one ldg 1 32
run_one ldg 1 128
run_one ldg 1 256
run_one ldg 1 512
run_one ldg 1 1024
run_one sts 2 32
run_one sts 2 256
run_one sts 2 512
run_one fadd 3 32
run_one fadd 3 256
run_one fadd 3 512
run_one ffma 4 32
run_one ffma 4 256
run_one ffma 4 512
echo SWEEP_DONE
