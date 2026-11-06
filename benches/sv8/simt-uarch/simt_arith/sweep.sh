#!/bin/bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${ROOT}/../thread-sweep"
mkdir -p "${OUT}"
source /mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/set_env.sh
SIM_LIB="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib"
export LD_LIBRARY_PATH="${SIM_LIB}:${ASCEND_HOME_PATH}/runtime/lib64/stub:${ASCEND_HOME_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "${ROOT}"
for t in 32 128 256 512 1024 2048; do
  echo "BUILD t=${t}"
  SIMT_THREAD_X="${t}" SIMT_THREAD_Y=1 bash run.sh -c case_fmul > "${OUT}/fmul-t${t}.log" 2>&1
  echo "fmul t=${t} rc=$?"
  python3 count_simt_vf_cycles.py build_a5/bin > "${OUT}/fmul-t${t}.count"
  rm -f build_a5/bin/core*.dump
  for op in fdiv fmnmax redux; do
    (cd build_a5/bin && ./simt_arith --gtest_filter="SIMTArithTest.case_${op}") > "${OUT}/${op}-t${t}.log" 2>&1
    echo "${op} t=${t} rc=$?"
    python3 count_simt_vf_cycles.py build_a5/bin > "${OUT}/${op}-t${t}.count"
    rm -f build_a5/bin/core*.dump
  done
done
echo SWEEP_DONE
