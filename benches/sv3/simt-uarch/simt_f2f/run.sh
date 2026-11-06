#!/bin/bash
set -euo pipefail

CASE=""
PTO_ARCH="${PTO_ARCH:-a5}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--case) CASE="$2"; shift 2 ;;
        --a5) PTO_ARCH=a5; shift ;;
        --a6) PTO_ARCH=a6; shift ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

A5_SET_ENV="${A5_SET_ENV:-/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/set_env.sh}"
if [[ "${PTO_ARCH}" == "a6" ]]; then
    echo "A6 is not set up for this copy" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${A5_SET_ENV}"
if [[ -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ASCEND_HOME_PATH}/bin/setenv.bash"
fi

RUN_MODE="${RUN_MODE:-sim}"
SOC_VERSION="${SOC_VERSION:-Ascend950PR_9599}"
BUILD_DIR="${BUILD_DIR:-build_a5}"
SIM_LIB="${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib"
export LD_LIBRARY_PATH="${SIM_LIB}:${ASCEND_HOME_PATH}/runtime/lib64/stub:${ASCEND_HOME_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cmake -S . -B "${BUILD_DIR}" \
    -DPTO_ARCH="${PTO_ARCH}" \
    -DRUN_MODE="${RUN_MODE}" \
    -DSOC_VERSION="${SOC_VERSION}" \
    -DSIMT_THREAD_X="${SIMT_THREAD_X:-1024}" \
    -DSIMT_THREAD_Y="${SIMT_THREAD_Y:-1}"
cmake --build "${BUILD_DIR}" -j

cd "${BUILD_DIR}"
python3 ../gen_data.py
cd bin
if [[ -n "${CASE}" && "${CASE}" != *.* ]]; then
    CASE="SIMTF2fTest.${CASE}"
fi
if [[ -n "${CASE}" ]]; then
    ./simt_f2f --gtest_filter="${CASE}"
else
    ./simt_f2f
fi
