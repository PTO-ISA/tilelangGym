#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Usage:
#   bash run.sh                         # A5, C assign (default); all cases
#   bash run.sh --a6                    # A6 (dav-920r1-vec / dav_9201)
#   bash run.sh --asm                   # LDS.b128/STS.b128 inline asm (A5)
#   bash run.sh -c case1_float_64_128   # one case; suite prefix is optional
#   bash run.sh --case SIMTLds128Test.case2_float_128_128
#
# Env overrides:
#   PTO_ARCH=a5|a6  SOC_VERSION=...  RUN_MODE=sim|npu
#   A5_SET_ENV / A6_SET_ENV  paths to set_env.sh
#   SKIP_SET_ENV=1  if ASCEND_HOME_PATH is already configured

set -euo pipefail

usage()
{
    echo "Usage: $0 [-c|--case <gtest-case>] [--a5|--a6] [--asm|--no-asm]"
    echo "  -c, --case   GTest case to run. Accepts case1_float_64_128 or SIMTLds128Test.case1_float_64_128."
    echo "               Omit to run all cases."
    echo "  --a5         Build for A5: --cce-aicore-arch=dav-c310-vec, SOC=Ascend950PR_9599 (default)."
    echo "  --a6         Build for A6: --cce-aicore-arch=dav-920r1-vec, SOC=dav_9201."
    echo "  --asm        Build with SIMT_LDS128_USE_INLINE_ASM=ON (explicit LDS.b128/STS.b128)."
    echo "  --no-asm     Build with C dst[i] = src[i] (default)."
}

CASE=""
PTO_ARCH="${PTO_ARCH:-a5}"
USE_INLINE_ASM="${SIMT_LDS128_USE_INLINE_ASM:-OFF}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c | --case)
            if [[ $# -lt 2 ]]; then
                echo "missing case name after $1" >&2
                usage >&2
                exit 1
            fi
            CASE="$2"
            shift 2
            ;;
        --a5)
            PTO_ARCH=a5
            shift
            ;;
        --a6)
            PTO_ARCH=a6
            shift
            ;;
        --asm)
            USE_INLINE_ASM=ON
            shift
            ;;
        --no-asm)
            USE_INLINE_ASM=OFF
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

A5_SET_ENV="${A5_SET_ENV:-/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/set_env.sh}"
A6_SET_ENV="${A6_SET_ENV:-/home/chenjinlin/cann9.1b3/cann-9.1.0-beta.3/set_env.sh}"

if [[ -z "${SKIP_SET_ENV:-}" ]]; then
    if [[ "${PTO_ARCH}" == "a6" ]]; then
        # shellcheck disable=SC1090
        source "${A6_SET_ENV}"
    else
        # shellcheck disable=SC1090
        source "${A5_SET_ENV}"
    fi
fi

if [[ -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ASCEND_HOME_PATH}/bin/setenv.bash"
fi

RUN_MODE="${RUN_MODE:-sim}"
if [[ "${PTO_ARCH}" == "a6" ]]; then
    SOC_VERSION="${SOC_VERSION:-dav_9201}"
    BUILD_DIR="${BUILD_DIR:-build_a6}"
    # A6 camodel opens many dump files.
    ulimit -n 1048576 || true
    export CAMODEL_LOG_PATH="${CAMODEL_LOG_PATH:-camodel_log}"
else
    SOC_VERSION="${SOC_VERSION:-Ascend950PR_9599}"
    BUILD_DIR="${BUILD_DIR:-build_a5}"
fi

# libruntime_camodel.so needs libnpu_drv_camodel.so from the simulator tree.
# set_env.sh does not put that directory on LD_LIBRARY_PATH.
SIM_LIB="${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib"
export LD_LIBRARY_PATH="${SIM_LIB}:${ASCEND_HOME_PATH}/runtime/lib64/stub:${ASCEND_HOME_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "PTO_ARCH=${PTO_ARCH} SOC_VERSION=${SOC_VERSION} RUN_MODE=${RUN_MODE}"
echo "SIMT_LDS128_USE_INLINE_ASM=${USE_INLINE_ASM} BUILD_DIR=${BUILD_DIR}"
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"

cmake -S . -B "${BUILD_DIR}" \
    -DPTO_ARCH="${PTO_ARCH}" \
    -DRUN_MODE="${RUN_MODE}" \
    -DSOC_VERSION="${SOC_VERSION}" \
    -DSIMT_THREAD_X="${SIMT_THREAD_X:-1024}" \
    -DSIMT_THREAD_Y="${SIMT_THREAD_Y:-1}" \
    -DSIMT_LDS128_USE_INLINE_ASM="${USE_INLINE_ASM}"
cmake --build "${BUILD_DIR}" -j

# The binary reads ../SIMTLds128Test.<case> relative to build/bin.
cd "${BUILD_DIR}"
python3 ../gen_data.py
cd bin

GTEST_ARGS=()
if [[ -n "${CASE}" ]]; then
    if [[ "${CASE}" != *.* ]]; then
        CASE="SIMTLds128Test.${CASE}"
    fi
    echo "Running ${CASE}"
    GTEST_ARGS+=(--gtest_filter="${CASE}")
else
    echo "Running all SIMTLds128Test cases"
fi
./simt_lds128 "${GTEST_ARGS[@]}"
