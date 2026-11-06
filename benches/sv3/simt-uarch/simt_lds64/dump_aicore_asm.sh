#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Dump the NPU device object embedded in a kernel shared library.
#   dump_aicore_asm.sh /path/to/xxx_kernel.so
# Writes xxx_kernel.o and xxx_kernel.S next to the input .so.

set -euo pipefail

if [[ $# -ne 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 <xxx_kernel.so>" >&2
    exit 1
fi

SO="$(readlink -f "$1")"
if [[ ! -f "${SO}" ]]; then
    echo "kernel so not found: $1" >&2
    exit 1
fi

export ASCEND_HOME_PATH=/home/chenjinlin/cann9.3.0-1014/cann-9.3.0
# shellcheck disable=SC1091
source "${ASCEND_HOME_PATH}/set_env.sh"

OBJCOPY="${ASCEND_HOME_PATH}/bin/llvm-objcopy"
OBJDUMP="${ASCEND_HOME_PATH}/bin/llvm-objdump"

if [[ -z "${OBJCOPY}" || ! -x "${OBJCOPY}" || -z "${OBJDUMP}" || ! -x "${OBJDUMP}" ]]; then
    echo "llvm-objcopy / llvm-objdump not found. Source CANN set_env.sh first." >&2
    exit 1
fi

STEM="${SO%.so}"
OBJ="${STEM}.o"
ASM="${STEM}.S"

"${OBJCOPY}" -O binary -j .aicore_binary "${SO}" "${OBJ}"
"${OBJDUMP}" -d "${OBJ}" > "${ASM}"

echo "wrote ${OBJ}"
echo "wrote ${ASM}"
