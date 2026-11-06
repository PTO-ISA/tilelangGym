#!/usr/bin/env python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os

import numpy as np

np.random.seed(19)


def gen_golden_data(param):
    dtype = param.dtype
    height, width = param.tile_row, param.tile_col
    src = np.random.randint(1, 10, size=[height, width]).astype(dtype)
    golden = src.copy()
    src.tofile("input1.bin")
    golden.tofile("golden.bin")


class SimtLds128AsmParams:
    def __init__(self, dtype, tile_row, tile_col):
        self.dtype = dtype
        self.tile_row = tile_row
        self.tile_col = tile_col


def generate_case_name(idx, param):
    dtype_str = {
        np.float32: "float",
        np.float16: "half",
        np.int32: "int32",
        np.int16: "int16",
        np.int8: "int8",
        np.uint8: "uint8",
    }[param.dtype]
    return f"SIMTLds128AsmTest.case{idx}_{dtype_str}_{param.tile_row}_{param.tile_col}"


if __name__ == "__main__":
    case_params_list = [
        SimtLds128AsmParams(np.float32, 64, 128),
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(i + 1, param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(param)
        os.chdir(original_dir)
