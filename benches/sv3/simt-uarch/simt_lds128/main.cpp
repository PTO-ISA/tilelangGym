/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <typename T, int tRows, int tCols>
void LaunchSimtLds128(T* out, T* src, void* stream);

class SIMTLds128Test : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

template <typename T, int tRows, int tCols>
void test_simt_lds128(const std::string& refCase)
{
    const std::string refDir = "../SIMTLds128Test." + refCase;
    size_t fileSize = static_cast<size_t>(tRows) * tCols * sizeof(T);
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;
    aclrtMallocHost((void**)(&dstHost), fileSize);
    aclrtMallocHost((void**)(&srcHost), fileSize);
    aclrtMalloc((void**)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(refDir + "/input1.bin", fileSize, srcHost, fileSize);
    aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchSimtLds128<T, tRows, tCols>(dstDevice, srcDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<T> golden(fileSize / sizeof(T));
    std::vector<T> result(dstHost, dstHost + fileSize / sizeof(T));
    ReadFile(refDir + "/golden.bin", fileSize, golden.data(), fileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    bool ret = ResultCmp(golden, result, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(SIMTLds128Test, case1_float_64_128) { test_simt_lds128<float, 64, 128>("case1_float_64_128"); }
