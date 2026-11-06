#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>
#include <vector>

using namespace PtoTestCommon;

extern "C" void LaunchSimtF2f(float* out, void* src, void* stream);

void test_simt_f2f()
{
    const std::string refDir = "../SIMTF2fTest.case1_half_64_128";
    constexpr int rows = 64;
    constexpr int cols = 128;
    const size_t n = static_cast<size_t>(rows) * cols;
    const size_t srcBytes = n * sizeof(uint16_t);
    const size_t dstBytes = n * sizeof(float);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint16_t* srcHost = nullptr;
    float* dstHost = nullptr;
    void* srcDevice = nullptr;
    void* dstDevice = nullptr;
    aclrtMallocHost((void**)(&srcHost), srcBytes);
    aclrtMallocHost((void**)(&dstHost), dstBytes);
    aclrtMalloc(&srcDevice, srcBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dstDevice, dstBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(refDir + "/input1.bin", srcBytes, srcHost, srcBytes);
    aclrtMemcpy(srcDevice, srcBytes, srcHost, srcBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchSimtF2f(static_cast<float*>(dstDevice), srcDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstBytes, dstDevice, dstBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<float> golden(n);
    std::vector<float> result(dstHost, dstHost + n);
    ReadFile(refDir + "/golden.bin", dstBytes, golden.data(), dstBytes);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclrtFinalize();

    EXPECT_TRUE(ResultCmp(golden, result, 0.001f));
}

TEST(SIMTF2fTest, case1_half_64_128) { test_simt_f2f(); }
