#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <typename T, int tRows, int tCols>
void LaunchSimtFmul(T* out, T* src0, T* src1, void* stream);
template <typename T, int tRows, int tCols>
void LaunchSimtFdiv(T* out, T* src0, T* src1, void* stream);
template <typename T, int tRows, int tCols>
void LaunchSimtFmnmax(T* out, T* src0, T* src1, void* stream);
template <typename T, int tRows, int tCols>
void LaunchSimtRedux(T* out, T* src0, T* src1, void* stream);

class SIMTArithTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

template <typename T, int tRows, int tCols>
void test_op(const std::string& refCase, void (*launch)(T*, T*, T*, void*))
{
    const std::string refDir = "../SIMTArithTest." + refCase;
    size_t fileSize = static_cast<size_t>(tRows) * tCols * sizeof(T);
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;
    aclrtMallocHost((void**)(&dstHost), fileSize);
    aclrtMallocHost((void**)(&src0Host), fileSize);
    aclrtMallocHost((void**)(&src1Host), fileSize);
    aclrtMalloc((void**)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(refDir + "/input1.bin", fileSize, src0Host, fileSize);
    ReadFile(refDir + "/input2.bin", fileSize, src1Host, fileSize);
    aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launch(dstDevice, src0Device, src1Device, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<T> golden(fileSize / sizeof(T));
    std::vector<T> result(dstHost, dstHost + fileSize / sizeof(T));
    ReadFile(refDir + "/golden.bin", fileSize, golden.data(), fileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    EXPECT_TRUE(ResultCmp(golden, result, 0.001f));
}

TEST_F(SIMTArithTest, case_fmul) {
    test_op<float, 64, 128>("case_fmul", LaunchSimtFmul<float, 64, 128>);
}
TEST_F(SIMTArithTest, case_fdiv) {
    test_op<float, 64, 128>("case_fdiv", LaunchSimtFdiv<float, 64, 128>);
}
TEST_F(SIMTArithTest, case_fmnmax) {
    test_op<float, 64, 128>("case_fmnmax", LaunchSimtFmnmax<float, 64, 128>);
}
TEST_F(SIMTArithTest, case_redux) {
    test_op<float, 64, 128>("case_redux", LaunchSimtRedux<float, 64, 128>);
}
