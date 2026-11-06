#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#if defined(PTO_NPU_A6)
#include <pto/npu/a6/common.hpp>
#include <pto/npu/a6/utils.hpp>
#else
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#endif
#include "acl/acl.h"

using namespace pto;

#ifndef SIMT_THREAD_X
#define SIMT_THREAD_X 1024
#endif
#ifndef SIMT_THREAD_Y
#define SIMT_THREAD_Y 1
#endif
constexpr int kThreadX = SIMT_THREAD_X;
constexpr int kThreadY = SIMT_THREAD_Y;
constexpr int kThreadCount = kThreadX * kThreadY;
constexpr int kElemsPerThread = (64 * 128) / kThreadCount;

__simt_vf__ PTO_INTERNAL LAUNCH_BOUND(kThreadCount) void SimtFmulF32(
    __ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x) +
        static_cast<uint32_t>(threadIdx.y) * static_cast<uint32_t>(kThreadX);
    for (uint32_t k = 0; k < static_cast<uint32_t>(kElemsPerThread); ++k) {
        const uint32_t i = tid + k * static_cast<uint32_t>(kThreadCount);
        dst[i] = src0[i] * src1[i];
    }
}

__simt_vf__ PTO_INTERNAL LAUNCH_BOUND(kThreadCount) void SimtFdivF32(
    __ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x) +
        static_cast<uint32_t>(threadIdx.y) * static_cast<uint32_t>(kThreadX);
    for (uint32_t k = 0; k < static_cast<uint32_t>(kElemsPerThread); ++k) {
        const uint32_t i = tid + k * static_cast<uint32_t>(kThreadCount);
        dst[i] = src0[i] / src1[i];
    }
}

__simt_vf__ PTO_INTERNAL LAUNCH_BOUND(kThreadCount) void SimtFmnmaxF32(
    __ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x) +
        static_cast<uint32_t>(threadIdx.y) * static_cast<uint32_t>(kThreadX);
    for (uint32_t k = 0; k < static_cast<uint32_t>(kElemsPerThread); ++k) {
        const uint32_t i = tid + k * static_cast<uint32_t>(kThreadCount);
        dst[i] = max(src0[i], src1[i]);
    }
}

__simt_vf__ PTO_INTERNAL LAUNCH_BOUND(kThreadCount) void SimtReduxF32(
    __ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1)
{
    (void)src1;
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x) +
        static_cast<uint32_t>(threadIdx.y) * static_cast<uint32_t>(kThreadX);
    for (uint32_t k = 0; k < static_cast<uint32_t>(kElemsPerThread); ++k) {
        const uint32_t i = tid + k * static_cast<uint32_t>(kThreadCount);
        dst[i] = __reduce_max(src0[i]);
    }
}

#define SIMT_WRAP(TAG, FN)                                                                 \
    template <typename TileData>                                                           \
    __tf__ PTO_INTERNAL void Simt##TAG(                                                    \
        typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,\
        typename TileData::TileDType __in__ src1)                                          \
    {                                                                                      \
        __ubuf__ float* dstPtr = (__ubuf__ float*)__cce_get_tile_ptr(dst);                 \
        __ubuf__ float* src0Ptr = (__ubuf__ float*)__cce_get_tile_ptr(src0);               \
        __ubuf__ float* src1Ptr = (__ubuf__ float*)__cce_get_tile_ptr(src1);               \
        cce::async_invoke<FN>(cce::dim3{kThreadX, kThreadY}, dstPtr, src0Ptr, src1Ptr);    \
    }                                                                                      \
    template <typename T, int tRows, int tCols>                                            \
    __global__ AICORE void runSimt##TAG(                                                   \
        __gm__ T __out__* out, __gm__ T __in__* src0, __gm__ T __in__* src1, int vRows,    \
        int vCols)                                                                         \
    {                                                                                      \
        using DynShape = pto::Shape<-1, -1, -1, -1, -1>;                                   \
        using DynStride = pto::Stride<-1, -1, -1, -1, -1>;                                 \
        using GlobalData = GlobalTensor<T, DynShape, DynStride>;                          \
        constexpr int tileH = tRows;                                                       \
        constexpr int tileW = tCols;                                                       \
        constexpr unsigned tileBytes =                                                     \
            static_cast<unsigned>(tileH) * static_cast<unsigned>(tileW) * sizeof(T);       \
        static_assert(tileH * tileW == kThreadCount * kElemsPerThread, "tile coverage");  \
        static_assert(kThreadCount <= 2048 && kThreadCount >= 32 && kThreadCount % 32 == 0,\
                      "full warps, at most 2048");                                         \
        static_assert((64 * 128) % kThreadCount == 0, "tile divides across threads");     \
        static_assert(tileBytes * 3 <= 256 * 1024, "UB overflow");                         \
        using TileData = Tile<TileType::Vec, T, tileH, tileW, BLayout::RowMajor, -1, -1>; \
        TileData src0Tile(vRows, vCols);                                                   \
        TileData src1Tile(vRows, vCols);                                                   \
        TileData dstTile(vRows, vCols);                                                    \
        TASSIGN(src0Tile, 0x0);                                                            \
        TASSIGN(src1Tile, tileBytes);                                                      \
        TASSIGN(dstTile, tileBytes * 2);                                                   \
        GlobalData src0Global(src0, pto::Shape(1, 1, 1, vRows, vCols),                     \
                              pto::Stride(1, 1, 1, tCols, 1));                             \
        GlobalData src1Global(src1, pto::Shape(1, 1, 1, vRows, vCols),                     \
                              pto::Stride(1, 1, 1, tCols, 1));                             \
        GlobalData dstGlobal(out, pto::Shape(1, 1, 1, vRows, vCols),                       \
                             pto::Stride(1, 1, 1, tCols, 1));                              \
        TLOAD(src0Tile, src0Global);                                                       \
        TLOAD(src1Tile, src1Global);                                                       \
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);                                            \
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);                                           \
        Simt##TAG<TileData>(dstTile.data(), src0Tile.data(), src1Tile.data());             \
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);                                            \
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);                                           \
        TSTORE(dstGlobal, dstTile);                                                        \
    }                                                                                      \
    template <typename T, int tRows, int tCols>                                            \
    void LaunchSimt##TAG(T* out, T* src0, T* src1, void* stream)                           \
    {                                                                                      \
        runSimt##TAG<T, tRows, tCols><<<1, nullptr, stream>>>(out, src0, src1, tRows, tCols);\
    }

SIMT_WRAP(Fmul, SimtFmulF32)
SIMT_WRAP(Fdiv, SimtFdivF32)
SIMT_WRAP(Fmnmax, SimtFmnmaxF32)
SIMT_WRAP(Redux, SimtReduxF32)

template void LaunchSimtFmul<float, 64, 128>(float*, float*, float*, void*);
template void LaunchSimtFdiv<float, 64, 128>(float*, float*, float*, void*);
template void LaunchSimtFmnmax<float, 64, 128>(float*, float*, float*, void*);
template void LaunchSimtRedux<float, 64, 128>(float*, float*, float*, void*);
