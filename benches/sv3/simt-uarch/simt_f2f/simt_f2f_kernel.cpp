// SIMT half-to-float conversion. Same 64x128 tile as the other copies.
//   TLOAD  GM -> UB (half src)
//   SIMT   dst[i] = (float)src[i]
//   TSTORE UB -> GM (float dst)

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

__simt_vf__ PTO_INTERNAL LAUNCH_BOUND(kThreadCount) void SimtF2F(
    __ubuf__ float* dst, __ubuf__ half* src)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x) +
        static_cast<uint32_t>(threadIdx.y) * static_cast<uint32_t>(kThreadX);

    for (uint32_t k = 0; k < static_cast<uint32_t>(kElemsPerThread); ++k) {
        const uint32_t i = tid + k * static_cast<uint32_t>(kThreadCount);
        dst[i] = static_cast<float>(src[i]);
    }
}

template <typename SrcTile, typename DstTile>
__tf__ PTO_INTERNAL void SimtF2fCopy(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src)
{
    __ubuf__ float* dstPtr = (__ubuf__ float*)__cce_get_tile_ptr(dst);
    __ubuf__ half* srcPtr = (__ubuf__ half*)__cce_get_tile_ptr(src);
    cce::async_invoke<SimtF2F>(cce::dim3{kThreadX, kThreadY}, dstPtr, srcPtr);
}

template <int tRows, int tCols>
__global__ AICORE void runSimtF2f(__gm__ float __out__* out, __gm__ half __in__* src, int vRows, int vCols)
{
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;

    constexpr int tileH = tRows;
    constexpr int tileW = tCols;
    constexpr unsigned srcBytes = static_cast<unsigned>(tileH) * static_cast<unsigned>(tileW) * sizeof(half);
    constexpr unsigned dstBytes = static_cast<unsigned>(tileH) * static_cast<unsigned>(tileW) * sizeof(float);
    static_assert(tileH * tileW == kThreadCount * kElemsPerThread, "tile must cover every thread element");
    static_assert(kThreadCount <= 2048, "at most 2048 threads");
    static_assert(kThreadCount >= 32 && kThreadCount % 32 == 0, "full warps");
    static_assert((64 * 128) % kThreadCount == 0, "tile divides across threads");
    static_assert(srcBytes + dstBytes <= 256 * 1024, "UB overflow");

    using SrcTile = Tile<TileType::Vec, half, tileH, tileW, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, float, tileH, tileW, BLayout::RowMajor, -1, -1>;
    SrcTile srcTile(vRows, vCols);
    DstTile dstTile(vRows, vCols);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, srcBytes);

    GlobalTensor<half, DynShape, DynStride> srcGlobal(
        src, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(1, 1, 1, tCols, 1));
    GlobalTensor<float, DynShape, DynStride> dstGlobal(
        out, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(1, 1, 1, tCols, 1));

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    SimtF2fCopy<SrcTile, DstTile>(dstTile.data(), srcTile.data());

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
}

extern "C" void LaunchSimtF2f(float* out, void* src, void* stream)
{
    runSimtF2f<64, 128><<<1, nullptr, stream>>>((float*)out, (half*)src, 64, 128);
}
