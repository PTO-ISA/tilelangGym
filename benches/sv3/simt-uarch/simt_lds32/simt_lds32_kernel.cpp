/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// SIMT LDS/STS bandwidth kernel, structured like ttrans runTTRANS_kc_st:
//   TLOAD  GM -> UB (src)
//   SIMT   copy in UB (LDS on src, STS on dst)
//   TSTORE UB -> GM
// Launch is threadX=1024, threadY=1 (1024 threads, 8 elements each). That shape
// had the lowest SIMT VF cycle count among power-of-two shapes with
// threadX * threadY <= 1024. Element k of thread tid is tid + k * 1024, so
// lanes in a warp still touch consecutive floats.

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

__simt_vf__ PTO_INTERNAL LAUNCH_BOUND(kThreadCount) void SimtLds32F32(
    __ubuf__ float* dst, __ubuf__ float* src)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x) +
        static_cast<uint32_t>(threadIdx.y) * static_cast<uint32_t>(kThreadX);

    for (uint32_t k = 0; k < static_cast<uint32_t>(kElemsPerThread); ++k) {
        const uint32_t i = tid + k * static_cast<uint32_t>(kThreadCount);
        dst[i] = src[i];
    }
}

template <typename TileData>
__tf__ PTO_INTERNAL void SimtLds32(
    typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src)
{
    __ubuf__ float* dstPtr = (__ubuf__ float*)__cce_get_tile_ptr(dst);
    __ubuf__ float* srcPtr = (__ubuf__ float*)__cce_get_tile_ptr(src);
    cce::async_invoke<SimtLds32F32>(cce::dim3{kThreadX, kThreadY}, dstPtr, srcPtr);
}

template <typename T, int tRows, int tCols>
__global__ AICORE void runSimtLds32(__gm__ T __out__* out, __gm__ T __in__* src, int vRows, int vCols)
{
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;

    constexpr int tileH = tRows;
    constexpr int tileW = tCols;
    constexpr unsigned tileBytes = static_cast<unsigned>(tileH) * static_cast<unsigned>(tileW) * sizeof(T);
    static_assert(tileH * tileW == kThreadCount * kElemsPerThread, "tile must cover every thread element");
    static_assert(kThreadCount <= 2048, "at most 2048 threads");
    static_assert(kThreadCount >= 32 && kThreadCount % 32 == 0, "full warps");
    static_assert((64 * 128) % kThreadCount == 0, "tile divides across threads");
    static_assert(tileBytes * 2 <= 256 * 1024, "UB overflow");

    using TileData = Tile<TileType::Vec, T, tileH, tileW, BLayout::RowMajor, -1, -1>;
    TileData srcTile(vRows, vCols);
    TileData dstTile(vRows, vCols);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, tileBytes);

    GlobalData srcGlobal(src, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(1, 1, 1, tCols, 1));
    GlobalData dstGlobal(out, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(1, 1, 1, tCols, 1));

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    SimtLds32<TileData>(dstTile.data(), srcTile.data());

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int tRows, int tCols>
void LaunchSimtLds32(T* out, T* src, void* stream)
{
    runSimtLds32<T, tRows, tCols><<<1, nullptr, stream>>>(out, src, tRows, tCols);
}

template void LaunchSimtLds32<float, 64, 128>(float* out, float* src, void* stream);
