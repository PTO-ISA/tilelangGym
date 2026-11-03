#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(32) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  float scores[64];
  int32_t idxs[64];
  int32_t idx_cand[64];
  float amax[1];
  int32_t best[1];
  #pragma unroll
  for (int32_t i = 0; i < 32; ++i) {
    *(float2*)(scores + (i * 2)) = *(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + ((i * 64) + (((int32_t)threadIdx.x) * 2)));
    *(int2*)(idxs + (i * 2)) = make_int2((((i * 64) + (((int32_t)threadIdx.x) * 2)))+(1*0), (((i * 64) + (((int32_t)threadIdx.x) * 2)))+(1*1));
  }
  for (int32_t k = 0; k < 4; ++k) {
    amax[0] = tl::limits::kFloatNInf;
    #pragma unroll
    for (int32_t rv = 0; rv < 64; ++rv) {
      amax[0] = max(amax[0], scores[(((rv & 31) * 2) + (rv >> 5))]);
    }
    amax[0] = tl::AscendAllReduce<tl::MaxOp, 32, 1, 0>::run(amax[0]);
    #pragma unroll
    for (int32_t i_1 = 0; i_1 < 64; ++i_1) {
      if (scores[i_1] == amax[0]) {
        idx_cand[i_1] = idxs[i_1];
      } else {
        idx_cand[i_1] = 2147483647;
      }
    }
    best[0] = 2147483647;
    #pragma unroll
    for (int32_t rv_1 = 0; rv_1 < 64; ++rv_1) {
      best[0] = min(best[0], idx_cand[(((rv_1 & 31) * 2) + (rv_1 >> 5))]);
    }
    best[0] = tl::AscendAllReduce<tl::MinOp, 32, 1, 0>::run(best[0]);
    if (((int32_t)threadIdx.x) == 0) {
      ((__ubuf__ int32_t*)buf_dyn_shmem)[(k + 2048)] = best[0];
    }
    #pragma unroll
    for (int32_t i_2 = 0; i_2 < 64; ++i_2) {
      if (idxs[i_2] == best[0]) {
        scores[i_2] = float(-0x1.fffff966ad924p+127f/*-3.402823e+38*/);
      }
    }
  }
}

extern "C" __global__ __vector__ void main_kernel(__gm__ float* A, __gm__ int32_t* IdxOut) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(A[0]))), 1, 8192, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 8192, 8192);
  asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(32), buf_dyn_shmem);
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(IdxOut[0]))), (__ubuf__ uint8_t*)((&(((__ubuf__ int32_t*)buf_dyn_shmem)[2048]))), 1, 16, static_cast<asc_store_l2_cache_mode>(4), 16, 16);
}

