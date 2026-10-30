#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(256) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  float acc[16];
  float xk = float(0x0p+0f/*0.000000e+00*/);
  #pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    float broadcast_var = float(0x0p+0f/*0.000000e+00*/);
    *(float2*)(acc + (i * 2)) = make_float2(broadcast_var, broadcast_var);
  }
  for (int32_t k = 0; k < 8; ++k) {
    xk = ((float)((__ubuf__ half*)buf_dyn_shmem)[(k + 40960)]);
    #pragma unroll
    for (int32_t i_1 = 0; i_1 < 8; ++i_1) {
      half a_ub_local_cast[2];
      *(uint1*)(a_ub_local_cast + 0) = *(__ubuf__ uint1*)(((__ubuf__ half*)buf_dyn_shmem) + (((k * 4096) + (i_1 * 512)) + (((int32_t)threadIdx.x) * 2)));
      float2 _;
        uint1 v_ = *(uint1*)(a_ub_local_cast + 0);
        ((float2*)(&_))[0] = __half22float2(((half2*)(&v_))[0]);
      *(float2*)(acc + (i_1 * 2)) = (*(float2*)(acc + (i_1 * 2)) + (_ * make_float2(xk, xk)));
    }
  }
  #pragma unroll
  for (int32_t i_2 = 0; i_2 < 8; ++i_2) {
    *(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + (((i_2 * 512) + (((int32_t)threadIdx.x) * 2)) + 16384)) = *(float2*)(acc + (i_2 * 2));
  }
}

extern "C" __global__ __vector__ void main_kernel(__gm__ half* A, __gm__ float* Out, __gm__ half* X) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[40960]))), (__gm__ uint8_t*)((&(X[0]))), 1, 16, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 16, 16);
  for (int32_t r = 0; r < 32; ++r) {
    for (int32_t k = 0; k < 8; ++k) {
      asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[((k * 4096) + (r * 128))]))), (__gm__ uint8_t*)((&(A[((k * 4096) + (r * 128))]))), 1, 256, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 256, 256);
      if ((r == 31) && (k == 7)) {
        asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
      }
    }
  }
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(256), buf_dyn_shmem);
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(Out[0]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[16384]))), 1, 16384, static_cast<asc_store_l2_cache_mode>(4), 16384, 16384);
}

