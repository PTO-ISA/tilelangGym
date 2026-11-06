#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(1024) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  float held[4];
  float inv[4];
  for (int32_t step = 0; step < 4; ++step) {
    #pragma unroll
    for (int32_t k = 0; k < 4; ++k) {
      held[k] = ((__ubuf__ float*)buf_dyn_shmem)[(((step * 4096) + (k * 1024)) + ((int32_t)threadIdx.x))];
    }
    #pragma unroll
    for (int32_t k_1 = 0; k_1 < 4; ++k_1) {
      float x = held[k_1];
      float m = asc_reduce_max(max(x, (x * float(-0x1p+0f/*-1.000000e+00*/))));
      inv[k_1] = (float(0x1p+0f/*1.000000e+00*/) / max(m, float(0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/)));
    }
    #pragma unroll
    for (int32_t k_2 = 0; k_2 < 4; ++k_2) {
      ((__ubuf__ float*)buf_dyn_shmem)[((((step * 4096) + (k_2 * 1024)) + ((int32_t)threadIdx.x)) + 16384)] = (held[k_2] * inv[k_2]);
    }
  }
}

extern "C" __global__ __vector__ void main_kernel(__gm__ float* Out, __gm__ float* X) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(X[0]))), 1, 65536, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 65536, 65536);
  asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(1024), buf_dyn_shmem);
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(Out[0]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[16384]))), 1, 65536, static_cast<asc_store_l2_cache_mode>(4), 65536, 65536);
}

