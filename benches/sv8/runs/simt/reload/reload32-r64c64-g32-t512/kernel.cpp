#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(512) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  float m = float(0x0p+0f/*0.000000e+00*/);
  m = float(0x0p+0f/*0.000000e+00*/);
  if (((int32_t)threadIdx.x) < 128) {
    #pragma unroll
    for (int32_t t = 0; t < 32; ++t) {
      float v = ((__ubuf__ float*)buf_dyn_shmem)[((((int32_t)threadIdx.x) * 32) + t)];
      m = max(m, max(v, (v * float(-0x1p+0f/*-1.000000e+00*/))));
    }
    float inv = (float(0x1p+0f/*1.000000e+00*/) / max(m, float(0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/)));
    #pragma unroll
    for (int32_t t_1 = 0; t_1 < 32; ++t_1) {
      float x = ((__ubuf__ float*)buf_dyn_shmem)[((((int32_t)threadIdx.x) * 32) + t_1)];
      ((__ubuf__ float*)buf_dyn_shmem)[(((((int32_t)threadIdx.x) * 32) + t_1) + 4096)] = (x * inv);
    }
  }
}

extern "C" __global__ __vector__ void main_kernel(__gm__ float* Out, __gm__ float* X) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(X[0]))), 1, 16384, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 16384, 16384);
  asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(512), buf_dyn_shmem);
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(Out[0]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[4096]))), 1, 16384, static_cast<asc_store_l2_cache_mode>(4), 16384, 16384);
}

