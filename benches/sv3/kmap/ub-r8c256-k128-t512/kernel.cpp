#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(512) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, int32_t s) {
  float acc[1];
  float xk = float(0x0p+0f/*0.000000e+00*/);
  acc[0] = float(0x0p+0f/*0.000000e+00*/);
  for (int32_t k = 0; k < 128; ++k) {
    xk = ((float)((__ubuf__ half*)buf_dyn_shmem)[(k + 69632)]);
    acc[0] = (acc[0] + (((float)((__ubuf__ half*)buf_dyn_shmem)[((k * 512) + ((int32_t)threadIdx.x))]) * xk));
  }
  asc_syncthreads();
  ((__ubuf__ float*)buf_dyn_shmem)[(((s * 512) + ((int32_t)threadIdx.x)) + 32768)] = acc[0];
}

extern "C" __global__ __vector__ void main_kernel(__gm__ half* A, __gm__ float* Out, __gm__ half* X) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[69632]))), (__gm__ uint8_t*)((&(X[0]))), 1, 256, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 256, 256);
  for (int32_t s = 0; s < 4; ++s) {
    for (int32_t r = 0; r < 2; ++r) {
      for (int32_t k = 0; k < 128; ++k) {
        if ((r == 0) && (k == 0)) {
          asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
        }
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[((k * 512) + (r * 256))]))), (__gm__ uint8_t*)((&(A[(((k * 2048) + (s * 512)) + (r * 256))]))), 1, 512, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 512, 512);
        if ((r == 1) && (k == 127)) {
          asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
        }
      }
    }
    asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
    asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(512), buf_dyn_shmem, s);
    asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
    if (s == 3) {
      asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
    }
  }
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(Out[0]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[32768]))), 1, 8192, static_cast<asc_store_l2_cache_mode>(4), 8192, 8192);
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
}

