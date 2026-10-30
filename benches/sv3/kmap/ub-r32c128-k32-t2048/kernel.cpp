#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(2048) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, int32_t s) {
  float acc[1];
  float xk = float(0x0p+0f/*0.000000e+00*/);
  acc[0] = float(0x0p+0f/*0.000000e+00*/);
  for (int32_t k = 0; k < 32; ++k) {
    xk = ((float)((__ubuf__ half*)buf_dyn_shmem)[(k + 73728)]);
    acc[0] = (acc[0] + (((float)((__ubuf__ half*)buf_dyn_shmem)[((k * 2048) + ((int32_t)threadIdx.x))]) * xk));
  }
  asc_syncthreads();
  ((__ubuf__ float*)buf_dyn_shmem)[(((s * 2048) + ((int32_t)threadIdx.x)) + 32768)] = acc[0];
}

extern "C" __global__ __vector__ void main_kernel(__gm__ half* A, __gm__ float* Out, __gm__ half* X) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[73728]))), (__gm__ uint8_t*)((&(X[0]))), 1, 64, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 64, 64);
  for (int32_t s = 0; s < 2; ++s) {
    for (int32_t r = 0; r < 16; ++r) {
      for (int32_t k = 0; k < 32; ++k) {
        if ((r == 0) && (k == 0)) {
          asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
        }
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[((k * 2048) + (r * 128))]))), (__gm__ uint8_t*)((&(A[(((k * 4096) + (s * 2048)) + (r * 128))]))), 1, 256, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 256, 256);
        if ((r == 15) && (k == 31)) {
          asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
        }
      }
    }
    asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
    asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(2048), buf_dyn_shmem, s);
    asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
    if (s == 1) {
      asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
    }
  }
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(Out[0]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[32768]))), 1, 16384, static_cast<asc_store_l2_cache_mode>(4), 16384, 16384);
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
}

