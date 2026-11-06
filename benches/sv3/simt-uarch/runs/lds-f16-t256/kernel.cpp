#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(256) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  float acc[1];
  acc[0] = float(0x0p+0f/*0.000000e+00*/);
  for (int32_t _rep = 0; _rep < 4; ++_rep) {
    for (int32_t step = 0; step < 128; ++step) {
      acc[0] = (acc[0] + ((float)((__ubuf__ half*)buf_dyn_shmem)[((step * 256) + ((int32_t)threadIdx.x))]));
    }
  }
  ((__ubuf__ float*)buf_dyn_shmem)[(((int32_t)threadIdx.x) + 16384)] = acc[0];
}

extern "C" __global__ __vector__ void main_kernel(__gm__ float* Dst, __gm__ half* Src) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(Src[0]))), 1, 65536, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 65536, 65536);
  asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(256), buf_dyn_shmem);
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(Dst[0]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[16384]))), 1, 1024, static_cast<asc_store_l2_cache_mode>(4), 1024, 1024);
}

