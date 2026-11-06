#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(256) inline void main_kernel_simt_vf_0(__ubuf__ float* out) {
  for (int32_t step = 0; step < 128; ++step) {
    out[((step * 256) + ((int32_t)threadIdx.x))] = float(0x1p+0f/*1.000000e+00*/);
  }
}

extern "C" __global__ __vector__ void main_kernel(__gm__ float* Dst, __gm__ float* Src) {
  asc_init();
  __ubuf__ float *out = (__ubuf__ float *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(out[0]))), (__gm__ uint8_t*)((&(Src[0]))), 1, 131072, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 131072, 131072);
  asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(256), out);
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(Dst[0]))), (__ubuf__ uint8_t*)((&(out[0]))), 1, 131072, static_cast<asc_store_l2_cache_mode>(4), 131072, 131072);
}

