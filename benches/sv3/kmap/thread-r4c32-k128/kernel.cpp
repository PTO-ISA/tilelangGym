#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(128) inline void main_kernel_simt_vf_0(__gm__ half* X, __gm__ half* A, __ubuf__ float* out_ub) {
  float acc[1];
  float xk = float(0x0p+0f/*0.000000e+00*/);
  acc[0] = float(0x0p+0f/*0.000000e+00*/);
  for (int32_t k = 0; k < 128; ++k) {
    xk = ((float)X[k]);
    acc[0] = (acc[0] + (((float)A[((k * 128) + ((int32_t)threadIdx.x))]) * xk));
  }
  out_ub[((int32_t)threadIdx.x)] = acc[0];
}

extern "C" __global__ __vector__ void main_kernel(__gm__ half* A, __gm__ float* Out, __gm__ half* X) {
  asc_init();
  __ubuf__ float *out_ub = (__ubuf__ float *)0;
  asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(128), X, A, out_ub);
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(Out[0]))), (__ubuf__ uint8_t*)((&(out_ub[0]))), 1, 512, static_cast<asc_store_l2_cache_mode>(4), 512, 512);
}

