#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(256) inline void main_kernel_simt_vf_0(__ubuf__ float* out_ub) {
  #pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    float broadcast_var = float(0x1p+0f/*1.000000e+00*/);
    *(__ubuf__ float2*)(out_ub + ((i * 512) + (((int32_t)threadIdx.x) * 2))) = make_float2(broadcast_var, broadcast_var);
  }
}

extern "C" __global__ __vector__ void main_kernel(__gm__ float* Out) {
  asc_init();
  __ubuf__ float *out_ub = (__ubuf__ float *)0;
  asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(256), out_ub);
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(Out[0]))), (__ubuf__ uint8_t*)((&(out_ub[0]))), 1, 16384, static_cast<asc_store_l2_cache_mode>(4), 16384, 16384);
}

