#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(256) inline void main_kernel_simt_vf_0(__gm__ half* A, __gm__ half* X, __ubuf__ float* out_ub) {
  float acc = float(0x0p+0f/*0.000000e+00*/);
  float total = float(0x0p+0f/*0.000000e+00*/);
  for (int32_t c = 0; c < 512; ++c) {
    acc = float(0x0p+0f/*0.000000e+00*/);
    for (int32_t t = 0; t < 2; ++t) {
      acc = (acc + (((float)A[((((t * 131072) + ((((int32_t)threadIdx.x) & 31) * 4096)) + ((((int32_t)threadIdx.x) >> 5) * 512)) + c)]) * ((float)X[((t * 32) + (((int32_t)threadIdx.x) & 31))])));
    }
    total = asc_reduce_add(acc);
    if ((((int32_t)threadIdx.x) % 32) == 0) {
      out_ub[(((((int32_t)threadIdx.x) >> 5) * 512) + c)] = total;
    }
  }
}

extern "C" __global__ __vector__ void main_kernel(__gm__ half* A, __gm__ float* Out, __gm__ half* X) {
  asc_init();
  __ubuf__ float *out_ub = (__ubuf__ float *)0;
  asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(256), A, X, out_ub);
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(Out[0]))), (__ubuf__ uint8_t*)((&(out_ub[0]))), 1, 16384, static_cast<asc_store_l2_cache_mode>(4), 16384, 16384);
}

