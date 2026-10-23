#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(128) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  float acc[1];
  float m = float(0x0p+0f/*0.000000e+00*/);
  float result[1];
  acc[0] = tl::limits::kFloatNInf;
  m = float(0x0p+0f/*0.000000e+00*/);
  float vv = ((__ubuf__ float*)buf_dyn_shmem)[((int32_t)threadIdx.x)];
  m = max(m, max(vv, (vv * float(-0x1p+0f/*-1.000000e+00*/))));
  acc[0] = max(acc[0], m);
  acc[0] = tl::AscendAllReduce<tl::MaxOp, 128, 1, 0>::run(acc[0], (&(((__ubuf__ float*)buf_dyn_shmem)[128])));
  result[0] = acc[0];
  if (((int32_t)threadIdx.x) == 0) {
    ((__ubuf__ float*)buf_dyn_shmem)[256] = max(result[0], float(0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  }
}

extern "C" __global__ __vector__ void main_kernel(__gm__ float* Scale, __gm__ float* X) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(X[0]))), 1, 512, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 512, 512);
  asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(128), buf_dyn_shmem);
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(Scale[0]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[256]))), 1, 4, static_cast<asc_store_l2_cache_mode>(4), 4, 4);
}

