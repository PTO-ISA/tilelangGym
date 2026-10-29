#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(1024) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  float v = float(0x0p+0f/*0.000000e+00*/);
  float v_1 = float(0x0p+0f/*0.000000e+00*/);
  float v_2 = float(0x0p+0f/*0.000000e+00*/);
  float v_3 = float(0x0p+0f/*0.000000e+00*/);
  float acc[1];
  float m = float(0x0p+0f/*0.000000e+00*/);
  float result[1];
  acc[0] = tl::limits::kFloatNInf;
  m = float(0x0p+0f/*0.000000e+00*/);
  v = ((__ubuf__ float*)buf_dyn_shmem)[((int32_t)threadIdx.x)];
  float vv = v;
  m = max(m, max(vv, (vv * float(-0x1p+0f/*-1.000000e+00*/))));
  v_1 = ((__ubuf__ float*)buf_dyn_shmem)[(((int32_t)threadIdx.x) + 1024)];
  float vv_1 = v_1;
  m = max(m, max(vv_1, (vv_1 * float(-0x1p+0f/*-1.000000e+00*/))));
  v_2 = ((__ubuf__ float*)buf_dyn_shmem)[(((int32_t)threadIdx.x) + 2048)];
  float vv_2 = v_2;
  m = max(m, max(vv_2, (vv_2 * float(-0x1p+0f/*-1.000000e+00*/))));
  v_3 = ((__ubuf__ float*)buf_dyn_shmem)[(((int32_t)threadIdx.x) + 3072)];
  float vv_3 = v_3;
  m = max(m, max(vv_3, (vv_3 * float(-0x1p+0f/*-1.000000e+00*/))));
  acc[0] = max(acc[0], m);
  acc[0] = tl::AscendAllReduce<tl::MaxOp, 1024, 1, 0>::run(acc[0], (&(((__ubuf__ float*)buf_dyn_shmem)[8192])));
  result[0] = acc[0];
  float scale = max(result[0], float(0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  ((__ubuf__ float*)buf_dyn_shmem)[(((int32_t)threadIdx.x) + 4096)] = (v / scale);
  ((__ubuf__ float*)buf_dyn_shmem)[(((int32_t)threadIdx.x) + 5120)] = (v_1 / scale);
  ((__ubuf__ float*)buf_dyn_shmem)[(((int32_t)threadIdx.x) + 6144)] = (v_2 / scale);
  ((__ubuf__ float*)buf_dyn_shmem)[(((int32_t)threadIdx.x) + 7168)] = (v_3 / scale);
}

extern "C" __global__ __vector__ void main_kernel(__gm__ float* Out, __gm__ float* X) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(X[0]))), 1, 16384, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 16384, 16384);
  asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(1024), buf_dyn_shmem);
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(Out[0]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[4096]))), 1, 16384, static_cast<asc_store_l2_cache_mode>(4), 16384, 16384);
}

