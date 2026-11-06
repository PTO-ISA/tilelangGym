#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/simd_inst.h>
__simd_vf__ inline void main_kernel_simd_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  vector_f32 acc[1];
  vector_bool mask = asc_create_mask_b32(PAT_ALL);
  vector_bool v = asc_create_mask_b32(PAT_ALL);
  acc[0] = simd_inst::vdup<float>(float(0x0p+0f/*0.000000e+00*/), v, MODE_ZEROING);
  for (int32_t i = 0; i < 256; ++i) {
    acc[0] = simd_inst::vadd(acc[0], simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(i * 64)])), 0), mask, MODE_ZEROING);
  }
  simd_inst::vsts_norm(acc[0], (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[16384])), 0, mask);
}

extern "C" __global__ __vector__ void main_kernel(__gm__ float* Dst, __gm__ float* Src) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(Src[0]))), 1, 65536, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 65536, 65536);
  asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  main_kernel_simd_vf_0(buf_dyn_shmem);
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(Dst[0]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[16384]))), 1, 256, static_cast<asc_store_l2_cache_mode>(4), 256, 256);
}

