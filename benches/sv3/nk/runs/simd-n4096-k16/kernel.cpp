#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/simd_inst.h>
__simd_vf__ inline void main_kernel_simd_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  vector_f32 prod[1];
  vector_f32 part[1];
  vector_bool mask = asc_create_mask_b32(PAT_ALL);
  vector_bool store_mask = asc_create_mask_b32(PAT_VL4);
  vector_bool v = asc_create_mask_b32(PAT_ALL);
  auto scale = simd_inst::vdup<float>(float(0x0p+0f/*0.000000e+00*/), v, MODE_ZEROING);
  auto scale16 = simd_inst::vlds_unpack<half>((__ubuf__ half*)(&(((__ubuf__ half*)buf_dyn_shmem)[73728])), 0);
  vector_bool v_1 = asc_create_mask_b16(PAT_ALL);
  auto scale_1 = simd_inst::vcvt<float>(scale16, v_1, PART_EVEN, MODE_ZEROING);
  for (int32_t g = 0; g < 1024; ++g) {
    prod[0] = simd_inst::vdup<float>(float(0x0p+0f/*0.000000e+00*/), v, MODE_ZEROING);
    auto loaded = simd_inst::vlds_unpack<half>((__ubuf__ half*)(&(((__ubuf__ half*)buf_dyn_shmem)[(g * 64)])), 0);
    auto values = simd_inst::vcvt<float>(loaded, v_1, PART_EVEN, MODE_ZEROING);
    part[0] = simd_inst::vmul(values, scale_1, mask, MODE_ZEROING);
    #pragma unroll
    for (int32_t _step = 0; _step < 4; ++_step) {
      part[0] = simd_inst::vcpadd(part[0], mask, MODE_ZEROING);
    }
    prod[0] = simd_inst::vadd(prod[0], part[0], mask, MODE_ZEROING);
    simd_inst::vsts_norm(prod[0], (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((g * 4) + 32768)])), 0, store_mask);
  }
}

extern "C" __global__ __vector__ void main_kernel(__gm__ half* A, __gm__ float* Out, __gm__ half* X) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[73792]))), (__gm__ uint8_t*)((&(X[0]))), 1, 32, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 32, 32);
  asc_sync_notify(PIPE_MTE2, PIPE_S, static_cast<event_t>(0));
  for (int32_t i = 0; i < 64; ++i) {
    if (i == 0) {
      asc_sync_wait(PIPE_MTE2, PIPE_S, static_cast<event_t>(0));
    }
    ((__ubuf__ half*)buf_dyn_shmem)[(i + 73728)] = ((__ubuf__ half*)buf_dyn_shmem)[((i & 15) + 73792)];
    if (i == 63) {
      asc_sync_notify(PIPE_S, PIPE_V, static_cast<event_t>(0));
    }
  }
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(A[0]))), 1, 131072, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 131072, 131072);
  asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_S, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  main_kernel_simd_vf_0(buf_dyn_shmem);
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(Out[0]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[32768]))), 1, 16384, static_cast<asc_store_l2_cache_mode>(4), 16384, 16384);
}

