#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/simd_inst.h>
__simd_vf__ inline void main_kernel_simd_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  vector_f32 acc[1];
  vector_bool mask = asc_create_mask_b32(PAT_ALL);
  for (int32_t chunk = 0; chunk < 2; ++chunk) {
    vector_bool v = asc_create_mask_b32(PAT_ALL);
    acc[0] = simd_inst::vdup<float>(float(0x0p+0f/*0.000000e+00*/), v, MODE_ZEROING);
    for (int32_t k = 0; k < 64; ++k) {
      auto loaded = simd_inst::vlds_unpack<half>((__ubuf__ half*)(&(((__ubuf__ half*)buf_dyn_shmem)[((k * 128) + (chunk * 64))])), 0);
      vector_bool v_1 = asc_create_mask_b16(PAT_ALL);
      auto values = simd_inst::vcvt<float>(loaded, v_1, PART_EVEN, MODE_ZEROING);
      auto scale16 = simd_inst::vlds_brc_elem<half>((__ubuf__ half*)(&(((__ubuf__ half*)buf_dyn_shmem)[(k + 8448)])), 0);
      auto scale = simd_inst::vcvt<float>(scale16, v_1, PART_EVEN, MODE_ZEROING);
      simd_inst::vmula((&(acc[0])), values, scale, mask, MODE_ZEROING);
    }
    simd_inst::vsts_norm(acc[0], (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((chunk * 64) + 4096)])), 0, v);
  }
}

extern "C" __global__ __vector__ void main_kernel(__gm__ half* A, __gm__ float* Out, __gm__ half* X) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[8448]))), (__gm__ uint8_t*)((&(X[0]))), 1, 128, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 128, 128);
  for (int32_t row = 0; row < 32; ++row) {
    for (int32_t k = 0; k < 64; ++k) {
      if (k == 0) {
        asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
      }
      asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[(k * 128)]))), (__gm__ uint8_t*)((&(A[((k * 4096) + (row * 128))]))), 1, 256, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 256, 256);
      if (k == 63) {
        asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
      }
    }
    asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
    asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
    main_kernel_simd_vf_0(buf_dyn_shmem);
    asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
    asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
    asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
    asc_copy_ub2gm_align((__gm__ uint8_t*)((&(Out[(row * 128)]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[4096]))), 1, 512, static_cast<asc_store_l2_cache_mode>(4), 512, 512);
    asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
  }
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
}

