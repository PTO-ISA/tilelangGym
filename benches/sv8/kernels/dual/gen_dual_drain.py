"""Their SoftLib dual_drain motif, with the copy sized to N.

The published generator fixes E at 8192 and a 32768-byte copy. This keeps the
same load-all, abs-all, paired warp max, then multiply, and sizes both copies
to N floats.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
MBAR = 'asm volatile("" ::: "memory");'
HDR = """#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
"""


def source(E: int, T: int) -> str:
    if E % T != 0:
        raise ValueError(f"E={E} must divide by T={T}")
    rounds = E // T
    if rounds % 2 != 0:
        raise ValueError(f"rounds={rounds} must be even")
    nbytes = E * 4
    body = f"""  const int tid = (int32_t)threadIdx.x;
  __ubuf__ float* A0 = (__ubuf__ float*)buf_dyn_shmem;
  __ubuf__ float* C0 = (__ubuf__ float*)buf_dyn_shmem + {E};
  // motif=warp-coop schedule=back_loadall_absall_dual_drain_e2e Case-3-e2e G=32 T={T} axis=1xG KEEP-X
  const int lane = tid & 31;
  const int wid = tid >> 5;
  float a[{rounds}];
  float av[{rounds}];
  float s[{rounds}];
"""
    for r in range(rounds):
        body += f"  a[{r}] = A0[{r}*{T} + tid];\n"
    body += f"  {MBAR}  // wall LD\n"
    for r in range(rounds):
        body += (
            f"  {{ float v = a[{r}]; float x = v; if (v < 0.0f) x = -v; av[{r}] = x; }}\n"
        )
    body += f"  {MBAR}  // wall ABS\n"
    for r in range(0, rounds, 2):
        body += f"""  {{
    float wm0 = asc_reduce_max(av[{r}]);
    float wm1 = asc_reduce_max(av[{r + 1}]);
    float m0 = wm0; if (m0 < 1.0e-6f) m0 = 1.0e-6f;
    float m1 = wm1; if (m1 < 1.0e-6f) m1 = 1.0e-6f;
    s[{r}] = 1.0f / m0;
    s[{r + 1}] = 1.0f / m1;
  }}
"""
    body += f"  {MBAR}  // wall REDUX/recip\n"
    for r in range(rounds):
        body += f"  C0[{r}*{T} + tid] = a[{r}] * s[{r}];\n"
    return HDR + f"""
__simt_vf__ __launch_bounds__({T}) inline void simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {{
{body}}}

extern "C" __global__ __vector__ void main_kernel(__gm__ float* A, __gm__ float* B, __gm__ float* C) {{
  AscendC::InitSocState();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  copy_gm_to_ubuf_align_v2((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(A[0]))), 0, 1, {nbytes}, 0, 0, 0, 0, {nbytes}, {nbytes});
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
  asc_vf_call<simt_vf_0>(cce::dim3({T}), buf_dyn_shmem);
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
  copy_ubuf_to_gm_align_v2((__gm__ void*)((&(C[0]))), (__ubuf__ void*)((&(((__ubuf__ float*)buf_dyn_shmem)[{E}]))), 0, 1, {nbytes}, 4, {nbytes}, {nbytes});
}}
#ifdef _WIN32
#define TL_EXPORT __declspec(dllexport)
#else
#define TL_EXPORT
#endif
#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];
extern "C" TL_EXPORT const char* get_last_error() {{ return error_buf; }}
extern "C" TL_EXPORT int init() {{ error_buf[0] = '\\0'; return 0; }}
extern "C" TL_EXPORT int call(void * A, void * B, void * C, void *stream) {{
  main_kernel<<<(1 * 1 * 1), 16384, stream>>>((__gm__ float *)A, (__gm__ float *)B, (__gm__ float *)C);
  return 0;
}}
"""


# One file per dual-drain cell in the tables. E is rows × cols.
PAIRS = (
    (4096, 128), (4096, 512), (4096, 1024), (4096, 2048),
    (8192, 256), (8192, 512), (8192, 1024), (8192, 2048),
    (16384, 512), (16384, 1024), (16384, 2048),
)


def main():
    for E, T in PAIRS:
        path = HERE / f"dual_drain_t{T}_e{E}.asc"
        path.write_text(source(E, T))
        print(path)


if __name__ == "__main__":
    main()
