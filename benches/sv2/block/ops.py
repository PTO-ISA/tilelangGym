#!/usr/bin/env python3
"""SV2 SIMT operators as separate kernels. 16 rows, 1024 threads.

Same tile as the head-to-head SIMT kernel. The block scale is a max across
threads, and pass 2 divides. There is no multiply.

`load` writes the 16 values back, or the loads are deleted. `max` loads the
16 rows and keeps the column max. `redux` loads one value per thread and
reduces it. `fdiv` loads the 16 rows and divides each by the column's first
value. `sts` writes the constant 1.
"""

import sys
from pathlib import Path

R = 16
C = 1024
HERE = Path(__file__).resolve().parent
OUT = HERE / "ops"
NAMES = ("load", "max", "redux", "fdiv", "sts")


def _tid_head() -> str:
    return f"""  const int tid = (int)threadIdx.x;
  __ubuf__ float* X0 = (__ubuf__ float*)buf_dyn_shmem;
  __ubuf__ float* O0 = X0 + {R * C};
"""


def _load_body() -> str:
    lines = ["  float held[16];"]
    for i in range(R):
        lines.append(f"  held[{i}] = X0[{i} * {C} + tid];")
    for i in range(R):
        lines.append(f"  O0[{i} * {C} + tid] = held[{i}];")
    return "\n".join(lines)


def _max_body() -> str:
    lines = ["  float m = 0.0f;", "  float vv;"]
    for i in range(R):
        lines.append(f"  vv = X0[{i} * {C} + tid];")
        lines.append("  m = max(m, max(vv, -vv));")
    lines.append("  O0[tid] = m;")
    return "\n".join(lines)


def _redux_body() -> str:
    return "\n".join(
        [
            "  float v = X0[tid];",
            "  float m = max(v, -v);",
            "  __ubuf__ float* red = X0 + 32768;",
            "  float scale = tl::AscendAllReduce<tl::MaxOp, 1024, 1, 0>::run(m, red);",
            "  O0[tid] = scale;",
        ]
    )


def _fdiv_body() -> str:
    lines = ["  float held[16];"]
    for i in range(R):
        lines.append(f"  held[{i}] = X0[{i} * {C} + tid];")
    lines.append("  float scale = max(max(held[0], -held[0]), 1.0e-6f);")
    for i in range(R):
        lines.append(f"  O0[{i} * {C} + tid] = held[{i}] / scale;")
    return "\n".join(lines)


def _sts_body() -> str:
    return "\n".join(f"  O0[{i} * {C} + tid] = 1.0f;" for i in range(R))


BODIES = {
    "load": _load_body,
    "max": _max_body,
    "redux": _redux_body,
    "fdiv": _fdiv_body,
    "sts": _sts_body,
}


def source(name: str) -> str:
    nbytes = R * C * 4
    return f"""#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>

__simt_vf__ __launch_bounds__({C}) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {{
{_tid_head()}{BODIES[name]()}
}}

extern "C" __global__ __vector__ void main_kernel(__gm__ float* Out, __gm__ float* X) {{
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(X[0]))), 1, {nbytes}, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), {nbytes}, {nbytes});
  asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_vf_call<main_kernel_simt_vf_0>(cce::dim3({C}), buf_dyn_shmem);
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(Out[0]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[{R * C}]))), 1, {nbytes}, static_cast<asc_store_l2_cache_mode>(4), {nbytes}, {nbytes});
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
extern "C" TL_EXPORT int call(void* X, void* Out, void* stream) {{
  main_kernel<<<1, 0, stream>>>((__gm__ float*)Out, (__gm__ float*)X);
  return 0;
}}
"""


def write_sources() -> None:
    for name in NAMES:
        dest = OUT / name
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "kernel.asc").write_text(source(name))
        print(f"wrote {dest / 'kernel.asc'}", flush=True)


def compile_sources() -> None:
    import tilelang  # noqa: F401

    from tilelang.contrib.bisheng import compile_ascend

    src_inc = "/home/onasismelchiorjeyakumar/workspace/tilelang/src"
    for name in NAMES:
        dest = OUT / name
        code = (dest / "kernel.asc").read_text()
        compile_ascend(
            code,
            target_format="so",
            npu_arch="dav-3510",
            options=["-I" + src_inc],
            path_target=str(dest / "kernel.so"),
            verbose=True,
        )
        print(f"compiled {name}", flush=True)


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args == ["write"]:
        write_sources()
        return 0
    if args == ["compile"]:
        write_sources()
        compile_sources()
        return 0
    print("usage: ops.py write|compile", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
