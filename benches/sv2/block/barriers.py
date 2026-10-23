#!/usr/bin/env python3
"""SV2 SIMT schedule barriers. 16 rows, 1024 threads.

Stages, in order: load, max, redux, divide, store.
A cut flushes the values that stage produced. An empty memory clobber is
deleted inside this VF, so the cut is a volatile read and write of the same
bits. The fused case loads and maxes each row together, reduces once, then
divides and stores each row.
"""

import sys
from pathlib import Path

R = 16
C = 1024
HERE = Path(__file__).resolve().parent
OUT = HERE / "barriers"

VARIANTS = {
    "fused": None,
    "sep": set(),
    "cut-load": {"load"},
    "cut-max": {"max"},
    "cut-redux": {"redux"},
    "cut-div": {"div"},
    "cut-mem": {"load", "div"},
    "cut-all": {"load", "max", "redux", "div"},
}


def _load():
    return [f"  held[{i}] = X0[{i} * {C} + tid];" for i in range(R)]


def _max():
    lines = ["  m = 0.0f;"]
    for i in range(R):
        lines.append(f"  vv = held[{i}];")
        lines.append("  m = max(m, max(vv, -vv));")
    return lines


def _redux():
    return [
        "  scale = tl::AscendAllReduce<tl::MaxOp, 1024, 1, 0>::run(m, X0 + 32768);",
        "  scale = max(scale, 1.0e-6f);",
    ]


def _div():
    return [f"  prod[{i}] = held[{i}] / scale;" for i in range(R)]


def _store():
    return [f"  O0[{i} * {C} + tid] = prod[{i}];" for i in range(R)]


def _flush(name: str) -> str:
    return f"  {{ volatile float* vp = &{name}; float t = *vp; *vp = t; }}"


def _fence(stage: str) -> str:
    if stage == "load":
        names = [f"held[{i}]" for i in range(R)]
    elif stage == "max":
        names = ["m"]
    elif stage == "redux":
        names = ["scale"]
    elif stage == "div":
        names = [f"prod[{i}]" for i in range(R)]
    else:
        raise ValueError(stage)
    return "\n".join(_flush(name) for name in names)


def _fused():
    lines = ["  m = 0.0f;"]
    for i in range(R):
        lines.append(f"  held[{i}] = X0[{i} * {C} + tid];")
        lines.append(f"  vv = held[{i}];")
        lines.append("  m = max(m, max(vv, -vv));")
    lines.extend(_redux())
    for i in range(R):
        lines.append(f"  O0[{i} * {C} + tid] = held[{i}] / scale;")
    return lines


def body(name: str) -> str:
    if name == "fused":
        lines = _fused()
        decls = "  float held[16];\n  float m;\n  float scale;\n  float vv;"
    else:
        cuts = VARIANTS[name]
        stages = [
            ("load", _load()),
            ("max", _max()),
            ("redux", _redux()),
            ("div", _div()),
            ("store", _store()),
        ]
        lines = []
        for stage, stmts in stages:
            lines.extend(stmts)
            if stage in cuts:
                lines.append(_fence(stage))
        decls = "  float held[16];\n  float prod[16];\n  float m;\n  float scale;\n  float vv;"
    head = f"""  const int tid = (int)threadIdx.x;
  __ubuf__ float* X0 = (__ubuf__ float*)buf_dyn_shmem;
  __ubuf__ float* O0 = X0 + {R * C};
{decls}
"""
    return head + "\n".join(lines) + "\n"


def source(name: str) -> str:
    nbytes = R * C * 4
    return f"""#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>

__simt_vf__ __launch_bounds__({C}) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {{
{body(name)}}}

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
    for name in VARIANTS:
        dest = OUT / name
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "kernel.asc").write_text(source(name))
        print(f"wrote {dest / 'kernel.asc'}", flush=True)


def compile_sources() -> None:
    import tilelang  # noqa: F401

    from tilelang.contrib.bisheng import compile_ascend

    src_inc = "/home/onasismelchiorjeyakumar/workspace/tilelang/src"
    for name in VARIANTS:
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
    print("usage: barriers.py write|compile", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
