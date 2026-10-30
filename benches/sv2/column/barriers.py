#!/usr/bin/env python3
"""SV2.A SIMT schedule barriers. 16 rows, 1024 threads.

Stages, in order: load, max, inverse, multiply, store.
A cut pins the values that stage produced, so the compiler has to finish
that stage before the next one starts. An empty memory clobber is not enough:
the compiler deletes it and emits one binary for every fence placement.
The fused case is the stock loop: each row loads and maxes together,
then one inverse, then each row multiplies and stores.
"""

import sys
from pathlib import Path

R = 16
C = 1024
HERE = Path(__file__).resolve().parent
OUT = HERE / "barriers"

# Cuts sit after these stages. "mem" is the two memory cuts. "all" is every cut.
VARIANTS = {
    "fused": None,
    "sep": set(),
    "cut-load": {"load"},
    "cut-max": {"max"},
    "cut-inv": {"inv"},
    "cut-mul": {"mul"},
    "cut-mem": {"load", "mul"},
    "cut-all": {"load", "max", "inv", "mul"},
}


def _load():
    return [f"  held[{i}] = X0[{i} * {C} + tid];" for i in range(R)]


def _max():
    lines = ["  m = 0.0f;"]
    for i in range(R):
        lines.append(f"  vv = held[{i}];")
        lines.append("  m = max(m, max(vv, -vv));")
    return lines


def _inv():
    return ["  inv = 1.0f / max(m, 1.0e-6f);"]


def _mul():
    return [f"  prod[{i}] = held[{i}] * inv;" for i in range(R)]


def _store():
    return [f"  O0[{i} * {C} + tid] = prod[{i}];" for i in range(R)]


def _flush(name: str) -> str:
    return f"  {{ volatile float* vp = &{name}; float t = *vp; *vp = t; }}"


def _fence(stage: str) -> str:
    """Flush the stage's values through a volatile access.

    An empty asm in this VF is deleted, so every fence placement compiled to
    one binary. A simt function cannot call out. The volatile flush is inline,
    and it sits on a different value in each kernel.
    """
    if stage == "load":
        names = [f"held[{i}]" for i in range(R)]
    elif stage == "max":
        names = ["m"]
    elif stage == "inv":
        names = ["inv"]
    elif stage == "mul":
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
    lines.extend(_inv())
    for i in range(R):
        lines.append(f"  O0[{i} * {C} + tid] = held[{i}] * inv;")
    return lines


def body(name: str) -> str:
    if name == "fused":
        lines = _fused()
        decls = "  float held[16];\n  float m;\n  float inv;\n  float vv;"
    else:
        cuts = VARIANTS[name]
        stages = [("load", _load()), ("max", _max()), ("inv", _inv()), ("mul", _mul()), ("store", _store())]
        lines = []
        for stage, stmts in stages:
            lines.extend(stmts)
            if stage in cuts:
                lines.append(_fence(stage))
        decls = "  float held[16];\n  float prod[16];\n  float m;\n  float inv;\n  float vv;"
    head = f"""  const int tid = (int)threadIdx.x;
  __ubuf__ float* X0 = (__ubuf__ float*)buf_dyn_shmem;
  __ubuf__ float* O0 = X0 + {R * C};
{decls}
"""
    return head + "\n".join(lines) + "\n"


def source(name: str) -> str:
    return f"""#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>

__simt_vf__ __launch_bounds__({C}) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {{
{body(name)}}}

extern "C" __global__ __vector__ void main_kernel(__gm__ float* Out, __gm__ float* X) {{
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(X[0]))), 1, {R * C * 4}, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), {R * C * 4}, {R * C * 4});
  asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_vf_call<main_kernel_simt_vf_0>(cce::dim3({C}), buf_dyn_shmem);
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(Out[0]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[{R * C}]))), 1, {R * C * 4}, static_cast<asc_store_l2_cache_mode>(4), {R * C * 4}, {R * C * 4});
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
    import tilelang  # noqa: F401  loads the in-tree TVM before bisheng imports it

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
