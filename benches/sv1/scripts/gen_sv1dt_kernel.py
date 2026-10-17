#!/usr/bin/env python3
"""Generate fully-unrolled SV1d tiled SIMD kernel (no Python tile for-loop)."""
from __future__ import annotations
import argparse
from pathlib import Path

HEADER = '''#!/usr/bin/env python3
"""Auto-generated SV1d tiled SIMD — fully unrolled tiles. Do not edit."""
import ctypes
import time
import numpy as np
from ptodsl import pto

E = {E}
TILE = {TILE}
VL = 64
N_TILES = E // TILE
N_CHUNKS = TILE // VL
TILE_BYTES = TILE * 4

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


@pto.jit(
    name="SV1dt_e{E}_tile{TILE}",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def kernel(
    c_gm: pto.ptr(pto.f32, "gm"),
    a_gm: pto.ptr(pto.f32, "gm"),
    b_gm: pto.ptr(pto.f32, "gm"),
):
    a_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    b_ub = pto.castptr(pto.const(TILE_BYTES, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    c_ub = pto.castptr(pto.const(2 * TILE_BYTES, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
'''

TILE_BLOCK = '''
    # ---- tile {t} ----
{barrier}    a_src_{t} = pto.addptr(a_gm, {elem_off})
    b_src_{t} = pto.addptr(b_gm, {elem_off})
    c_dst_{t} = pto.addptr(c_gm, {elem_off})
    pto.mte_load(a_src_{t}, a_ub, 0, TILE_BYTES, nburst=(1, TILE_BYTES, TILE_BYTES))
    pto.mte_load(b_src_{t}, b_ub, 0, TILE_BYTES, nburst=(1, TILE_BYTES, TILE_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)
    mask_{t} = pto.vmi.create_mask(VL, size=VL)
    for i in range(N_CHUNKS):
        off = i * VL
        a = pto.vmi.vload(a_ub, off, size=VL)
        b = pto.vmi.vload(b_ub, off, size=VL)
        c = pto.vmi.vadd(a, b, mask_{t})
        pto.vmi.vstore(c, c_ub, off, mask_{t})
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(c_ub, c_dst_{t}, TILE_BYTES, nburst=(1, TILE_BYTES, TILE_BYTES))
'''

BARRIER = '''    pto.set_flag("MTE3", "MTE2", event_id=0)
    pto.wait_flag("MTE3", "MTE2", event_id=0)
'''

FOOTER = '''

def reference(a, b):
    return (a + b).astype(np.float32)


def _load_acl():
    lib = ctypes.CDLL("libascendcl.so")
    for n, t in [
        ("aclInit", ctypes.c_char_p),
        ("aclFinalize", None),
        ("aclrtSetDevice", ctypes.c_int),
        ("aclrtResetDevice", ctypes.c_int),
    ]:
        getattr(lib, n).argtypes = [t] if t else []
        getattr(lib, n).restype = ctypes.c_int
    lib.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtCreateStream.restype = ctypes.c_int
    lib.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
    lib.aclrtDestroyStream.restype = ctypes.c_int
    lib.aclrtMalloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int
    ]
    lib.aclrtMalloc.restype = ctypes.c_int
    lib.aclrtFree.argtypes = [ctypes.c_void_p]
    lib.aclrtFree.restype = ctypes.c_int
    lib.aclrtMemcpy.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    ]
    lib.aclrtMemcpy.restype = ctypes.c_int
    lib.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
    lib.aclrtSynchronizeStream.restype = ctypes.c_int
    return lib


def _chk(rc, what):
    if rc != 0:
        raise RuntimeError(f"{{what}} failed {{rc}}")


def run_acl(tag: str) -> int:
    rng = np.random.RandomState(2026)
    a = rng.uniform(-3, 3, E).astype(np.float32)
    b = rng.uniform(-3, 3, E).astype(np.float32)
    ref = reference(a, b)
    nbytes = E * 4
    lib = _load_acl()
    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    st = ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")
    ad = ctypes.c_void_p(); bd = ctypes.c_void_p(); cd = ctypes.c_void_p()
    _chk(lib.aclrtMalloc(ctypes.byref(ad), nbytes, _ACL_MEM_MALLOC_HUGE_FIRST), "ma")
    _chk(lib.aclrtMalloc(ctypes.byref(bd), nbytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mb")
    _chk(lib.aclrtMalloc(ctypes.byref(cd), nbytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mc")
    _chk(lib.aclrtMemcpy(ad, nbytes, np.ascontiguousarray(a).ctypes.data_as(ctypes.c_void_p), nbytes, _ACL_MEMCPY_HOST_TO_DEVICE), "h2d_a")
    _chk(lib.aclrtMemcpy(bd, nbytes, np.ascontiguousarray(b).ctypes.data_as(ctypes.c_void_p), nbytes, _ACL_MEMCPY_HOST_TO_DEVICE), "h2d_b")
    t0 = time.perf_counter()
    comp = kernel.compile()
    cs = time.perf_counter() - t0
    t0 = time.perf_counter()
    comp[1, st](int(cd.value), int(ad.value), int(bd.value))
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    ls = time.perf_counter() - t0
    ch = np.zeros(E, dtype=np.float32)
    _chk(lib.aclrtMemcpy(ch.ctypes.data_as(ctypes.c_void_p), nbytes, cd, nbytes, _ACL_MEMCPY_DEVICE_TO_HOST), "d2h")
    lib.aclrtFree(ad); lib.aclrtFree(bd); lib.aclrtFree(cd)
    lib.aclrtDestroyStream(st); lib.aclrtResetDevice(0); lib.aclFinalize()
    maxabs = float(np.max(np.abs(ch - ref)))
    if not np.allclose(ch, ref, atol=1e-4, rtol=1e-4):
        raise AssertionError(f"mismatch max_diff={{maxabs}}")
    print("PASS %s  compile=%.3fs launch=%.3fs maxabs=%s" % (tag, cs, ls, maxabs))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_acl("__TAG__"))
'''


def generate(E: int, TILE: int) -> str:
    assert E % TILE == 0 and TILE % 64 == 0
    n_tiles = E // TILE
    tag = f"sv1dt_e{E}_tile{TILE}_vl64"
    parts = [HEADER.format(E=E, TILE=TILE)]
    for t in range(n_tiles):
        barrier = BARRIER if t > 0 else ""
        parts.append(
            TILE_BLOCK.format(t=t, elem_off=t * TILE, barrier=barrier)
        )
    # FOOTER uses {{ }} for literal braces in f-strings inside the template;
    # format only {tag}
    foot = FOOTER.replace("__TAG__", tag)
    foot = foot.replace("{{", "{").replace("}}", "}")
    parts.append(foot)
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("E", type=int)
    ap.add_argument("TILE", type=int)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    text = generate(args.E, args.TILE)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"WROTE {args.out} tiles={args.E // args.TILE}")


if __name__ == "__main__":
    main()
