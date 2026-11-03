"""SIMD keep for one row of E scores. VL is 64. Scores and indices stay live.

Usage: simd_keep.py E [K]
K defaults to 8. The gold matches launch.py: seed 2026, one forced tie.
"""

import sys

import numpy as np
from ptodsl import pto

VL = 64
NEG = float(np.float32(-3.402823e38))
INT_MAX = 2147483647

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


def _resident(E: int, k: int):
    """Every chunk stays in a vector register. Fits at 256 and 512."""
    n = E // VL
    scores_bytes = E * 4
    out_bytes = k * 4

    @pto.jit(
        name=f"SV9d_e{E}_k{k}_keep",
        target="a5",
        backend="vpto",
        mode="explicit",
        kernel_kind="vector",
        insert_sync=False,
        ast_rewrite=False,
    )
    def body(out_gm: pto.ptr(pto.i32, "gm"), scores_gm: pto.ptr(pto.f32, "gm")):
        scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        out_ub = pto.castptr(pto.const(scores_bytes, dtype=pto.i64), pto.ptr(pto.i32, "ub"))
        pto.mte_load(scores_gm, scores_ub, 0, scores_bytes, nburst=(1, scores_bytes, scores_bytes))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        mask = pto.vmi.create_mask(VL, size=VL)
        neg = pto.vmi.vbrc(pto.f32(NEG), size=VL)
        int_max = pto.vmi.vbrc(pto.i32(INT_MAX), size=VL)
        arange = pto.vmi.vci(pto.i32(0), size=VL)
        scores = [pto.vmi.vload(scores_ub, i * VL, size=VL) for i in range(n)]
        idxs = [
            pto.vmi.vadd(pto.vmi.vbrc(pto.i32(i * VL), size=VL), arange, mask) for i in range(n)
        ]

        for rnd in range(k):
            folded = scores[0]
            for i in range(1, n):
                folded = pto.vmi.vmax(folded, scores[i], mask)
            top = pto.vmi.vbrc(pto.vmi.vcmax(folded, mask), size=VL)
            cand = None
            for i in range(n):
                piece = pto.vmi.vsel(pto.vmi.vcmp(scores[i], top, mask, "eq"), idxs[i], int_max)
                cand = piece if cand is None else pto.vmi.vmin(cand, piece, mask)
            win = pto.vmi.vcmin(cand, mask)
            pto.vmi.vstore(win, out_ub, rnd)
            win_b = pto.vmi.vbrc(win, size=VL)
            for i in range(n):
                scores[i] = pto.vmi.vsel(pto.vmi.vcmp(idxs[i], win_b, mask, "eq"), neg, scores[i])

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, out_bytes, nburst=(1, out_bytes, out_bytes))

    return body


def _looped(E: int, k: int):
    """One chunk in registers at a time. The row stays in UB, and the kill is stored back."""
    n = E // VL
    scores_bytes = E * 4
    out_bytes = k * 4

    @pto.jit(
        name=f"SV9d_e{E}_k{k}_loop",
        target="a5",
        backend="vpto",
        mode="explicit",
        kernel_kind="vector",
        insert_sync=False,
        ast_rewrite=False,
    )
    def body(out_gm: pto.ptr(pto.i32, "gm"), scores_gm: pto.ptr(pto.f32, "gm")):
        scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        out_ub = pto.castptr(pto.const(scores_bytes, dtype=pto.i64), pto.ptr(pto.i32, "ub"))
        pto.mte_load(scores_gm, scores_ub, 0, scores_bytes, nburst=(1, scores_bytes, scores_bytes))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        mask = pto.vmi.create_mask(VL, size=VL)
        neg = pto.vmi.vbrc(pto.f32(NEG), size=VL)
        int_max = pto.vmi.vbrc(pto.i32(INT_MAX), size=VL)

        for rnd in range(k):
            folded0 = pto.vmi.vload(scores_ub, 0, size=VL)
            fold = pto.for_(1, n, step=1).carry(acc=folded0)
            with fold:
                nxt = pto.vmi.vload(scores_ub, fold.iv * VL, size=VL)
                fold.update(acc=pto.vmi.vmax(fold.acc, nxt, mask))
            top = pto.vmi.vbrc(pto.vmi.vcmax(fold.final("acc"), mask), size=VL)

            chunk0 = pto.vmi.vload(scores_ub, 0, size=VL)
            idx0 = pto.vmi.vci(pto.i32(0), size=VL)
            cand0 = pto.vmi.vsel(pto.vmi.vcmp(chunk0, top, mask, "eq"), idx0, int_max)
            pick = pto.for_(1, n, step=1).carry(best=cand0)
            with pick:
                chunk = pto.vmi.vload(scores_ub, pick.iv * VL, size=VL)
                idx = pto.vmi.vci(pick.iv * VL, size=VL)
                piece = pto.vmi.vsel(pto.vmi.vcmp(chunk, top, mask, "eq"), idx, int_max)
                pick.update(best=pto.vmi.vmin(pick.best, piece, mask))
            win = pto.vmi.vcmin(pick.final("best"), mask)
            pto.vmi.vstore(win, out_ub, rnd)
            win_b = pto.vmi.vbrc(win, size=VL)

            with pto.for_(0, n, step=1) as i:
                chunk = pto.vmi.vload(scores_ub, i * VL, size=VL)
                idx = pto.vmi.vci(i * VL, size=VL)
                updated = pto.vmi.vsel(pto.vmi.vcmp(idx, win_b, mask, "eq"), neg, chunk)
                pto.vmi.vstore(updated, scores_ub, i * VL, mask)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, out_bytes, nburst=(1, out_bytes, out_bytes))

    return body


def kernel(E: int, k: int):
    # 512 scores is 8 chunks of scores and 8 of indices. 1024 does not fit that way.
    if E <= 512:
        return _resident(E, k)
    return _looped(E, k)


def gold(scores, k):
    work = scores.astype(np.float32).copy()
    out = np.empty(k, np.int32)
    neg = np.float32(NEG)
    for i in range(k):
        m = np.max(work)
        idx = int(np.flatnonzero(work == m)[0])
        out[i] = idx
        work[idx] = neg
    return out


def main():
    import ctypes

    E = int(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if E % VL != 0:
        raise SystemExit(f"E={E} is not a multiple of {VL}")
    rng = np.random.RandomState(2026)
    scores = rng.uniform(-1, 1, (E,)).astype(np.float32)
    scores[3] = scores[1]
    expect = gold(scores, k)
    lib = ctypes.CDLL("libascendcl.so")
    for name, arg in [
        ("aclInit", ctypes.c_char_p),
        ("aclFinalize", None),
        ("aclrtSetDevice", ctypes.c_int),
        ("aclrtResetDevice", ctypes.c_int),
    ]:
        getattr(lib, name).argtypes = [arg] if arg else []
        getattr(lib, name).restype = ctypes.c_int
    lib.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtCreateStream.restype = ctypes.c_int
    lib.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
    lib.aclrtDestroyStream.restype = ctypes.c_int
    lib.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
    lib.aclrtMalloc.restype = ctypes.c_int
    lib.aclrtFree.argtypes = [ctypes.c_void_p]
    lib.aclrtFree.restype = ctypes.c_int
    lib.aclrtMemcpy.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    ]
    lib.aclrtMemcpy.restype = ctypes.c_int
    lib.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
    lib.aclrtSynchronizeStream.restype = ctypes.c_int

    def chk(rc, what):
        if rc != 0:
            raise RuntimeError(f"{what} failed {rc}")

    chk(lib.aclInit(None), "init")
    chk(lib.aclrtSetDevice(0), "dev")
    st, sd, od = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")
    sb, ib = scores.nbytes, k * 4
    chk(lib.aclrtMalloc(ctypes.byref(sd), sb, _ACL_MEM_MALLOC_HUGE_FIRST), "ms")
    chk(lib.aclrtMalloc(ctypes.byref(od), ib, _ACL_MEM_MALLOC_HUGE_FIRST), "mo")
    chk(
        lib.aclrtMemcpy(sd, sb, np.ascontiguousarray(scores).ctypes.data_as(ctypes.c_void_p), sb, _ACL_MEMCPY_HOST_TO_DEVICE),
        "h2d",
    )
    kernel(E, k).compile()[1, st](int(od.value), int(sd.value))
    chk(lib.aclrtSynchronizeStream(st), "sync")
    got = np.empty(k, np.int32)
    chk(lib.aclrtMemcpy(got.ctypes.data_as(ctypes.c_void_p), ib, od, ib, _ACL_MEMCPY_DEVICE_TO_HOST), "d2h")
    ok = bool(np.array_equal(got, expect))
    print(f"PASS idx={got.tolist()}" if ok else f"FAIL got={got.tolist()} gold={expect.tolist()}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
