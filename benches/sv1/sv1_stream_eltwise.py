#!/usr/bin/env python3
"""ST-SV1 — Stream Parallel eltwise baseline (no loop-carry).

Single T.Parallel TileLang kernel. Mode / compile target come from
``ST_VF_MODE`` via ``vf_region`` / ``common_asc_harness`` — not a duplicate
``sv1v_*`` twin. Handwritten PTO-DSL twin remains ``sv1d_stream_eltwise.py``.

Matrix: E×T (Simt threads; T ignored for SimdVF frame).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402
from vf_mode import resolve_vf_mode, vf_region  # noqa: E402


def build_sv1(E: int, threads: int):
    import tilelang.language as T

    mode = resolve_vf_mode()
    use_shared_temps = mode.startswith("simd")

    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        B: T.Tensor((E,), "float32"),
        C: T.Tensor((E,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            b_ub = T.alloc_shared((E,), "float32")
            c_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with vf_region(threads=threads):
                if use_shared_temps:
                    t1 = T.alloc_shared((E,), "float32")
                    t2 = T.alloc_shared((E,), "float32")
                    t3 = T.alloc_shared((E,), "float32")
                else:
                    t1 = T.alloc_fragment((E,), "float32")
                    t2 = T.alloc_fragment((E,), "float32")
                    t3 = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    t1[i] = a_ub[i]
                    t2[i] = b_ub[i]
                    t3[i] = t1[i] + t2[i]
                    c_ub[i] = t3[i]
            T.copy(c_ub, C)

    return main


def main():
    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    E = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    # Tag is shape-only; ST_VF_MODE selects frame/target (logged by harness).
    tag = f"sv1_e{E}_t{threads}"
    so = compile_prim(build_sv1(E, threads), tag)
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
