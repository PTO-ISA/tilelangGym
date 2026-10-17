#!/usr/bin/env python3
"""SV1d tiled SIMD twin of Simt sv1tt (short-live stream).

Tag: sv1dt_e{E}_tile{TILE}_vl64

Per tile: MTE load A/B tile -> VL=64 vload/vadd/vstore -> MTE store C tile.
Fixed E=16384 with TILE in {2048,4096,8192,16384} for aligned §3 vs Simt §4.1.

This module is a thin CLI that materializes a static generated kernel
(from _sv1dt_tiled_template.py) so @pto.jit sees module-level annotations.
"""
from __future__ import annotations

import argparse
import runpy
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TPL = HERE / "_sv1dt_tiled_template.py"
if not TPL.is_file():
    # allow OUT/kernels layout where template sits beside this file
    TPL = Path("/tmp/sv1d_aligned_e16k_tile8k_20261017/kernels/_sv1dt_tiled_template.py")


def materialize(E: int, TILE: int) -> Path:
    assert E % TILE == 0
    assert TILE % 64 == 0
    tag = f"sv1dt_e{E}_tile{TILE}_vl64"
    text = TPL.read_text(encoding="utf-8")
    text = (
        text.replace("__E__", str(E))
        .replace("__TILE__", str(TILE))
        .replace("__TAG__", tag)
    )
    out_dir = Path(tempfile.mkdtemp(prefix=f"{tag}_"))
    out = out_dir / f"{tag}.py"
    out.write_text(text, encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("E", nargs="?", type=int, default=16384)
    ap.add_argument("TILE", nargs="?", type=int, default=8192)
    args = ap.parse_args(argv)
    py = materialize(args.E, args.TILE)
    # Execute generated module as __main__
    sys.argv = [str(py)]
    runpy.run_path(str(py), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
