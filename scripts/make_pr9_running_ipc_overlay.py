#!/usr/bin/env python3
"""Forced recipe: PR9 running-IPC + opcode-lane issue overlay.

See also: REGRESSION_salvage L6_ANALYSIS_INVENTORY.md § "PR9 running-IPC + instr-mix overlay".

Inputs (per arm):
  - popped / instr_log.dump  (preferred)
  - OR chrome trace + instr_exe.csv (via arms.yaml artifacts)
  - OR rolling CSV (IPC-only — NOT PR9; mark MISSING lanes)

Example:
  PYTHONPATH=/workspace/tilelangGym-pr8-tip \
    python scripts/make_pr9_running_ipc_overlay.py \
      --dump /path/instr_log.dump \
      --out-png out/arm_running_ipc_pr9.png \
      --title "sv8 / SoftLib … — running IPC (W=20)" --E 8192

Expected:
  - PNG dual-pane (lanes top, IPC bottom), plot_style=opcode_lanes_ipc
  - JSON sidecar with mid_med_ipc + lane_opcodes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="instr_log.dump / popped_dump")
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--E", type=float, default=None)
    ap.add_argument("--W", type=float, default=20.0)
    ap.add_argument("--repo", default="/workspace/tilelangGym-pr8-tip")
    ap.add_argument(
        "--frontend",
        default="",
        help="STAGE_TAXONOMY frontend (softlib_simt|simd) — annotate defining-opcode stage windows",
    )
    ap.add_argument(
        "--staged-out",
        default="",
        help="optional second PNG with stage bands (*_pr9_staged.png); implies --frontend",
    )
    args = ap.parse_args()

    repo = Path(args.repo)
    sys.path.insert(0, str(repo))
    from scripts.st_report.rolling_ipc import build_from_artifacts

    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    art = {"popped_dump": str(Path(args.dump).resolve())}
    if args.E is not None:
        art["E"] = float(args.E)
    result = build_from_artifacts(
        art,
        st_root=out.parent,
        repo_root=repo,
        W=args.W,
        out_png=out,
        title=args.title,
        E=args.E,
    )
    meta = result.get("meta") or {}
    style = meta.get("plot_style")
    print(
        f"png={out} exists={out.is_file()} style={style} "
        f"mid-med={meta.get('mid_med_ipc')} lanes={meta.get('lane_opcodes')}"
    )
    print(f"json={result.get('json')}")
    if style != "opcode_lanes_ipc":
        print("FAIL: need plot_style=opcode_lanes_ipc (IPC-only is not PR9 overlay)", file=sys.stderr)
        return 1

    frontend = (args.frontend or "").strip()
    staged_out = (args.staged_out or "").strip()
    if staged_out and not frontend:
        # infer from title
        tl = args.title.lower()
        frontend = "softlib_simt" if ("softlib" in tl or "simt" in tl) else "simd"
    if frontend or staged_out:
        art2 = dict(art)
        art2["frontend"] = frontend or "simd"
        out2 = Path(staged_out) if staged_out else out.with_name(out.stem + "_staged" + out.suffix)
        result2 = build_from_artifacts(
            art2,
            st_root=out2.parent,
            repo_root=repo,
            W=args.W,
            out_png=out2,
            title=args.title + " — staged",
            E=args.E,
        )
        meta2 = result2.get("meta") or {}
        print(
            f"staged={out2} exists={out2.is_file()} stages={len(meta2.get('stages') or [])} "
            f"taxonomy={meta2.get('stage_taxonomy')}"
        )
        print(f"staged_json={result2.get('json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
