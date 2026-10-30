"""Aggregate instr_exe.csv opcode call_counts → per-N×4B ISA-mix JSON (t546u).

Family map (SIMT SoftLib / SIMD RV prefixes stripped for matching):
  load     : LDS, VLDI, LD*
  store    : STS, VSTI, ST*
  reduce   : REDUX, VCMAX, VMAX, VAND, FADD (as reduce-fold), …
  scale    : FDIV*, VDIV, VSEL, VDUP, …
  eltwise  : FMUL, VMUL, …
  pred_tax : IADD*, FSETP*, SEL (SIMT predication)

Bridge:
  ops_per_N = call_count / (E / N)
  pred_el/cyc ≈ IPC × (N / ops_per_N)   # prefer body IPC = body_calls / VF_cyc

Orthogonal to STAGE_TAXONOMY (§8.1/§8.2 defining-opcode windows) — do not merge.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

FAMILY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("load", re.compile(r"(LDS|VLDI|VLDS|^LD\b|LOAD)", re.I)),
    ("store", re.compile(r"(STS|VSTI|VSTS|^ST\b|STORE)", re.I)),
    ("reduce", re.compile(r"(REDUX|VCMAX|VCGMAX|VC?MAX\b|VAND|VABS|FADD)", re.I)),
    ("scale", re.compile(r"(FDIV|VDIV|VSEL|VDUP|RECIP)", re.I)),
    ("eltwise", re.compile(r"(FMUL|VMUL|FMA)", re.I)),
    ("pred_tax", re.compile(r"(IADD|FSETP|SETP|^SEL$|SIMT_SEL)", re.I)),
]

FAMILIES = ["load", "store", "reduce", "scale", "eltwise", "pred_tax"]


def _norm_op(name: str) -> str:
    s = (name or "").strip()
    for pref in ("SIMT_", "RV_", "PTO_", "AIC_"):
        if s.upper().startswith(pref):
            s = s[len(pref) :]
            break
    return s


def classify(op: str) -> str | None:
    n = _norm_op(op)
    # SEL alone → pred_tax; VSEL → scale (checked in order: scale before generic SEL)
    for fam, pat in FAMILY_PATTERNS:
        if pat.search(n) or pat.search(op):
            return fam
    return None


def read_call_counts(csv_path: Path) -> Counter[str]:
    """Sum call_count (or count rows) per opcode from instr_exe.csv."""
    counts: Counter[str] = Counter()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = {h.lower(): h for h in (reader.fieldnames or [])}
        op_key = fields.get("opcode") or fields.get("instr") or fields.get("name")
        cc_key = fields.get("call_count") or fields.get("count") or fields.get("calls")
        if not op_key:
            raise SystemExit(f"no opcode column in {csv_path}: {reader.fieldnames}")
        for row in reader:
            op = (row.get(op_key) or "").strip()
            if not op:
                continue
            if cc_key and row.get(cc_key) not in (None, ""):
                try:
                    n = int(float(row[cc_key]))
                except ValueError:
                    n = 1
            else:
                n = 1
            counts[op] += n
    return counts


def family_totals(counts: Counter[str]) -> dict[str, int]:
    out = {f: 0 for f in FAMILIES}
    for op, n in counts.items():
        fam = classify(op)
        if fam:
            out[fam] += n
    return out


def arm_block(
    *,
    arm: str,
    counts: Counter[str],
    E: int,
    N: int,
    VF_cyc: int,
    body_calls: int | None,
    el_cyc: float | None,
    mid_IPC_W20: float | None,
) -> dict[str, Any]:
    units = E / N
    fam = family_totals(counts)
    ops_per_N = {k: (v / units) for k, v in fam.items()}
    listed = sum(ops_per_N.values())
    bc = body_calls if body_calls is not None else sum(counts.values())
    body_ipc = (bc / VF_cyc) if VF_cyc else None
    ops_body = bc / units if units else None
    pred = (body_ipc * (N / ops_body)) if body_ipc and ops_body else None
    return {
        "arm": arm,
        "VF_cyc": VF_cyc,
        "el_cyc": el_cyc,
        "body_calls": bc,
        "body_IPC": body_ipc,
        "mid_IPC_W20": mid_IPC_W20,
        "ops_raw": fam,
        "ops_per_N": ops_per_N,
        "ops_per_N_listed_total": listed,
        "ops_per_N_body_calls": ops_body,
        "pred_el_cyc_body": pred,
        "opcode_detail": dict(counts),
    }


def build_report(
    softlib: dict[str, Any],
    simd: dict[str, Any],
    *,
    E: int,
    N: int,
    era: str = "",
    shape: str = "",
) -> dict[str, Any]:
    pred_ratio = None
    if softlib.get("pred_el_cyc_body") and simd.get("pred_el_cyc_body"):
        pred_ratio = softlib["pred_el_cyc_body"] / simd["pred_el_cyc_body"]
    meas_ratio = None
    if softlib.get("el_cyc") and simd.get("el_cyc"):
        meas_ratio = softlib["el_cyc"] / simd["el_cyc"]
    return {
        "E": E,
        "N": N,
        "units": float(E / N),
        "era": era,
        "shape": shape,
        "IPC_def": "body_calls / VF_cyc",
        "SoftLib": softlib,
        "SIMD": simd,
        "pred_ratio_body": pred_ratio,
        "meas_ratio": meas_ratio,
        "opcode_detail": {
            "SoftLib": softlib.get("opcode_detail") or {},
            "SIMD": simd.get("opcode_detail") or {},
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--softlib-csv", type=Path, required=True)
    p.add_argument("--simd-csv", type=Path, required=True)
    p.add_argument("--E", type=int, default=8192)
    p.add_argument("--N", type=int, default=32, dest="norm_elems")
    p.add_argument("--softlib-vf", type=int, required=True)
    p.add_argument("--simd-vf", type=int, required=True)
    p.add_argument("--softlib-body-calls", type=int, default=None)
    p.add_argument("--simd-body-calls", type=int, default=None)
    p.add_argument("--softlib-el-cyc", type=float, default=None)
    p.add_argument("--simd-el-cyc", type=float, default=None)
    p.add_argument("--softlib-mid", type=float, default=None)
    p.add_argument("--simd-mid", type=float, default=None)
    p.add_argument("--softlib-arm", type=str, default="SoftLib")
    p.add_argument("--simd-arm", type=str, default="SIMD")
    p.add_argument("--era", type=str, default="")
    p.add_argument("--shape", type=str, default="")
    p.add_argument("-o", "--out", type=Path, required=True)
    args = p.parse_args(argv)

    sl_counts = read_call_counts(args.softlib_csv)
    sm_counts = read_call_counts(args.simd_csv)
    soft = arm_block(
        arm=args.softlib_arm,
        counts=sl_counts,
        E=args.E,
        N=args.norm_elems,
        VF_cyc=args.softlib_vf,
        body_calls=args.softlib_body_calls,
        el_cyc=args.softlib_el_cyc,
        mid_IPC_W20=args.softlib_mid,
    )
    simd = arm_block(
        arm=args.simd_arm,
        counts=sm_counts,
        E=args.E,
        N=args.norm_elems,
        VF_cyc=args.simd_vf,
        body_calls=args.simd_body_calls,
        el_cyc=args.simd_el_cyc,
        mid_IPC_W20=args.simd_mid,
    )
    report = build_report(
        soft, simd, E=args.E, N=args.norm_elems, era=args.era, shape=args.shape
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    if report.get("pred_ratio_body") is not None:
        print(f"pred_ratio_body={report['pred_ratio_body']:.6g} meas_ratio={report.get('meas_ratio')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
