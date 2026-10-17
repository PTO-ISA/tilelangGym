#!/usr/bin/env python3
"""Simt Body4 30-cyc IPC from chrome-trace (aligned TILE8K@T512)."""
from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WIN_CYC = 30.0
STEP = 1.0
BODY4 = ("SIMT_LDS", "SIMT_STS", "SIMT_FADD", "SIMT_IADD", "SIMT_IADD_I")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--instr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--E", type=int, default=16384)
    ap.add_argument("--TILE", type=int, default=8192)
    ap.add_argument("--T", type=int, default=512)
    args = ap.parse_args()

    tr = json.load(open(args.trace))
    evs = [e for e in tr.get("traceEvents", []) if e.get("ph") == "X"]
    body = sorted(
        [e for e in evs if any(str(e.get("name", "")).startswith(p) or e.get("name") == p for p in BODY4)
         or any(p in str(e.get("name", "")) for p in BODY4)],
        key=lambda e: e["ts"],
    )
    # Prefer exact RV_/SIMT_ issue names
    body2 = [e for e in evs if str(e.get("name", "")).startswith("SIMT_")]
    if len(body2) > len(body) // 2:
        body = sorted(body2, key=lambda e: e["ts"])

    vf_cyc = None
    for r in csv.DictReader(open(args.instr)):
        name = (r.get("instr") or "").strip()
        if name in ("VF", "VF_SIMD", "VF_SIMT") or name.startswith("VF_"):
            vf_cyc = int(float(r["cycles"]))
            break
    if not body or vf_cyc is None:
        # fallback: use instr_exe counts only — no time series
        print(f"WARN no body events in trace (n={len(body)}) vf={vf_cyc}; skip plot")
        return 1

    vf_evs = [e for e in evs if str(e.get("name", "")).startswith("VF")]
    if not vf_evs:
        print("WARN no VF event")
        return 1
    vf_e = vf_evs[0]
    vf_ts, vf_dur = vf_e["ts"], vf_e["dur"]
    ghz = vf_cyc / (vf_dur * 1000.0)
    us2cyc = ghz * 1000.0

    def to_cyc(ts):
        return (ts - vf_ts) * us2cyc

    ic = [to_cyc(e["ts"]) for e in body if vf_ts <= e["ts"] <= vf_ts + vf_dur]
    if len(ic) < 10:
        print(f"WARN too few body issues {len(ic)}")
        return 1
    body_start, body_end = ic[0], ic[-1]
    ts, ipcs = [], []
    mid_ipcs = []
    mid_lo = body_start + 0.15 * (body_end - body_start)
    mid_hi = body_start + 0.85 * (body_end - body_start)
    t = body_start
    while t <= body_end - WIN_CYC + 1e-9:
        lo = bisect.bisect_left(ic, t)
        hi = bisect.bisect_left(ic, t + WIN_CYC)
        ipc = (hi - lo) / WIN_CYC
        c = t + WIN_CYC / 2
        ts.append(c)
        ipcs.append(ipc)
        if mid_lo <= c <= mid_hi:
            mid_ipcs.append(ipc)
        t += STEP

    med = float(np.median(mid_ipcs)) if mid_ipcs else float("nan")
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.plot(ts, ipcs, color="C2", lw=1.0, label="Body4 IPC (30-cyc)")
    ax.axhline(med, color="C2", ls="--", label=f"mid-med IPC={med:.3f}")
    ax.set_xlabel("cycle (from VF)")
    ax.set_ylabel("IPC")
    ax.set_title(f"Simt Body4 — E={args.E} TILE={args.TILE} T={args.T} — 30-cyc IPC")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    footer = (
        f"E={args.E} TILE={args.TILE} T={args.T}  VF={vf_cyc}  "
        f"el/cyc={args.E/vf_cyc:.4f}  mid-med Body4 IPC={med:.3f}  n_issues={len(ic)}"
    )
    fig.text(0.5, 0.01, footer, ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    meta = {
        "vf_cyc": vf_cyc,
        "mid_med_ipc": med,
        "el_per_cyc": args.E / vf_cyc,
        "n_issues": len(ic),
        "E": args.E,
        "TILE": args.TILE,
        "T": args.T,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"WROTE {out}")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
