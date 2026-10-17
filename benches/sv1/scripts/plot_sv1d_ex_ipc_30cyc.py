#!/usr/bin/env python3
"""30-cyc sliding-window IPC / el/cyc for SV1d tiled SIMD (chrome-trace).

Handles multi-VF kernels (one VF per tile): sums VF cycles, and only
slides windows inside VF intervals so inter-tile MTE gaps do not zero mid-IPC.
"""
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
ROOF_DERIVED = 42.67
ROOF_LOK = 44.0
VL = 64


def load(trace: Path, instr: Path):
    tr = json.load(open(trace))
    evs = [e for e in tr.get("traceEvents", []) if e.get("ph") == "X"]
    rv = [e for e in evs if str(e.get("name", "")).startswith(("RV_", "VF"))]
    vld = sorted([e for e in rv if e["name"] == "RV_VLDI"], key=lambda e: e["ts"])
    vst = sorted([e for e in rv if e["name"] == "RV_VSTI"], key=lambda e: e["ts"])
    vadd = sorted([e for e in rv if e["name"] == "RV_VADD"], key=lambda e: e["ts"])
    vfs = sorted(
        [e for e in rv if e.get("name") == "VF" or str(e.get("name", "")).startswith("VF")],
        key=lambda e: e["ts"],
    )
    vf_cyc = 0
    for r in csv.DictReader(open(instr)):
        name = (r.get("instr") or "").strip()
        if name in ("VF", "VF_SIMD", "VF_SIMT") or name.startswith("VF_"):
            vf_cyc += int(float(r["cycles"]))
    if vf_cyc <= 0 or not vfs or not vld or not vst or not vadd:
        raise RuntimeError(
            f"incomplete: vf_cyc={vf_cyc} vfs={len(vfs)} vld={len(vld)} vst={len(vst)} vadd={len(vadd)}"
        )
    return vld, vst, vadd, vfs, vf_cyc


def sliding(vld, vst, vadd, vfs, vf_cyc, E: int):
    # Calibrate us→cyc from total VF duration vs total VF cycles
    vf_dur = sum(float(v.get("dur") or 0) for v in vfs)
    ghz = vf_cyc / (vf_dur * 1000.0) if vf_dur > 0 else 1.8
    us2cyc = ghz * 1000.0
    t0 = vfs[0]["ts"]

    def to_cyc(ts_us):
        return (ts_us - t0) * us2cyc

    # VF intervals in cycles (from first VF start)
    vf_intervals = []
    for v in vfs:
        a = to_cyc(v["ts"])
        b = to_cyc(v["ts"] + float(v.get("dur") or 0))
        vf_intervals.append((a, b))

    def in_vf(c):
        return any(a <= c <= b for a, b in vf_intervals)

    issues = []
    for e in sorted(vld + vst + vadd, key=lambda z: z["ts"]):
        name = e["name"]
        c = to_cyc(e["ts"])
        if not in_vf(c):
            continue
        issues.append(
            {
                "c": c,
                "name": name,
                "el_st": VL if name == "RV_VSTI" else 0,
                "el_add": VL if name == "RV_VADD" else 0,
            }
        )
    if len(issues) < 10:
        raise RuntimeError(f"too few issues in VF intervals: {len(issues)}")

    ic = [x["c"] for x in issues]
    el_st = [x["el_st"] for x in issues]
    body_start, body_end = ic[0], ic[-1]

    ts, ipcs, els = [], [], []
    mid_ipcs, mid_els = [], []
    mid_lo = body_start + 0.15 * (body_end - body_start)
    mid_hi = body_start + 0.85 * (body_end - body_start)
    mid25 = body_start + 0.25 * (body_end - body_start)
    mid75 = body_start + 0.75 * (body_end - body_start)
    mid25_ipcs = []
    peak_ipc = -1.0
    peak_excl = -1.0
    early_ipc = -1.0

    # Slide only inside each VF interval to avoid MTE gaps
    for a, b in vf_intervals:
        t = max(a, body_start)
        max_t = min(b, body_end) - WIN_CYC
        while t <= max_t + 1e-9:
            t1 = t + WIN_CYC
            lo = bisect.bisect_left(ic, t)
            hi = bisect.bisect_left(ic, t1)
            n = hi - lo
            ipc = n / WIN_CYC
            el = sum(el_st[lo:hi]) / WIN_CYC
            center = t + WIN_CYC / 2
            ts.append(center)
            ipcs.append(ipc)
            els.append(el)
            if mid_lo <= center <= mid_hi and ipc > 0:
                mid_ipcs.append(ipc)
                mid_els.append(el)
            if mid25 <= center <= mid75 and ipc > 0:
                mid25_ipcs.append(ipc)
                peak_excl = max(peak_excl, ipc)
            if center < mid_lo:
                early_ipc = max(early_ipc, ipc)
            peak_ipc = max(peak_ipc, ipc)
            t += STEP

    def pct(xs, p):
        if not xs:
            return float("nan")
        s = sorted(xs)
        return s[min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))]

    summary = {
        "vf_cyc": vf_cyc,
        "n_vf": len(vfs),
        "ghz": ghz,
        "overall_el_per_cyc": E / vf_cyc,
        "mid_med_ipc": pct(mid_ipcs, 50),
        "mid_med_el": pct(mid_els, 50),
        "peak_ipc": peak_ipc,
        "peak_excl_fill_drain_ipc": peak_excl,
        "peak_mid_ipc": pct(mid25_ipcs, 95) if mid25_ipcs else float("nan"),
        "early_fill_ipc": early_ipc,
        "body_start": body_start,
        "body_end": body_end,
        "n_windows": len(ipcs),
        "n_mid_windows": len(mid_ipcs),
    }
    return np.array(ts), np.array(ipcs), np.array(els), summary


def plot(ts, ipcs, els, summary, out_png: Path, title: str, E: int):
    fig, ax1 = plt.subplots(figsize=(11, 5.2))
    ax2 = ax1.twinx()
    ax1.plot(ts, ipcs, color="C0", lw=1.0, label="IPC (30-cyc win, in-VF)")
    ax2.plot(ts, els, color="C1", lw=1.0, alpha=0.85, label="el/cyc via ST (30-cyc)")
    ax1.axhline(
        summary["mid_med_ipc"],
        color="C0",
        ls="--",
        lw=1.0,
        alpha=0.8,
        label=f"mid-med IPC={summary['mid_med_ipc']:.2f}",
    )
    ax2.axhline(ROOF_DERIVED, color="0.35", ls=":", lw=1.2, label=f"derived roof {ROOF_DERIVED}")
    ax2.axhline(ROOF_LOK, color="0.55", ls="-.", lw=1.0, label=f"Lok ~{ROOF_LOK}")
    ax1.set_xlabel("cycle (from first VF start; gaps = inter-tile MTE)")
    ax1.set_ylabel("IPC")
    ax2.set_ylabel("el/cyc")
    ax1.set_title(title)
    footer = (
        f"E={E}  VF_sum={summary['vf_cyc']} cyc (n_VF={summary['n_vf']})  "
        f"overall={summary['overall_el_per_cyc']:.2f} el/cyc  |  "
        f"mid-med IPC={summary['mid_med_ipc']:.2f} → {summary['mid_med_el']:.1f} el/cyc  |  "
        f"peak excl fill/drain IPC={summary['peak_excl_fill_drain_ipc']:.2f}  "
        f"peak mid≈{summary['peak_mid_ipc']:.2f}  early fill={summary['early_fill_ipc']:.2f}  |  "
        f"vs roof {ROOF_DERIVED}: mid {100 * summary['mid_med_el'] / ROOF_DERIVED:.0f}%  "
        f"overall {100 * summary['overall_el_per_cyc'] / ROOF_LOK:.0f}% of ~{ROOF_LOK}"
    )
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=8, wrap=True)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"WROTE {out_png}")
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--instr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--E", type=int, default=16384)
    ap.add_argument("--TILE", type=int, default=8192)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    title = args.title or f"SV1d SIMD VL=64 — E={args.E} TILE={args.TILE} — 30-cyc IPC / el/cyc"
    vld, vst, vadd, vfs, vf_cyc = load(Path(args.trace), Path(args.instr))
    ts, ipcs, els, summary = sliding(vld, vst, vadd, vfs, vf_cyc, args.E)
    summary["E"] = args.E
    summary["TILE"] = args.TILE
    plot(ts, ipcs, els, summary, Path(args.out), title, args.E)
    meta = Path(args.out).with_suffix(".json")
    meta.write_text(json.dumps(summary, indent=2))
    print(f"WROTE {meta}")


if __name__ == "__main__":
    main()
