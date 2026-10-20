#!/usr/bin/env python3
"""30-cyc sliding-window IPC / el/cyc for SV1d tiled SIMD (chrome-trace).

SoT plot = **one VF** (default: first body VF). Multi-tile ping/pong is noise for
the IPC shape — do not draw dual bursts with an empty inter-tile MTE gap.

Overall el/cyc still uses sum of all VF cycles (matches SUMMARY). Mid-body IPC /
el/cyc are measured inside the selected VF only (15–85% of that VF's issue span).
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
ROOF_THEORY = 44.0  # theory rooftop (design target)
VL = 64


def load(trace: Path, instr: Path):
    tr = json.load(open(trace))
    evs = [e for e in tr.get("traceEvents", []) if e.get("ph") == "X"]
    rv = [e for e in evs if str(e.get("name", "")).startswith(("RV_", "VF"))]
    vld = sorted([e for e in rv if e["name"] == "RV_VLDI"], key=lambda e: e["ts"])
    vst = sorted([e for e in rv if e["name"] == "RV_VSTI"], key=lambda e: e["ts"])
    vadd = sorted([e for e in rv if e["name"] == "RV_VADD"], key=lambda e: e["ts"])
    vfs = sorted([e for e in rv if e.get("name") == "VF"], key=lambda e: e["ts"])
    vf_rows = []
    for r in csv.DictReader(open(instr)):
        name = (r.get("instr") or "").strip()
        if name in ("VF", "VF_SIMD", "VF_SIMT") or name.startswith("VF_"):
            vf_rows.append(int(float(r["cycles"])))
    vf_cyc = sum(vf_rows)
    if vf_cyc <= 0 or not vfs or not vld or not vst or not vadd:
        raise RuntimeError(
            f"incomplete: vf_cyc={vf_cyc} vfs={len(vfs)} vld={len(vld)} vst={len(vst)} vadd={len(vadd)}"
        )
    if len(vf_rows) != len(vfs):
        each = vf_cyc // len(vfs)
        vf_rows = [each] * len(vfs)
        vf_rows[-1] = vf_cyc - each * (len(vfs) - 1)
    return vld, vst, vadd, vfs, vf_rows, vf_cyc


def pick_vf_index(vfs, vf_rows, spec: str) -> int:
    """spec: '0'..'N-1', 'first', or 'longest'."""
    n = len(vfs)
    if n == 0:
        raise RuntimeError("no VF events")
    s = (spec or "first").strip().lower()
    if s in ("first", "0"):
        return 0
    if s in ("longest", "main", "auto"):
        return max(range(n), key=lambda i: (vf_rows[i], float(vfs[i].get("dur") or 0)))
    idx = int(s)
    if idx < 0 or idx >= n:
        raise RuntimeError(f"vf-index {idx} out of range 0..{n-1}")
    return idx


def sliding_one_vf(vld, vst, vadd, vfs, vf_rows, vf_cyc, E: int, vf_index: int):
    mid_ipcs, mid_els = [], []
    mid25_ipcs = []
    peak_ipc = -1.0
    peak_excl = -1.0
    early_ipc = -1.0

    vf_e = vfs[vf_index]
    vf_c = vf_rows[vf_index]
    vf_ts = vf_e["ts"]
    vf_dur = float(vf_e.get("dur") or 0)
    if vf_dur <= 0 or vf_c <= 0:
        raise RuntimeError(f"VF[{vf_index}] has no duration/cycles")
    us2cyc = vf_c / vf_dur  # ts is us → local cycles

    def to_local(ts):
        return (ts - vf_ts) * us2cyc

    t_end = vf_ts + vf_dur
    local_issues = []
    for e in sorted(vld + vst + vadd, key=lambda z: z["ts"]):
        if not (vf_ts <= e["ts"] <= t_end):
            continue
        name = e["name"]
        local_issues.append(
            {
                "c": to_local(e["ts"]),
                "name": name,
                "el_st": VL if name == "RV_VSTI" else 0,
            }
        )
    if len(local_issues) < 5:
        raise RuntimeError(f"VF[{vf_index}] has too few issues ({len(local_issues)})")

    ic = [x["c"] for x in local_issues]
    el_st = [x["el_st"] for x in local_issues]
    body_start, body_end = ic[0], ic[-1]
    span = body_end - body_start
    mid_lo = body_start + 0.15 * span
    mid_hi = body_start + 0.85 * span
    mid25 = body_start + 0.25 * span
    mid75 = body_start + 0.75 * span

    all_ts, all_ipcs, all_els = [], [], []
    t = body_start
    max_t = body_end - WIN_CYC
    while t <= max_t + 1e-9:
        lo = bisect.bisect_left(ic, t)
        hi = bisect.bisect_left(ic, t + WIN_CYC)
        n = hi - lo
        ipc = n / WIN_CYC
        el = sum(el_st[lo:hi]) / WIN_CYC
        center = t + WIN_CYC / 2
        all_ts.append(center)
        all_ipcs.append(ipc)
        all_els.append(el)
        if mid_lo <= center <= mid_hi:
            mid_ipcs.append(ipc)
            mid_els.append(el)
        if mid25 <= center <= mid75:
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
        "vf_index": vf_index,
        "vf_cyc_selected": vf_c,
        "overall_el_per_cyc": E / vf_cyc,
        "mid_med_ipc": pct(mid_ipcs, 50),
        "mid_med_el": pct(mid_els, 50),
        "peak_ipc": peak_ipc,
        "peak_excl_fill_drain_ipc": peak_excl,
        "peak_mid_ipc": pct(mid25_ipcs, 95) if mid25_ipcs else float("nan"),
        "early_fill_ipc": early_ipc,
        "n_windows": len(all_ipcs),
        "n_mid_windows": len(mid_ipcs),
    }
    return (
        np.array(all_ts, dtype=float),
        np.array(all_ipcs, dtype=float),
        np.array(all_els, dtype=float),
        summary,
    )


def plot(ts, ipcs, els, summary, out_png: Path, title: str, E: int):
    fig, ax1 = plt.subplots(figsize=(11, 5.2))
    ax2 = ax1.twinx()
    ax1.plot(ts, ipcs, color="C0", lw=1.0, label="IPC (30-cyc win, single VF)")
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
    ax2.axhline(
        ROOF_THEORY,
        color="0.55",
        ls="-.",
        lw=1.0,
        label=f"theory rooftop ~{ROOF_THEORY:.0f}",
    )
    ax1.set_xlabel(f"cycle within VF[{summary['vf_index']}] (issue span)")
    ax1.set_ylabel("IPC")
    ax2.set_ylabel("el/cyc")
    ax1.set_title(title)
    footer = (
        f"E={E}  single VF[{summary['vf_index']}]={summary['vf_cyc_selected']} cyc  "
        f"(n_VF={summary['n_vf']} total, VF_sum={summary['vf_cyc']})  "
        f"overall={summary['overall_el_per_cyc']:.2f} el/cyc  |  "
        f"mid-med IPC={summary['mid_med_ipc']:.2f} → {summary['mid_med_el']:.1f} el/cyc  |  "
        f"peak excl fill/drain IPC={summary['peak_excl_fill_drain_ipc']:.2f}  "
        f"peak mid≈{summary['peak_mid_ipc']:.2f}  early fill={summary['early_fill_ipc']:.2f}  |  "
        f"vs derived roof {ROOF_DERIVED}: mid {100 * summary['mid_med_el'] / ROOF_DERIVED:.0f}%  "
        f"overall {100 * summary['overall_el_per_cyc'] / ROOF_THEORY:.0f}% of theory rooftop (~{ROOF_THEORY:.0f})"
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
    ap.add_argument(
        "--vf-index",
        default="first",
        help="which VF to plot: first|longest|0..N-1 (default first)",
    )
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    vld, vst, vadd, vfs, vf_rows, vf_cyc = load(Path(args.trace), Path(args.instr))
    vi = pick_vf_index(vfs, vf_rows, args.vf_index)
    title = args.title or (
        f"SV1d SIMD VL=64 — E={args.E} TILE={args.TILE} — single VF[{vi}] 30-cyc IPC / el/cyc"
    )
    ts, ipcs, els, summary = sliding_one_vf(
        vld, vst, vadd, vfs, vf_rows, vf_cyc, args.E, vi
    )
    summary["E"] = args.E
    summary["TILE"] = args.TILE
    plot(ts, ipcs, els, summary, Path(args.out), title, args.E)
    meta = Path(args.out).with_suffix(".json")
    safe = {k: (None if isinstance(v, float) and (v != v) else v) for k, v in summary.items()}
    meta.write_text(json.dumps(safe, indent=2))
    print(f"WROTE {meta}")


if __name__ == "__main__":
    main()