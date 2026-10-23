"""Running / rolling IPC from VF timeline logs.

Supported inputs (first match wins per arm):
  - chrome `trace.json` + `*_instr_exe.csv` (SV1-style RV_/SIMT_ issues)
  - `core0.veccore0.instr_popped_log.dump` (cycle-accurate Body4 / RV_ issues)
  - precomputed rolling CSV (`cycle` + `ipc` or `issue_rate`)
  - already-rendered PNG (wire only; no re-parse)

Window default W=20, left-open right-closed (t−W, t].
Never invents PNGs or numbers when inputs are missing.
"""

from __future__ import annotations

import bisect
import csv
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_W = 20
STEP = 1.0

# SIMT Body4 (SoftLib / Ascend SimtVF)
BODY4_PREFIXES = ("SIMT_LDS", "SIMT_STS", "SIMT_FADD", "SIMT_IADD", "SIMT_IADD_I")
# SIMD RV issues commonly used for IPC shape
SIMD_ISSUE_NAMES = ("RV_VLDI", "RV_VSTI", "RV_VADD", "RV_VMUL", "RV_VAND", "RV_VCGMAX", "RV_VLD", "RV_VST")

TS_RE = re.compile(r"\[info\] \[(\d+)\]")


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def resolve_path(raw: str | Path, *, st_root: Path, repo_root: Path) -> Path:
    """Resolve artifact path: absolute, repo-relative, or ST-dir-relative."""
    p = Path(str(raw))
    if p.is_absolute():
        return p
    cand = (st_root / p).resolve()
    if cand.exists():
        return cand
    cand2 = (repo_root / p).resolve()
    if cand2.exists():
        return cand2
    return cand  # prefer st-relative even if missing (caller checks)


def load_vf_cycles_from_instr(instr_csv: Path) -> list[int]:
    rows: list[int] = []
    with instr_csv.open(newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            name = (r.get("instr") or r.get("name") or "").strip()
            if name in ("VF", "VF_SIMD", "VF_SIMT") or name.startswith("VF_"):
                try:
                    rows.append(int(float(r.get("cycles") or r.get("cycle") or 0)))
                except ValueError:
                    continue
    return rows


def parse_chrome_trace(
    trace: Path,
    instr_csv: Path | None = None,
    *,
    vf_index: str = "first",
) -> dict[str, Any]:
    """Issue timestamps → local cycles inside one VF. Raises on incomplete input."""
    tr = json.loads(trace.read_text(encoding="utf-8"))
    evs = [e for e in tr.get("traceEvents", []) if e.get("ph") == "X"]
    vfs = sorted(
        [e for e in evs if str(e.get("name", "")).startswith("VF")],
        key=lambda e: e["ts"],
    )
    if not vfs:
        raise RuntimeError(f"no VF events in {trace}")

    vf_rows: list[int] = []
    if instr_csv is not None and instr_csv.is_file():
        vf_rows = load_vf_cycles_from_instr(instr_csv)
    if not vf_rows:
        # last resort: use event dur only → relative IPC ok, absolute el/cyc needs E+VF
        vf_rows = [max(1, int(float(v.get("dur") or 1))) for v in vfs]

    if len(vf_rows) != len(vfs):
        total = sum(vf_rows) if vf_rows else 0
        if total <= 0:
            total = len(vfs)
        each = total // len(vfs)
        vf_rows = [each] * len(vfs)
        vf_rows[-1] = total - each * (len(vfs) - 1)

    s = (vf_index or "first").strip().lower()
    if s in ("longest", "main", "auto"):
        idx = max(range(len(vfs)), key=lambda i: (vf_rows[i], float(vfs[i].get("dur") or 0)))
    elif s in ("first", "0"):
        idx = 0
    else:
        idx = int(s)
        if idx < 0 or idx >= len(vfs):
            raise RuntimeError(f"vf-index {idx} out of range 0..{len(vfs)-1}")

    vf_e = vfs[idx]
    vf_c = vf_rows[idx]
    vf_ts = float(vf_e["ts"])
    vf_dur = float(vf_e.get("dur") or 0)
    if vf_dur <= 0 or vf_c <= 0:
        raise RuntimeError(f"VF[{idx}] missing duration/cycles")
    us2cyc = vf_c / vf_dur
    t_end = vf_ts + vf_dur

    simt = [
        e
        for e in evs
        if any(str(e.get("name", "")).startswith(p) or str(e.get("name", "")) == p for p in BODY4_PREFIXES)
        or str(e.get("name", "")).startswith("SIMT_")
    ]
    simd = [e for e in evs if str(e.get("name", "")) in SIMD_ISSUE_NAMES or str(e.get("name", "")).startswith("RV_")]
    body = simt if len(simt) >= max(10, len(simd) // 2) else simd
    if len(body) < 10:
        body = simt or simd
    if len(body) < 5:
        raise RuntimeError(f"too few issue events in {trace} (simt={len(simt)} simd={len(simd)})")

    kind = "simt" if body is simt or (simt and len(simt) >= len(simd)) else "simd"
    issues = sorted(
        (ts for e in body if vf_ts <= (ts := float(e["ts"])) <= t_end),
    )
    local = [(ts - vf_ts) * us2cyc for ts in issues]
    if len(local) < 5:
        raise RuntimeError(f"too few in-VF issues ({len(local)}) in {trace}")
    return {
        "kind": kind,
        "issue_cycles": local,
        "vf_index": idx,
        "vf_cyc": int(vf_c),
        "vf_cyc_sum": int(sum(vf_rows)),
        "n_issues": len(local),
        "source": "chrome_trace",
        "trace": str(trace),
        "instr_csv": str(instr_csv) if instr_csv else "",
    }


def _longest_body_region(cycles: list[int], *, gap: int = 80) -> list[int]:
    """Split issues on silence > gap cycles; keep the densest (longest) body."""
    if not cycles:
        return []
    cs = sorted(cycles)
    regions: list[list[int]] = [[cs[0]]]
    for c in cs[1:]:
        if c - regions[-1][-1] > gap:
            regions.append([c])
        else:
            regions[-1].append(c)
    return max(regions, key=len)


def parse_popped_dump(dump: Path, *, kind: str = "auto", gap: int = 80) -> dict[str, Any]:
    """Cycle-accurate issue list from instr_popped_log.dump.

    Uses the longest Body4/RV_ region (silence gap default 80 cyc) so inter-VF
    MTE gaps do not zero out mid-window IPC.
    """
    text = dump.read_text(encoding="utf-8", errors="replace")
    simt_cyc: list[int] = []
    simd_cyc: list[int] = []
    for line in text.splitlines():
        m = TS_RE.search(line)
        if not m:
            continue
        t = int(m.group(1))
        if any(p in line for p in BODY4_PREFIXES):
            simt_cyc.append(t)
        elif "SIMT_" in line:
            simt_cyc.append(t)
        if any(n in line for n in SIMD_ISSUE_NAMES) or ("RV_" in line and "SIMT_" not in line):
            simd_cyc.append(t)

    if kind == "simt":
        cycles, used = simt_cyc, "simt"
    elif kind == "simd":
        cycles, used = simd_cyc, "simd"
    else:
        if len(simt_cyc) >= len(simd_cyc):
            cycles, used = simt_cyc, "simt"
        else:
            cycles, used = simd_cyc, "simd"

    if len(cycles) < 5:
        raise RuntimeError(f"too few issues in popped dump {dump} (simt={len(simt_cyc)} simd={len(simd_cyc)})")
    body = _longest_body_region(cycles, gap=gap)
    if len(body) < 5:
        body = sorted(cycles)
    base = body[0]
    local = [c - base for c in body]
    return {
        "kind": used,
        "issue_cycles": local,
        "vf_index": 0,
        "vf_cyc": int(local[-1] - local[0] + 1) if local else 0,
        "vf_cyc_sum": int(local[-1] - local[0] + 1) if local else 0,
        "n_issues": len(local),
        "n_issues_all": len(cycles),
        "source": "popped_dump",
        "popped_dump": str(dump),
        "absolute_first": base,
        "silence_gap": gap,
    }


def parse_rolling_csv(path: Path, *, ipc_col: str | None = None) -> dict[str, Any]:
    """Precomputed series: columns cycle + ipc|issue_rate."""
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty rolling csv {path}")
    cols = {c.lower(): c for c in rows[0].keys()}
    cyc_key = cols.get("cycle") or cols.get("t") or cols.get("rel")
    if ipc_col:
        rate_key = ipc_col
    else:
        rate_key = cols.get("ipc") or cols.get("issue_rate") or cols.get("issue_ipc")
    if not cyc_key or not rate_key:
        raise RuntimeError(f"rolling csv needs cycle+ipc columns: {path} got {list(rows[0])}")
    times: list[float] = []
    ipcs: list[float] = []
    for r in rows:
        try:
            times.append(float(r[cyc_key]))
            ipcs.append(float(r[rate_key]))
        except (KeyError, ValueError):
            continue
    if len(times) < 5:
        raise RuntimeError(f"too few rows in {path}")
    return {
        "kind": "precomputed",
        "times": times,
        "ipcs": ipcs,
        "issue_cycles": [],
        "vf_cyc": 0,
        "vf_cyc_sum": 0,
        "n_issues": 0,
        "source": "rolling_csv",
        "rolling_csv": str(path),
        "precomputed": True,
    }


def sliding_window_ipc(
    issue_cycles: list[float],
    *,
    W: float = DEFAULT_W,
    step: float = STEP,
) -> tuple[list[float], list[float], dict[str, float]]:
    """IPC over (t−W, t] for t from first+W .. last. Mid = 15–85% of issue span."""
    ic = sorted(float(x) for x in issue_cycles)
    if len(ic) < 5:
        raise RuntimeError("need ≥5 issues for sliding IPC")
    body_start, body_end = ic[0], ic[-1]
    if body_end - body_start < W:
        raise RuntimeError(f"body span {body_end - body_start:.1f} < W={W}")

    times: list[float] = []
    ipcs: list[float] = []
    t = body_start + W
    while t <= body_end + 1e-9:
        # (t−W, t] → bisect_right for right, bisect_right for left exclusive
        lo = bisect.bisect_right(ic, t - W)
        hi = bisect.bisect_right(ic, t)
        times.append(t - W / 2.0)  # window center for display
        ipcs.append((hi - lo) / W)
        t += step

    mid_lo = body_start + 0.15 * (body_end - body_start)
    mid_hi = body_start + 0.85 * (body_end - body_start)
    mid = [ipc for c, ipc in zip(times, ipcs) if mid_lo <= c <= mid_hi]
    mid.sort()
    meta = {
        "W": float(W),
        "body_start": float(body_start),
        "body_end": float(body_end),
        "peak_ipc": float(max(ipcs)) if ipcs else float("nan"),
        "mean_ipc_body": float(sum(ipcs) / len(ipcs)) if ipcs else float("nan"),
        "mid_med_ipc": float(mid[len(mid) // 2]) if mid else float("nan"),
        "mid_mean_ipc": float(sum(mid) / len(mid)) if mid else float("nan"),
        "n_windows": float(len(ipcs)),
        "n_issues": float(len(ic)),
    }
    return times, ipcs, meta


def render_running_ipc(
    *,
    times: list[float],
    ipcs: list[float],
    out_png: Path,
    title: str,
    W: float,
    mid_med: float | None = None,
    footer: str = "",
) -> Path:
    plt = _import_pyplot()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.plot(times, ipcs, color="C0", lw=1.0, label=f"running IPC (W={int(W) if W==int(W) else W})")
    if mid_med is not None and mid_med == mid_med:  # not NaN
        ax.axhline(mid_med, color="C1", ls="--", label=f"mid-med IPC={mid_med:.3f}")
    ax.set_xlabel("cycle (from VF / first issue)")
    ax.set_ylabel("IPC")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    if footer:
        fig.text(0.5, 0.01, footer, ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.06, 1, 1] if footer else None)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


def build_from_artifacts(
    artifacts: dict[str, Any],
    *,
    st_root: Path,
    repo_root: Path,
    W: float = DEFAULT_W,
    out_png: Path,
    title: str,
    E: float | None = None,
) -> dict[str, Any]:
    """Generate PNG+JSON from artifacts dict. Raises if no usable log input."""
    art = artifacts or {}

    def _p(key: str) -> Path | None:
        raw = art.get(key)
        if not raw:
            return None
        return resolve_path(raw, st_root=st_root, repo_root=repo_root)

    popped = _p("popped_dump")
    trace = _p("trace")
    instr = _p("instr_csv")
    rolling = _p("rolling_csv")
    kind = str(art.get("kind") or "auto")
    vf_index = str(art.get("vf_index") or "first")
    if E is None and art.get("E") is not None:
        try:
            E = float(art["E"])
        except (TypeError, ValueError):
            E = None

    parsed: dict[str, Any]
    if rolling is not None and rolling.is_file():
        parsed = parse_rolling_csv(rolling)
        times, ipcs = parsed["times"], parsed["ipcs"]
        mid = sorted(ipcs[len(ipcs) // 8 : (7 * len(ipcs)) // 8]) if ipcs else []
        meta = {
            "W": float(W),
            "peak_ipc": float(max(ipcs)) if ipcs else float("nan"),
            "mean_ipc_body": float(sum(ipcs) / len(ipcs)) if ipcs else float("nan"),
            "mid_med_ipc": float(mid[len(mid) // 2]) if mid else float("nan"),
            "mid_mean_ipc": float(sum(mid) / len(mid)) if mid else float("nan"),
            "n_windows": float(len(ipcs)),
            "n_issues": 0.0,
            "precomputed": True,
        }
    elif popped is not None and popped.is_file():
        parsed = parse_popped_dump(popped, kind=kind)
        times, ipcs, meta = sliding_window_ipc(parsed["issue_cycles"], W=W)
    elif trace is not None and trace.is_file():
        parsed = parse_chrome_trace(trace, instr if instr and instr.is_file() else None, vf_index=vf_index)
        times, ipcs, meta = sliding_window_ipc(parsed["issue_cycles"], W=W)
    else:
        missing = []
        if not (popped and popped.is_file()):
            missing.append("popped_dump")
        if not (trace and trace.is_file()):
            missing.append("trace")
        if not (rolling and rolling.is_file()):
            missing.append("rolling_csv")
        raise FileNotFoundError(
            "no usable IPC logs (" + ", ".join(missing) + "); set artifacts.popped_dump|trace+instr_csv|rolling_csv"
        )

    mid_med = meta.get("mid_med_ipc")
    vf_cyc = int(parsed.get("vf_cyc") or 0)
    el = (E / vf_cyc) if (E and vf_cyc) else None
    footer_parts = [f"W={int(W) if W == int(W) else W}  window=(t−W,t]", f"source={parsed.get('source')}"]
    if vf_cyc:
        footer_parts.append(f"VF={vf_cyc}")
    if el is not None:
        footer_parts.append(f"el/cyc={el:.4f}")
    if mid_med == mid_med:
        footer_parts.append(f"mid-med IPC={mid_med:.3f}")
    footer = "  ".join(footer_parts)

    render_running_ipc(
        times=times,
        ipcs=ipcs,
        out_png=out_png,
        title=title,
        W=W,
        mid_med=mid_med if mid_med == mid_med else None,
        footer=footer,
    )
    out_json = out_png.with_suffix(".json")
    payload = {
        "W": W,
        "window": "(t-W,t]",
        "title": title,
        "png": str(out_png),
        "E": E,
        "el_cyc": el,
        "parsed": {k: v for k, v in parsed.items() if k not in ("issue_cycles", "times", "ipcs")},
        "meta": meta,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"png": str(out_png), "json": str(out_json), "meta": meta, "source": parsed.get("source")}


def write_tiny_fixture(dir_path: Path) -> dict[str, Path]:
    """Synthetic chrome-trace + instr csv for self-test (not an ST measurement)."""
    dir_path.mkdir(parents=True, exist_ok=True)
    # 200 issues at ~1.0 IPC steady, VF=200 cycles, dur=200 us → us2cyc=1
    vf_cyc = 200
    issues = []
    ts0 = 1000.0
    for i in range(200):
        issues.append(
            {
                "name": "SIMT_FADD" if i % 4 == 0 else ("SIMT_LDS" if i % 4 == 1 else ("SIMT_STS" if i % 4 == 2 else "SIMT_IADD")),
                "ph": "X",
                "ts": ts0 + float(i),
                "dur": 0.5,
                "pid": 1,
                "tid": 1,
            }
        )
    trace = {
        "traceEvents": [
            {"name": "VF_SIMT", "ph": "X", "ts": ts0, "dur": float(vf_cyc), "pid": 1, "tid": 1},
            *issues,
        ]
    }
    trace_path = dir_path / "trace.json"
    instr_path = dir_path / "instr_exe.csv"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    instr_path.write_text("instr,cycles\nVF_SIMT,200\n", encoding="utf-8")
    return {"trace": trace_path, "instr_csv": instr_path}
