"""Running / rolling IPC from VF timeline logs + PR9-style per-opcode issue lanes.

Supported inputs (first match wins per arm):
  - chrome `trace.json` + `*_instr_exe.csv` (SV1-style RV_/SIMT_ issues)
  - `core0.veccore0.instr_popped_log.dump` (cycle-accurate Body4 / RV_ issues)
  - precomputed rolling CSV (`cycle` + `ipc` or `issue_rate`) — IPC-only (no lanes)
  - already-rendered PNG (wire only; no re-parse)

Window default W=20, left-open right-closed (t−W, t].

Plot semantics (adopted from SV2 / PR #9):
  - Top panel: one lane per instruction *kind* (short mnemonic); a mark at every issue.
  - Bottom panel: rolling IPC over the same VF-local tick axis.
  - Address-math opcodes (IADD / LEA / …) are counted in IPC but left off the lanes.
  - The VF opcode itself is excluded from both.

Never invents PNGs or numbers when inputs are missing.
"""

from __future__ import annotations

import bisect
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_W = 20
STEP = 1.0

# SIMT Body4 (SoftLib / Ascend SimtVF)
BODY4_PREFIXES = ("SIMT_LDS", "SIMT_STS", "SIMT_FADD", "SIMT_IADD", "SIMT_IADD_I")
# SIMD RV issues commonly used for IPC shape
SIMD_ISSUE_NAMES = (
    "RV_VLDI",
    "RV_VSTI",
    "RV_VADD",
    "RV_VMUL",
    "RV_VAND",
    "RV_VCGMAX",
    "RV_VLD",
    "RV_VST",
    "RV_VMAX",
)

TS_RE = re.compile(r"\[info\] \[(\d+)\]")
# opcode token inside popped dump / chrome name
OP_TOKEN_RE = re.compile(r"\b((?:SIMT_|RV_)[A-Z0-9_]+)\b")

# Shown as short names on lanes; stripped prefixes match PR #9.
STRIP_PREFIXES = ("SIMT_", "RV_")
STRIP_SUFFIXES = ("_THREAD_BLOCK",)

# Address / index math: counted in IPC, omitted from lane rows (PR #9).
LANE_HIDE = frozenset(
    {
        "IADD",
        "IADD_I",
        "LEA",
        "IMAD",
        "IMAD_I",
        "SHFI",
        "SHL",
        "SHR",
        "AND",  # scalar AND, not VAND
        "OR",
        "NEG",
        "NOP",
        "MOV_XD_IMM",
        "MOVK",
        "MOV_XD_SPR",
        "MOV_SPR_XN",
        "ADD",
        "ADD_IMM",
        "SUB_IMM",
        "MADD",
        "SBITSET",
        "MOVEMASK",
        "WAIT_FLAG",
        "SET_FLAG",
        "PUSH_PB",
        "DC_PRELOAD_XN_IMM",
        "INSERT_XD_XN",
        "MOV_SRC_TO_DST_ALIGNv2",
        "LD_XD_XN_IMM",
        "ST_XD_XN_IMM",
        "LDP_XI_XJ_XN",
        "DCCI",
        "END_LABEL",
        "SMOV",
        "PSET",
        "SEND",
        "VLOOPv2",
        "VDUPS",
    }
)

# Preferred lane order when present (PR #9 family); others follow by first-issue time.
LANE_ORDER = [
    "S2R",
    "LDS",
    "VLDI",
    "VLD",
    "F2F",
    "VMUL",
    "VAND",
    "VADD",
    "FADD",
    "FMNMX",
    "FMNMX_I",
    "VMAX",
    "VCGMAX",
    "VCMAX",
    "SEL",
    "FSETP",
    "FSETP_I",
    "REDUX",
    "BAR",
    "FDIV",
    "FDIV_I",
    "VDIV",
    "VDUP",
    "STS",
    "VSTI",
    "VST",
    "END",
]


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
    return cand


def short_op(name: str) -> str:
    s = str(name or "").strip()
    for pref in STRIP_PREFIXES:
        if s.startswith(pref):
            s = s[len(pref) :]
            break
    for suf in STRIP_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def is_vf_op(name: str) -> bool:
    s = str(name or "").strip()
    return s in ("VF", "VF_SIMD", "VF_SIMT") or s.startswith("VF_")


def is_issue_op(name: str) -> bool:
    """True for SIMT_/RV_ (and bare Body4 / SIMD mnemonics) worth counting."""
    s = str(name or "").strip()
    if not s or is_vf_op(s):
        return False
    if s.startswith("SIMT_") or s.startswith("RV_"):
        return True
    # already-short chrome names rarely appear; accept known shorts
    return short_op(s) in set(LANE_ORDER) or any(s.startswith(p) for p in BODY4_PREFIXES)


def load_vf_cycles_from_instr(instr_csv: Path) -> list[int]:
    rows: list[int] = []
    with instr_csv.open(newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            name = (r.get("instr") or r.get("name") or "").strip()
            if is_vf_op(name):
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
    """Issue timestamps + opcode names → local cycles inside one VF."""
    tr = json.loads(trace.read_text(encoding="utf-8"))
    evs = [e for e in tr.get("traceEvents", []) if e.get("ph") == "X"]
    vfs = sorted(
        [e for e in evs if is_vf_op(str(e.get("name", "")))],
        key=lambda e: e["ts"],
    )
    if not vfs:
        # also accept name startswith VF for robustness
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

    named: list[tuple[float, str]] = []
    for e in evs:
        name = str(e.get("name", ""))
        if not is_issue_op(name):
            continue
        ts = float(e["ts"])
        if vf_ts <= ts <= t_end:
            named.append(((ts - vf_ts) * us2cyc, short_op(name)))

    if len(named) < 5:
        raise RuntimeError(f"too few in-VF issues ({len(named)}) in {trace}")

    simt_n = sum(1 for _, op in named if op.startswith("LDS") or op in ("STS", "FADD", "FMNMX", "SEL", "REDUX"))
    simd_n = sum(1 for _, op in named if op.startswith("V") or op in ("VLDI", "VSTI", "VAND", "VCGMAX"))
    kind = "simt" if simt_n >= simd_n else "simd"
    local = [c for c, _ in named]
    return {
        "kind": kind,
        "issue_cycles": local,
        "issue_events": named,  # (cycle, short_op)
        "vf_index": idx,
        "vf_cyc": int(vf_c),
        "vf_cyc_sum": int(sum(vf_rows)),
        "n_issues": len(local),
        "source": "chrome_trace",
        "trace": str(trace),
        "instr_csv": str(instr_csv) if instr_csv else "",
    }


def _longest_body_region_events(
    events: list[tuple[int, str]], *, gap: int = 80
) -> list[tuple[int, str]]:
    if not events:
        return []
    evs = sorted(events, key=lambda x: x[0])
    regions: list[list[tuple[int, str]]] = [[evs[0]]]
    for e in evs[1:]:
        if e[0] - regions[-1][-1][0] > gap:
            regions.append([e])
        else:
            regions[-1].append(e)
    return max(regions, key=len)


def parse_popped_dump(dump: Path, *, kind: str = "auto", gap: int = 80) -> dict[str, Any]:
    """Cycle-accurate named issues from instr_popped_log.dump."""
    text = dump.read_text(encoding="utf-8", errors="replace")
    simt_ev: list[tuple[int, str]] = []
    simd_ev: list[tuple[int, str]] = []
    for line in text.splitlines():
        m = TS_RE.search(line)
        if not m:
            continue
        t = int(m.group(1))
        om = OP_TOKEN_RE.search(line)
        if not om:
            continue
        raw = om.group(1)
        if is_vf_op(raw):
            continue
        op = short_op(raw)
        if raw.startswith("SIMT_") or any(p in line for p in BODY4_PREFIXES):
            simt_ev.append((t, op))
        if raw.startswith("RV_") or any(n in line for n in SIMD_ISSUE_NAMES):
            if "SIMT_" not in line:
                simd_ev.append((t, op))

    if kind == "simt":
        events, used = simt_ev, "simt"
    elif kind == "simd":
        events, used = simd_ev, "simd"
    else:
        if len(simt_ev) >= len(simd_ev):
            events, used = simt_ev, "simt"
        else:
            events, used = simd_ev, "simd"

    if len(events) < 5:
        raise RuntimeError(
            f"too few issues in popped dump {dump} (simt={len(simt_ev)} simd={len(simd_ev)})"
        )
    body = _longest_body_region_events(events, gap=gap)
    if len(body) < 5:
        body = sorted(events, key=lambda x: x[0])
    base = body[0][0]
    named = [(float(c - base), op) for c, op in body]
    local = [c for c, _ in named]
    return {
        "kind": used,
        "issue_cycles": local,
        "issue_events": named,
        "vf_index": 0,
        "vf_cyc": int(local[-1] - local[0] + 1) if local else 0,
        "vf_cyc_sum": int(local[-1] - local[0] + 1) if local else 0,
        "n_issues": len(local),
        "n_issues_all": len(events),
        "source": "popped_dump",
        "popped_dump": str(dump),
        "absolute_first": base,
        "silence_gap": gap,
    }


def parse_rolling_csv(path: Path, *, ipc_col: str | None = None) -> dict[str, Any]:
    """Precomputed series: columns cycle + ipc|issue_rate (no opcode lanes)."""
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
        "issue_events": [],
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
        lo = bisect.bisect_right(ic, t - W)
        hi = bisect.bisect_right(ic, t)
        times.append(t - W / 2.0)
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


def lane_groups(events: list[tuple[float, str]], *, hide: frozenset[str] = LANE_HIDE) -> list[str]:
    """Opcode kinds to show as lanes: preferred order, then first-issue order."""
    by: dict[str, list[float]] = defaultdict(list)
    for t, op in events:
        if op in hide:
            continue
        by[op].append(t)
    if not by:
        # fall back: show top kinds even if all were "hidden"
        counts = Counter(op for _, op in events)
        return [op for op, _ in counts.most_common(12)]
    ordered: list[str] = []
    for name in LANE_ORDER:
        if name in by:
            ordered.append(name)
    rest = sorted((op for op in by if op not in ordered), key=lambda o: min(by[o]))
    return ordered + rest


# Stable-ish colors keyed by short mnemonic (PR #9 palette family).
_COLOR_MAP = {
    "LDS": "#1f4e79",
    "VLDI": "#1f4e79",
    "VLD": "#1f4e79",
    "STS": "#2e5c2a",
    "VSTI": "#2e5c2a",
    "VST": "#2e5c2a",
    "FADD": "#147e82",
    "VMUL": "#5a8fb8",
    "VAND": "#5a8fb8",
    "VADD": "#5a8fb8",
    "FMNMX": "#147e82",
    "FMNMX_I": "#c45911",
    "VMAX": "#147e82",
    "VCGMAX": "#c45911",
    "VCMAX": "#c45911",
    "SEL": "#147e82",
    "FSETP": "#8a8f98",
    "FSETP_I": "#8a8f98",
    "F2F": "#8a8f98",
    "REDUX": "#c45911",
    "BAR": "#963737",
    "FDIV": "#548235",
    "FDIV_I": "#548235",
    "VDIV": "#548235",
    "VDUP": "#8a8f98",
    "S2R": "#8a8f98",
    "END": "#b0b6bd",
}


def _color_for(op: str, idx: int) -> str:
    if op in _COLOR_MAP:
        return _COLOR_MAP[op]
    # cycle through tab10 for unknowns
    return f"C{idx % 10}"


def render_instr_issue_ipc(
    *,
    events: list[tuple[float, str]],
    times: list[float],
    ipcs: list[float],
    out_png: Path,
    title: str,
    W: float,
    mid_med: float | None = None,
    footer: str = "",
    vf_cyc: int | None = None,
    stages: list[dict[str, Any]] | None = None,
) -> Path:
    """PR #9 composite: opcode issue lanes (top) + rolling IPC (bottom).

    Optional ``stages``: list of {seg, t0, t1} from STAGE_TAXONOMY windows
    (``scripts.st_report.stage_taxonomy``). Draws light axvspan bands on both panes.
    """
    plt = _import_pyplot()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    groups = lane_groups(events)
    by: dict[str, list[float]] = defaultdict(list)
    for t, op in events:
        by[op].append(t)

    if times:
        x_max = max(times[-1], max((t for t, _ in events), default=0.0))
    else:
        x_max = max((t for t, _ in events), default=1.0)
    if vf_cyc and vf_cyc > x_max:
        x_max = float(vf_cyc)

    n_lanes = max(1, len(groups))
    fig_h = max(5.5, 1.6 + 0.38 * n_lanes + 2.4)
    fig, (ax_lanes, ax_ipc) = plt.subplots(
        2,
        1,
        figsize=(11.5, fig_h),
        sharex=True,
        gridspec_kw={"height_ratios": [max(1.2, 0.35 * n_lanes), 1.6], "hspace": 0.08},
    )

    # --- lanes ---
    for i, name in enumerate(groups):
        y = n_lanes - 1 - i
        ts = by.get(name) or []
        color = _color_for(name, i)
        if ts:
            ax_lanes.vlines(ts, y + 0.15, y + 0.85, colors=color, lw=1.0, alpha=0.95)
        ax_lanes.text(
            x_max * 1.005 if x_max else 1,
            y + 0.5,
            f"×{len(ts)}",
            ha="left",
            va="center",
            fontsize=8,
            color="#666666",
            clip_on=False,
        )

    ax_lanes.set_ylim(0, n_lanes)
    ax_lanes.set_yticks([n_lanes - 1 - i + 0.5 for i in range(n_lanes)])
    ax_lanes.set_yticklabels(groups, fontsize=9)
    ax_lanes.tick_params(axis="y", length=0)
    ax_lanes.set_xlim(0, x_max * 1.02 if x_max else 1)
    ax_lanes.set_title(title, fontsize=12, loc="left", pad=8)
    ax_lanes.text(
        0.0,
        1.02,
        "One lane per opcode kind. Address math counted in IPC, left off lanes. Marks = issues.",
        transform=ax_lanes.transAxes,
        fontsize=8,
        color="#666666",
        va="bottom",
    )
    ax_lanes.grid(True, axis="x", alpha=0.25)
    for spine in ("left", "right", "top"):
        ax_lanes.spines[spine].set_visible(False)

    # --- IPC ---
    ax_ipc.fill_between(times, ipcs, color="#e8dcee", alpha=0.9)
    ax_ipc.plot(
        times,
        ipcs,
        color="#60347a",
        lw=1.4,
        label=f"running IPC (W={int(W) if W == int(W) else W})",
    )
    if mid_med is not None and mid_med == mid_med:
        ax_ipc.axhline(mid_med, color="#c45911", ls="--", lw=1.0, label=f"mid-med={mid_med:.3f}")

    # Optional STAGE_TAXONOMY defining-opcode windows (may overlap).
    if stages:
        stage_colors = {
            "S_load": "#cfe2f3",
            "S_reduce": "#d9ead3",
            "S_scale": "#fff2cc",
            "S_eltwise": "#fce5cd",
            "S_store": "#ead1dc",
            "S_sync": "#efefef",
        }
        for stg in stages:
            t0, t1 = stg.get("t0"), stg.get("t1")
            if t0 is None or t1 is None:
                continue
            color = stage_colors.get(str(stg.get("seg") or ""), "#dddddd")
            ax_lanes.axvspan(float(t0), float(t1), color=color, alpha=0.22, lw=0, zorder=0)
            ax_ipc.axvspan(float(t0), float(t1), color=color, alpha=0.15, lw=0, zorder=0)
            # label at mid of span on lanes pane
            mid_t = 0.5 * (float(t0) + float(t1))
            ax_lanes.text(
                mid_t,
                n_lanes - 0.15,
                str(stg.get("seg") or ""),
                ha="center",
                va="top",
                fontsize=7,
                color="#555555",
                alpha=0.9,
                clip_on=True,
            )

    ax_ipc.set_xlabel("tick inside the VF window")
    ax_ipc.set_ylabel("IPC")
    ax_ipc.grid(True, alpha=0.3)
    ax_ipc.legend(fontsize=8, loc="upper right")
    if footer:
        fig.text(0.5, 0.01, footer, ha="center", fontsize=8, color="#444444")
    # Prefer subplots_adjust over tight_layout: twin y-axes on the IPC panel
    # make tight_layout warn and occasionally clip the shared legend.
    fig.subplots_adjust(left=0.08, right=0.96, top=0.94, bottom=0.08, hspace=0.18)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


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
    """Legacy IPC-only line chart (used when no opcode events, e.g. rolling_csv)."""
    plt = _import_pyplot()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.fill_between(times, ipcs, color="#e8dcee", alpha=0.9)
    ax.plot(times, ipcs, color="#60347a", lw=1.2, label=f"running IPC (W={int(W) if W==int(W) else W})")
    if mid_med is not None and mid_med == mid_med:
        ax.axhline(mid_med, color="#c45911", ls="--", label=f"mid-med IPC={mid_med:.3f}")
    ax.set_xlabel("tick inside the VF window")
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
        events: list[tuple[float, str]] = []
    elif popped is not None and popped.is_file():
        parsed = parse_popped_dump(popped, kind=kind)
        times, ipcs, meta = sliding_window_ipc(parsed["issue_cycles"], W=W)
        events = list(parsed.get("issue_events") or [])
    elif trace is not None and trace.is_file():
        parsed = parse_chrome_trace(
            trace, instr if instr and instr.is_file() else None, vf_index=vf_index
        )
        times, ipcs, meta = sliding_window_ipc(parsed["issue_cycles"], W=W)
        events = list(parsed.get("issue_events") or [])
    else:
        missing = []
        if not (popped and popped.is_file()):
            missing.append("popped_dump")
        if not (trace and trace.is_file()):
            missing.append("trace")
        if not (rolling and rolling.is_file()):
            missing.append("rolling_csv")
        raise FileNotFoundError(
            "no usable IPC logs ("
            + ", ".join(missing)
            + "); set artifacts.popped_dump|trace+instr_csv|rolling_csv"
        )

    mid_med = meta.get("mid_med_ipc")
    vf_cyc = int(parsed.get("vf_cyc") or 0)
    el = (E / vf_cyc) if (E and vf_cyc) else None
    footer_parts = [
        f"W={int(W) if W == int(W) else W}  window=(t−W,t]",
        f"source={parsed.get('source')}",
        "plot=opcode-lanes+IPC",
    ]
    if vf_cyc:
        footer_parts.append(f"VF≈{vf_cyc}")
    if el is not None:
        footer_parts.append(f"el/cyc={el:.4f}")
    if mid_med == mid_med:
        footer_parts.append(f"mid-med IPC={mid_med:.3f}")
    footer = "  ".join(footer_parts)

    if events:
        stage_spans = artifacts.get("stages")
        if stage_spans is None and artifacts.get("frontend"):
            try:
                from scripts.st_report.stage_taxonomy import stage_windows_from_events

                stage_spans = stage_windows_from_events(events, str(artifacts["frontend"]))
            except Exception:
                stage_spans = None
        render_instr_issue_ipc(
            events=events,
            times=times,
            ipcs=ipcs,
            out_png=out_png,
            title=title,
            W=W,
            mid_med=mid_med if mid_med == mid_med else None,
            footer=footer,
            vf_cyc=vf_cyc or None,
            stages=stage_spans,
        )
        if stage_spans:
            meta_stages = [
                {"seg": s.get("seg"), "t0": s.get("t0"), "t1": s.get("t1"),
                 "defining": s.get("defining"), "n_defining": s.get("n_defining")}
                for s in stage_spans
            ]
        else:
            meta_stages = None
        groups = lane_groups(events)
        meta = dict(meta)
        meta["lane_opcodes"] = groups
        meta["lane_counts"] = {g: sum(1 for _, op in events if op == g) for g in groups}
        meta["n_issues_all_named"] = float(len(events))
        meta["plot_style"] = "opcode_lanes_ipc"
        if meta_stages is not None:
            meta["stages"] = meta_stages
            meta["stage_taxonomy"] = "scripts.st_report.stage_taxonomy.STAGE_TAXONOMY"
    else:
        render_running_ipc(
            times=times,
            ipcs=ipcs,
            out_png=out_png,
            title=title,
            W=W,
            mid_med=mid_med if mid_med == mid_med else None,
            footer=footer,
        )
        meta = dict(meta)
        meta["plot_style"] = "ipc_only"

    out_json = out_png.with_suffix(".json")
    payload = {
        "W": W,
        "window": "(t-W,t]",
        "title": title,
        "png": str(out_png),
        "E": E,
        "el_cyc": el,
        "plot_style": meta.get("plot_style"),
        "parsed": {k: v for k, v in parsed.items() if k not in ("issue_cycles", "times", "ipcs", "issue_events")},
        "meta": meta,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"png": str(out_png), "json": str(out_json), "meta": meta, "source": parsed.get("source")}


def write_tiny_fixture(dir_path: Path) -> dict[str, Path]:
    """Synthetic chrome-trace + instr csv for self-test (not an ST measurement)."""
    dir_path.mkdir(parents=True, exist_ok=True)
    vf_cyc = 200
    issues = []
    ts0 = 1000.0
    # mix of opcodes so lanes appear (and some address math hidden from lanes)
    names = ["SIMT_LDS", "SIMT_FADD", "SIMT_STS", "SIMT_IADD", "SIMT_FMNMX", "SIMT_REDUX"]
    for i in range(200):
        issues.append(
            {
                "name": names[i % len(names)],
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
