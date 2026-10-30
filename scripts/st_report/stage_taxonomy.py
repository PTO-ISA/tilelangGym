"""Shared STAGE_TAXONOMY — defining-opcode map for §8.1 / §8.2 / staged PR9.

SoT for segment ids and per-frontend defining opcodes. Stage windows =
first..last issue of defining opcodes (overlap OK under dual_drain / fused
VL-res). ISA-mix (ops/N families in isa_mix.py) stays orthogonal — do not
fold family tables into these stage windows.

Seeded from dig ``sb8_g32_stage_mid_ipc.json`` (SoftLib SIMT + SIMD VL-res).
Prefer this module over hand-aligned maps in ANALYSIS prose.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from scripts.st_report.rolling_ipc import short_op, sliding_window_ipc

# ---------------------------------------------------------------------------
# SoT
# ---------------------------------------------------------------------------

STAGE_WINDOW_DEF = (
    "first..last issue of defining opcodes "
    "(windows may overlap under dual_drain / fused VL-res)"
)

# Ordered segment ids used by §8.1 / §8.2 / staged PR9.
SEG_ORDER: list[str] = [
    "S_load",
    "S_reduce",
    "S_scale",
    "S_eltwise",
    "S_store",
]

# Optional sync / fence bucket (usually empty on HW bar; companions may appear in CSV mix).
SEG_SYNC = "S_sync"

# frontend key → seg → defining opcode short-names (post SIMT_/RV_ strip)
STAGE_TAXONOMY: dict[str, dict[str, list[str]]] = {
    "softlib_simt": {
        "S_load": ["LDS"],
        "S_reduce": ["REDUX", "FADD", "SEL", "FSETP_I"],
        "S_scale": ["FDIV_I"],
        "S_eltwise": ["FMUL"],
        "S_store": ["STS"],
        "S_sync": [],
    },
    "simd": {
        "S_load": ["VLDI"],
        "S_reduce": ["VCMAX", "VMAX", "VAND"],
        "S_scale": ["VSEL", "VDIV", "VDUP"],
        "S_eltwise": ["VMUL"],
        "S_store": ["VSTI"],
        "S_sync": [],
    },
}

# Aliases so callers can pass path / role / arm-ish labels.
FRONTEND_ALIASES: dict[str, str] = {
    "softlib_simt": "softlib_simt",
    "softlib": "softlib_simt",
    "simt_softlib": "softlib_simt",
    "simt": "softlib_simt",
    "body4": "softlib_simt",
    "simd": "simd",
    "simd_dsl": "simd",
    "dsl": "simd",
    "vmi": "simd",
    "pto_dsl": "simd",
    "vlres": "simd",
}


def normalize_frontend(frontend: str) -> str:
    key = (frontend or "").strip().lower().replace("-", "_")
    if key in FRONTEND_ALIASES:
        return FRONTEND_ALIASES[key]
    if key in STAGE_TAXONOMY:
        return key
    raise KeyError(
        f"unknown STAGE_TAXONOMY frontend {frontend!r}; "
        f"known={sorted(STAGE_TAXONOMY)} aliases={sorted(FRONTEND_ALIASES)}"
    )


def defining_opcodes(frontend: str, seg: str) -> list[str]:
    fe = normalize_frontend(frontend)
    segs = STAGE_TAXONOMY[fe]
    if seg not in segs:
        raise KeyError(f"unknown seg {seg!r} for frontend {fe}; known={list(segs)}")
    return list(segs[seg])


def frontend_map(frontend: str) -> dict[str, list[str]]:
    return {k: list(v) for k, v in STAGE_TAXONOMY[normalize_frontend(frontend)].items()}


def opcode_matches_defining(op: str, defining: Iterable[str]) -> bool:
    """Substring-safe match on short opcodes (exact / underscore prefix).

    Avoids SEL⊂VSEL false positives: requires equality or ``stem_`` prefix.
    """
    s = short_op(op).upper()
    if not s:
        return False
    for d in defining:
        du = (d or "").strip().upper()
        if not du:
            continue
        if s == du:
            return True
        if s.startswith(du + "_") or du.startswith(s + "_"):
            return True
    return False


def classify_opcode(frontend: str, op: str) -> str | None:
    """Map one opcode → seg via defining list (first match in SEG_ORDER then S_sync)."""
    fe = normalize_frontend(frontend)
    segs = STAGE_TAXONOMY[fe]
    for seg in SEG_ORDER:
        if opcode_matches_defining(op, segs.get(seg) or []):
            return seg
    if opcode_matches_defining(op, segs.get(SEG_SYNC) or []):
        return SEG_SYNC
    return None


def opcode_to_seg_buckets(frontend: str) -> dict[str, list[str]]:
    """§8.1 helper: same seg names + defining opcodes (CSV aggregates assign via classify)."""
    return frontend_map(frontend)


def to_json_dict() -> dict[str, Any]:
    return {
        "schema": "stage_taxonomy.v1",
        "stage_window_def": STAGE_WINDOW_DEF,
        "seg_order": list(SEG_ORDER) + [SEG_SYNC],
        "frontends": {
            fe: {seg: list(ops) for seg, ops in segs.items()}
            for fe, segs in STAGE_TAXONOMY.items()
        },
        "aliases": dict(FRONTEND_ALIASES),
        "notes": {
            "§8.1": "CSV call aggregates: assign opcodes via classify_opcode; "
            "companions (MTE2/MTE3/IADD_…) may sit in the same harvest row but "
            "are not defining for windows.",
            "§8.2": "Stage mid-IPC windows = first..last issue of defining opcodes.",
            "PR9_staged": "Annotate *_pr9_staged.png from the same defining windows.",
            "isa_mix": "Orthogonal ops/N families (load/store/reduce/…); do not merge.",
        },
        "module": "scripts.st_report.stage_taxonomy.STAGE_TAXONOMY",
    }


def write_json(path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def default_export_path(st: str | None = None) -> Path:
    from scripts.st_report.registry import repo_root, st_dir

    if st:
        # Prefer benches/<st>/plots/ (SV8 convention) over report/plots.
        p = st_dir(st) / "plots" / "stage_taxonomy.json"
        return p
    return repo_root() / "scripts" / "st_report" / "stage_taxonomy.json"


# ---------------------------------------------------------------------------
# Stage windows + mid-IPC harvest (consumes STAGE_TAXONOMY)
# ---------------------------------------------------------------------------


def stage_windows_from_events(
    events: list[tuple[float, str]],
    frontend: str,
) -> list[dict[str, Any]]:
    """Return per-seg windows from VF-local (t, short_op) events."""
    fe = normalize_frontend(frontend)
    segs = STAGE_TAXONOMY[fe]
    out: list[dict[str, Any]] = []
    for seg in SEG_ORDER:
        defining = list(segs.get(seg) or [])
        hits = [(t, op) for t, op in events if opcode_matches_defining(op, defining)]
        if not hits:
            out.append(
                {
                    "seg": seg,
                    "defining": defining,
                    "n_defining": 0,
                    "n_all_in_window": 0,
                    "t0": None,
                    "t1": None,
                    "span": None,
                }
            )
            continue
        t0 = float(hits[0][0])
        t1 = float(hits[-1][0])
        n_all = sum(1 for t, _ in events if t0 <= t <= t1)
        out.append(
            {
                "seg": seg,
                "defining": defining,
                "n_defining": len(hits),
                "n_all_in_window": n_all,
                "t0": t0,
                "t1": t1,
                "span": t1 - t0 + 1.0,
            }
        )
    return out


def _mid_slice(xs: list[float], lo: float = 0.15, hi: float = 0.85) -> list[float]:
    if not xs:
        return []
    n = len(xs)
    a = int(n * lo)
    b = max(a + 1, int(n * hi))
    return xs[a:b]


def stage_mid_ipc_for_arm(
    *,
    events: list[tuple[float, str]],
    issue_cycles: list[float],
    frontend: str,
    W: float = 20.0,
    title: str = "",
    popped_dump: str | None = None,
    official: dict[str, Any] | None = None,
    plot_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Harvest one arm's stage mid-IPC using STAGE_TAXONOMY defining windows."""
    times, ipcs, _ipc_meta = sliding_window_ipc(issue_cycles, W=W)
    windows = stage_windows_from_events(events, frontend)
    stages: list[dict[str, Any]] = []
    for w in windows:
        row = dict(w)
        t0, t1 = w.get("t0"), w.get("t1")
        if t0 is None or t1 is None or not times:
            row["mid_med_ipc_W20"] = None
            row["mid_mean_ipc_W20"] = None
            row["full_issue_proxy_ipc"] = None
            stages.append(row)
            continue
        # IPC samples whose tick falls inside [t0, t1]
        in_win = [ipc for t, ipc in zip(times, ipcs) if t0 <= t <= t1]
        mid = _mid_slice(in_win)
        row["mid_med_ipc_W20"] = (
            float(statistics.median(mid)) if mid else (float(statistics.median(in_win)) if in_win else None)
        )
        row["mid_mean_ipc_W20"] = (
            float(statistics.fmean(mid)) if mid else (float(statistics.fmean(in_win)) if in_win else None)
        )
        # full-window issue proxy = n_all / span
        span = w.get("span") or 0.0
        n_all = w.get("n_all_in_window") or 0
        row["full_issue_proxy_ipc"] = (float(n_all) / span) if span else None
        stages.append(row)

    arm: dict[str, Any] = {
        "title": title,
        "frontend": normalize_frontend(frontend),
        "taxonomy": "scripts.st_report.stage_taxonomy.STAGE_TAXONOMY",
        "popped_dump": popped_dump,
        "stages": stages,
    }
    if official:
        arm.update(official)
    if plot_meta:
        arm.update(plot_meta)
    if times and ipcs:
        mid_all = _mid_slice(ipcs)
        arm.setdefault(
            "plot_mid_med_ipc",
            float(statistics.median(mid_all)) if mid_all else float(statistics.median(ipcs)),
        )
        arm.setdefault("plot_mean_body", float(statistics.fmean(ipcs)))
        arm.setdefault("plot_VF_cyc", int(max(issue_cycles)) + 1 if issue_cycles else None)
    return arm


def harvest_stage_mid(
    arms_spec: list[dict[str, Any]],
    *,
    W: float = 20.0,
) -> dict[str, Any]:
    """Build a stage_mid_ipc.json document from arm specs.

    Each spec needs: id, frontend, popped_dump (Path/str). Optional title / official / plot_meta.
    Parses dumps via rolling_ipc.parse_popped_dump.
    """
    from scripts.st_report.rolling_ipc import parse_popped_dump

    arms_out: dict[str, Any] = {}
    for spec in arms_spec:
        dump = Path(spec["popped_dump"])
        fe = spec["frontend"]
        kind = "simt" if normalize_frontend(fe) == "softlib_simt" else "simd"
        parsed = parse_popped_dump(dump, kind=kind)
        arm = stage_mid_ipc_for_arm(
            events=list(parsed["issue_events"]),
            issue_cycles=list(parsed["issue_cycles"]),
            frontend=fe,
            W=W,
            title=spec.get("title") or spec["id"],
            popped_dump=str(dump),
            official=spec.get("official"),
            plot_meta=spec.get("plot_meta"),
        )
        arms_out[spec["id"]] = arm
    return {
        "W": float(W),
        "window": "(t-W,t]",
        "stage_window_def": STAGE_WINDOW_DEF,
        "taxonomy": "scripts.st_report.stage_taxonomy.STAGE_TAXONOMY",
        "taxonomy_module": "scripts.st_report.stage_taxonomy",
        "arms": arms_out,
    }


def assert_matches_dig_defining(dig: dict[str, Any]) -> list[str]:
    """Return mismatch lines vs a dig stage_mid JSON (defining lists only)."""
    # Dig arm id → frontend
    guess = {
        "softlib_simt_c3e2e_dual_drain_g32": "softlib_simt",
        "simd_vlres_g32": "simd",
    }
    errs: list[str] = []
    for arm_id, arm in (dig.get("arms") or {}).items():
        fe = guess.get(arm_id)
        if not fe:
            # try title heuristics
            title = (arm.get("title") or "").lower()
            fe = "softlib_simt" if "softlib" in title or "simt" in title else "simd"
        for stg in arm.get("stages") or []:
            seg = stg["seg"]
            got = list(stg.get("defining") or [])
            want = defining_opcodes(fe, seg)
            if got != want:
                errs.append(f"{arm_id}.{seg}: dig={got} taxonomy={want}")
    return errs


__all__ = [
    "STAGE_TAXONOMY",
    "STAGE_WINDOW_DEF",
    "SEG_ORDER",
    "SEG_SYNC",
    "FRONTEND_ALIASES",
    "normalize_frontend",
    "defining_opcodes",
    "frontend_map",
    "opcode_matches_defining",
    "classify_opcode",
    "opcode_to_seg_buckets",
    "to_json_dict",
    "write_json",
    "default_export_path",
    "stage_windows_from_events",
    "stage_mid_ipc_for_arm",
    "harvest_stage_mid",
    "assert_matches_dig_defining",
]
