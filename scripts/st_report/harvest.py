"""Seed / update report/SUMMARY.tsv from arms.yaml (+ artifacts later)."""

from __future__ import annotations

from typing import Any

from scripts.st_report.registry import (
    SUMMARY_COLUMNS,
    load_arms,
    read_summary,
    report_dir,
    write_summary,
)


def _cell(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        # keep compact measured style
        s = f"{val:g}"
        return s
    return str(val)


def _seed_row(arm: dict[str, Any]) -> dict[str, str]:
    row = {c: "" for c in SUMMARY_COLUMNS}
    row["arm"] = arm["id"]
    row["path"] = _cell(arm.get("path"))
    if arm.get("VF") is not None:
        row["VF"] = _cell(arm.get("VF"))
    el = arm.get("el_cyc")
    if el is None:
        row["status"] = "TODO"
        row["source"] = ""
    else:
        row["el_cyc"] = _cell(el)
        row["status"] = "PASS"
        row["source"] = "arms.yaml#seed"
    lds = arm.get("LDS_width")
    if lds is not None:
        row["LDS_width"] = _cell(lds)
        row["roof_name"] = row["roof_name"] or f"LDS{lds}"
    elif arm.get("path") == "dsl":
        row["LDS_width"] = "n/a"
    # Never invent mid-IPC / roof % — leave empty unless arm explicitly sets them.
    for key in (
        "mid_ipc_W",
        "mid_in_el_cyc",
        "mid_out_el_cyc",
        "roof_name",
        "roof_value",
        "pct_overall",
        "pct_mid",
    ):
        if key in arm and arm[key] is not None and key in row:
            row[key] = _cell(arm[key])
    if arm.get("roof_name") is not None:
        row["roof_name"] = _cell(arm["roof_name"])
    if arm.get("roof_value") is not None:
        row["roof_value"] = _cell(arm["roof_value"])
    return row


def _merge_existing(seed: dict[str, str], prev: dict[str, str] | None) -> dict[str, str]:
    """Keep measured SUMMARY cells when arms seed leaves them empty (init-safe)."""
    if not prev:
        return seed
    out = dict(seed)
    # Prefer a prior non-seed harvest source wholesale when seed has no el_cyc.
    prev_src = str(prev.get("source") or "")
    if prev_src and not prev_src.startswith("arms.yaml") and not (seed.get("el_cyc") or "").strip():
        return {c: prev.get(c, "") or "" for c in SUMMARY_COLUMNS}
    for c in SUMMARY_COLUMNS:
        if c == "arm":
            continue
        if (not out.get(c) or not str(out.get(c)).strip()) and (prev.get(c) or "").strip():
            out[c] = prev[c]
    # If we restored el_cyc from prev, do not leave a stale TODO status.
    if (out.get("el_cyc") or "").strip() and (out.get("status") or "").upper() == "TODO":
        out["status"] = (prev.get("status") or "PASS") or "PASS"
        if not (out.get("source") or "").strip():
            out["source"] = prev.get("source") or "SUMMARY#preserved"
    return out


def harvest(st: str) -> dict[str, Any]:
    """Write SUMMARY.tsv from spine seeds. Preserve existing measured cells."""
    arms = load_arms(st)
    report_dir(st).mkdir(parents=True, exist_ok=True)
    existing = {r["arm"]: r for r in read_summary(st)}
    rows: list[dict[str, str]] = []
    for arm in arms["spine"]:
        seed = _seed_row(arm)
        prev = existing.get(arm["id"])
        rows.append(_merge_existing(seed, prev))
    path = write_summary(st, rows)
    return {"path": str(path), "rows": rows, "n": len(rows)}
