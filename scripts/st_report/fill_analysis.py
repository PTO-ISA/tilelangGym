"""Emit §4–§7 stub tables from SUMMARY.tsv. Dry-run default; never trail."""

from __future__ import annotations

from typing import Any

from scripts.st_report.registry import load_arms, read_summary, trail_ids


def _md_escape(s: str) -> str:
    return s.replace("|", "\\|")


def _row_cells(r: dict[str, str], keys: list[str]) -> str:
    return "| " + " | ".join(_md_escape(r.get(k, "") or "—") for k in keys) + " |"


def build_stub(st: str) -> str:
    arms = load_arms(st)
    rows = read_summary(st)
    by_id = {r["arm"]: r for r in rows}
    trail = set(trail_ids(arms))
    for r in rows:
        if r["arm"] in trail:
            raise RuntimeError(f"trail arm leaked into SUMMARY: {r['arm']}")

    lines: list[str] = [
        f"# ANALYSIS stub tables for `{st}` (fill dry-run)",
        "",
        "Generated from `report/SUMMARY.tsv`. Trail arms omitted. Numbers not invented.",
        "",
        "## §4 SIMD PTO-DSL performance",
        "",
    ]
    simd = [r for r in rows if r.get("path") == "dsl" or r["arm"] == "simd_dsl_best"]
    keys = ["arm", "VF", "el_cyc", "mid_ipc_W", "mid_in_el_cyc", "roof_name", "roof_value", "pct_overall", "pct_mid", "status"]
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("|" + "|".join(["---"] * len(keys)) + "|")
    for r in simd:
        lines.append(_row_cells(r, keys))
    if not simd:
        lines.append("| — | — | — | — | — | — | — | — | — | TODO |")

    lines += ["", "## §5 SIMT Ascend default", ""]
    asc = [r for r in rows if r.get("path") == "simt_asc" or r["arm"] == "simt_asc_default"]
    keys5 = ["arm", "el_cyc", "LDS_width", "status", "source"]
    lines.append("| " + " | ".join(keys5) + " |")
    lines.append("|" + "|".join(["---"] * len(keys5)) + "|")
    for r in asc:
        lines.append(_row_cells(r, keys5))

    lines += ["", "## §6 SIMT SoftLib hand-craft", ""]
    soft = [r for r in rows if r.get("path") == "softlib_simt" or r["arm"] == "softlib_simt"]
    keys6 = ["arm", "el_cyc", "LDS_width", "mid_ipc_W", "roof_name", "status"]
    lines.append("| " + " | ".join(keys6) + " |")
    lines.append("|" + "|".join(["---"] * len(keys6)) + "|")
    for r in soft:
        lines.append(_row_cells(r, keys6))

    lines += ["", "## §7 Conclusion ratios", ""]
    dsl = by_id.get("simd_dsl_best", {})
    asc_r = by_id.get("simt_asc_default", {})
    soft_r = by_id.get("softlib_simt", {})
    lines.append("| arm | el_cyc | status |")
    lines.append("|---|---:|---|")
    for label, r in (
        ("SIMD PTO-DSL best", dsl),
        ("SIMT Ascend default", asc_r),
        ("SIMT SoftLib", soft_r),
    ):
        lines.append(f"| {label} | {r.get('el_cyc') or 'TODO'} | {r.get('status') or 'TODO'} |")

    def _f(x: str) -> float | None:
        try:
            return float(x) if x else None
        except ValueError:
            return None

    a = _f(dsl.get("el_cyc", ""))
    b = _f(soft_r.get("el_cyc", ""))
    c = _f(asc_r.get("el_cyc", ""))
    lines.append("")
    if a is not None and b is not None and b != 0:
        lines.append(f"- SIMD ÷ SoftLib SIMT = {a} / {b} ≈ **{a / b:.2f}×**")
    else:
        lines.append("- SIMD ÷ SoftLib SIMT = TODO")
    if a is not None and c is not None and c != 0:
        lines.append(f"- SIMD ÷ Asc default = {a} / {c} ≈ **{a / c:.2f}×**")
    else:
        lines.append("- SIMD ÷ Asc default = TODO (need Asc el_cyc)")
    lines.append("")
    lines.append("*(dry-run only — no ANALYSIS.md write)*")
    lines.append("")
    return "\n".join(lines)


def fill(st: str, write: bool = False) -> dict[str, Any]:
    stub = build_stub(st)
    if write:
        raise RuntimeError(
            "fill --write refused in milestone 1 without explicit author OK; use --dry-run"
        )
    return {"dry_run": True, "text": stub}
