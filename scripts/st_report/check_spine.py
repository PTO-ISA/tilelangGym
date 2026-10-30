"""Gate: spine size, no trail in ANALYSIS, PASS rows need el_cyc."""

from __future__ import annotations

import re
from typing import Any

from scripts.st_report.registry import (
    analysis_path,
    load_arms,
    read_summary,
    spine_ids,
    trail_ids,
)


PLACEHOLDER_RE = re.compile(r"\b(TBD_EL_CYC|FIXME_NUMBER|XXX\.XX)\b", re.I)


def check(st: str) -> dict[str, Any]:
    arms = load_arms(st)
    spine = spine_ids(arms)
    trail = trail_ids(arms)
    errors: list[str] = []
    warnings: list[str] = []

    n = len(spine)
    if n not in (3, 4):
        errors.append(f"spine count {n} not in {{3,4}}: {spine}")

    rows = read_summary(st)
    by_arm = {r["arm"]: r for r in rows}
    for sid in spine:
        if sid not in by_arm:
            errors.append(f"spine arm missing from SUMMARY.tsv: {sid}")
            continue
        r = by_arm[sid]
        if (r.get("status") or "").upper() == "PASS" and not (r.get("el_cyc") or "").strip():
            errors.append(f"PASS row missing el_cyc: {sid}")

    for r in rows:
        if r["arm"] in trail:
            errors.append(f"trail id present in SUMMARY.tsv: {r['arm']}")

    ap = analysis_path(st)
    if ap.is_file():
        text = ap.read_text(encoding="utf-8")
        for tid in trail:
            # word-ish match on trail id tokens
            if re.search(rf"\b{re.escape(tid)}\b", text):
                errors.append(f"ANALYSIS.md mentions trail id `{tid}`")
        # also catch common trail labels the format says to drop from conclusion lead
        for needle, why in (
            ("softlib_simd_vand", "trail id"),
            ("fa_dual64_vand", "trail id"),
            ("softlib_simd_membar", "trail id"),
        ):
            if needle in trail and needle in text:
                if f"ANALYSIS.md mentions trail id `{needle}`" not in errors:
                    errors.append(f"ANALYSIS.md mentions trail id `{needle}` ({why})")
        if PLACEHOLDER_RE.search(text):
            errors.append("ANALYSIS.md contains invent-looking placeholder token")
    else:
        warnings.append(f"no ANALYSIS.md at {ap}")

    ok = not errors
    return {"ok": ok, "errors": errors, "warnings": warnings, "spine": spine, "trail": trail}
