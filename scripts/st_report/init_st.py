"""init: write arms.yaml (+ ANALYSIS stub if missing) from Aims/design only."""

from __future__ import annotations

import re
from typing import Any

from scripts.st_report.registry import (
    analysis_path,
    arms_path,
    load_arms,
    report_dir,
    st_dir,
)

# Default conclusion spine (3 arms). Measured fields stay null — harvest/SUMMARY own numbers.
DEFAULT_SPINE: list[dict[str, Any]] = [
    {
        "id": "simd_dsl_best",
        "label": "SIMD PTO-DSL best",
        "path": "dsl",
        "role": "simd_best",
        "el_cyc": None,
        "notes": "fill via harvest / measured SUMMARY; no invented el/cyc",
    },
    {
        "id": "simt_asc_default",
        "label": "SIMT Ascend default lowering",
        "path": "simt_asc",
        "role": "simt_asc",
        "el_cyc": None,
        "notes": "T-dim + LDS/STS width + el/cyc from measured run",
    },
    {
        "id": "softlib_simt",
        "label": "SIMT SoftLib / handcraft",
        "path": "softlib_simt",
        "role": "simt_hand",
        "el_cyc": None,
        "notes": "prefer LDS.32/STS.32 when SoT; stage-split when measured",
    },
]

OPTIONAL_SOFTLIB_SIMD: dict[str, Any] = {
    "id": "softlib_simd",
    "label": "SIMD SoftLib / handcraft",
    "path": "softlib_simd",
    "role": "simd_softlib",
    "el_cyc": None,
    "notes": "optional 4th conclusion arm when meaningfully distinct from DSL best",
}

# Design keywords → trail quarantine (never conclusion by default)
_TRAIL_HINTS: list[tuple[re.Pattern[str], dict[str, str]]] = [
    (
        re.compile(r"\b(fa|fallback[\s_-]?a|dual64)\b", re.I),
        {"id": "fa_dual64_vand", "reason": "Fallback-A / dual64 sensitivity trail"},
    ),
    (
        re.compile(r"\bmembar\b", re.I),
        {"id": "softlib_simd_membar", "reason": "membar / ST-separated experiment trail"},
    ),
    (
        re.compile(r"\b(bank[\s_-]?cliff|legacy\s+abs|vmul\s*\+\s*vmax)\b", re.I),
        {"id": "legacy_probe", "reason": "legacy / bank-cliff intermediate probe"},
    ),
]

_TRAIL_SOFTLIB_SIMD = re.compile(
    r"\b(softlib\s+simd|simd\s+softlib|softlib_simd)\b",
    re.I,
)

_PROMOTE_SOFTLIB_SIMD = re.compile(
    r"\b((4th|fourth)\s+arm|conclusion.*softlib\s+simd|softlib\s+simd.*(conclusion|spine|keep|promote)|promote.*softlib\s+simd)\b",
    re.I,
)


def _yaml_scalar(val: Any) -> str:
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    s = str(val)
    if "\n" in s or ":" in s or s.startswith(" ") or s.endswith(" "):
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if re.search(r"[#\[\]{},]", s) or s == "":
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return s


def _emit_arm(arm: dict[str, Any], indent: int = 2) -> list[str]:
    pad = " " * indent
    lines = [f"{pad}- id: {arm['id']}"]
    for key in ("label", "path", "role", "VF", "LDS_width", "el_cyc", "notes", "reason"):
        if key not in arm:
            continue
        val = arm[key]
        if key == "el_cyc" and val is None:
            lines.append(f"{pad}  el_cyc: null")
            continue
        if val is None and key != "el_cyc":
            continue
        lines.append(f"{pad}  {key}: {_yaml_scalar(val)}")
    return lines


def _emit_arms_yaml(
    st: str,
    aims: str,
    design: str,
    spine: list[dict[str, Any]],
    trail: list[dict[str, str]],
) -> str:
    aims_flat = " ".join(aims.split())
    lines: list[str] = [
        f"# benches/{st}/report/arms.yaml",
        f"st: {st}",
        "aims: >-",
        f"  {aims_flat}",
        "",
    ]
    if design.strip():
        design_flat = " ".join(design.split())
        lines += ["design: >-", f"  {design_flat}", ""]
    lines += [
        "# Exactly 3–4 conclusion arms. Trail must stay under `trail:`.",
        "# Measured el/cyc live in SUMMARY.tsv (harvest); do not invent here.",
        "spine:",
    ]
    for arm in spine:
        lines.extend(_emit_arm(arm))
    lines += [
        "",
        "# Quarantine: never appear in ANALYSIS conclusion tables / §7 ratios",
        "trail:",
    ]
    if trail:
        for t in trail:
            lines.extend(_emit_arm(t))
    else:
        lines.append("  []")
    lines.append("")
    return "\n".join(lines)


def _merge_preserve_measured(
    fresh: dict[str, Any], prev: dict[str, Any] | None
) -> dict[str, Any]:
    """Keep prior labels only. Measured el/cyc stay in SUMMARY.tsv (harvest)."""
    out = dict(fresh)
    if not prev:
        return out
    # Never copy dig numbers into arms.yaml — Aims/design contract only.
    if prev.get("label") and out.get("label") in (
        "SIMD PTO-DSL best",
        "SIMT Ascend default lowering",
        "SIMT SoftLib / handcraft",
        "SIMD SoftLib / handcraft",
    ):
        if prev.get("label") != out.get("label"):
            # Drop labels that embed measured numbers (e.g. "… 33.44").
            pl = str(prev["label"])
            if not any(ch.isdigit() for ch in pl):
                out["label"] = pl
    return out


def _build_spine_trail(
    design: str, existing: dict[str, Any] | None
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    prev_spine = {a["id"]: a for a in (existing or {}).get("spine") or []}
    prev_trail = {a["id"]: a for a in (existing or {}).get("trail") or []}

    spine: list[dict[str, Any]] = [
        _merge_preserve_measured(dict(a), prev_spine.get(a["id"])) for a in DEFAULT_SPINE
    ]
    trail: list[dict[str, str]] = []
    seen_trail: set[str] = set()

    promote_4th = bool(_PROMOTE_SOFTLIB_SIMD.search(design or ""))

    if promote_4th:
        spine.append(
            _merge_preserve_measured(
                dict(OPTIONAL_SOFTLIB_SIMD), prev_spine.get("softlib_simd")
            )
        )
    elif _TRAIL_SOFTLIB_SIMD.search(design or ""):
        tid = "softlib_simd_vand"
        trail.append(
            {
                "id": tid,
                "reason": prev_trail.get(tid, {}).get("reason")
                or "SoftLib SIMD mentioned in design; quarantined (not conclusion by default)",
            }
        )
        seen_trail.add(tid)

    for pat, meta in _TRAIL_HINTS:
        if pat.search(design or ""):
            tid = meta["id"]
            if tid in seen_trail:
                continue
            if any(a["id"] == tid for a in spine):
                continue
            reason = prev_trail.get(tid, {}).get("reason") or meta["reason"]
            trail.append({"id": tid, "reason": reason})
            seen_trail.add(tid)

    for tid, t in prev_trail.items():
        if tid in seen_trail:
            continue
        if any(a["id"] == tid for a in spine):
            continue
        trail.append(
            {"id": tid, "reason": t.get("reason") or "preserved from prior arms.yaml"}
        )
        seen_trail.add(tid)

    return spine, trail


_AIMS_SECTION_RE = re.compile(
    r"(^##\s*1\.\s*Aims\s*\n)(.*?)(?=^##\s*2\.|\Z)",
    re.M | re.S,
)


def _analysis_stub(st: str, aims: str, design: str, spine: list[dict[str, Any]]) -> str:
    aims_body = aims.strip() or "(TODO: confirm Aims with author)"
    design_body = (
        design.strip()
        or "(TODO: case matrix from design prompt — conclusion arms only)"
    )
    arm_rows = "\n".join(
        f"| {i+1} | **{a.get('label', a['id'])}** | `{a['id']}` | TODO |"
        for i, a in enumerate(spine)
    )
    return f"""# {st.upper()} analysis

**Format:** [`docs/ST_REPORT_FORMAT.md`](../../docs/ST_REPORT_FORMAT.md).
**Primary metric:** el/cyc = elements / VF cycles (name which VF).
**Never invent measured numbers.** Gaps marked **TODO**.

**Conclusion arms (spine only):**

| # | Path | id | el/cyc |
|---|------|----|-------:|
{arm_rows}

---

## 1. Aims

{aims_body}

---

## 2. Test-case design

{design_body}

| Axis | Choices |
|------|---------|
| **Shape / TILE** | TODO |
| **Datatype** | TODO |
| **SIMD PTO-DSL** | TODO |
| **SIMT Ascend default** | TODO |
| **SIMT SoftLib** | TODO |
| **Pass criteria** | Gold / verify PASS; el/cyc from measured VF cycles |

**Case matrix (spine arms only):** fill from harness `report/SUMMARY.tsv` after harvest — do not invent.

---

## 3. Layout and schedule (SIMT and SIMD)

### 3.1 SIMD PTO-DSL
TODO

### 3.2 SIMT Ascend default
TODO

### 3.3 SIMT SoftLib hand-craft
TODO

---

## 4. SIMD PTO-DSL performance

TODO — el/cyc, theory roof, sensitivity, instr mix, running IPC, overall/mid-avg rooftop % from harness.

---

## 5. SIMT (Ascend default) lowering performance

TODO — LDS/STS 32 vs 64, T-dim sensitivity, measured el/cyc.

---

## 6. SIMT hand-craft (SoftLib) analysis

TODO — prefer LDS.32/STS.32 when SoT; stage-split LD/compute/ST; stage IPC plots.

---

## 7. Conclusion

Spine arms only (3 or 4). Summary table and performance ratios from `report/SUMMARY.tsv`. Trail stays out.

| arm | el_cyc | status |
|---|---:|---|
| SIMD PTO-DSL best | TODO | TODO |
| SIMT Ascend default | TODO | TODO |
| SIMT SoftLib | TODO | TODO |

- SIMD ÷ SoftLib SIMT = TODO
- SIMD ÷ Asc default = TODO
"""


def _refresh_aims_section(text: str, aims: str) -> str:
    aims_body = aims.strip() + "\n\n"
    if _AIMS_SECTION_RE.search(text):
        return _AIMS_SECTION_RE.sub(rf"\1\n{aims_body}", text, count=1)
    return text.rstrip() + "\n\n## 1. Aims\n\n" + aims.strip() + "\n"


def init_st(
    st: str,
    aims: str,
    design: str = "",
    refresh_aims: bool = False,
) -> dict[str, Any]:
    """Write arms.yaml from Aims/design; stub ANALYSIS only if missing."""
    if not st or not re.match(r"^[A-Za-z0-9_-]+$", st):
        raise ValueError(f"invalid --st id: {st!r}")
    if not (aims or "").strip():
        raise ValueError("--aims is required (non-empty)")

    st_path = st_dir(st)
    st_path.mkdir(parents=True, exist_ok=True)
    report_dir(st).mkdir(parents=True, exist_ok=True)

    existing: dict[str, Any] | None = None
    ap = arms_path(st)
    if ap.is_file():
        try:
            existing = load_arms(st)
        except Exception:
            existing = None

    spine, trail = _build_spine_trail(design or "", existing)
    text = _emit_arms_yaml(st, aims, design or "", spine, trail)
    ap.write_text(text, encoding="utf-8")

    analysis = analysis_path(st)
    analysis_action = "left_existing"
    if not analysis.is_file():
        analysis.write_text(
            _analysis_stub(st, aims, design or "", spine), encoding="utf-8"
        )
        analysis_action = "wrote_stub"
    elif refresh_aims:
        prev = analysis.read_text(encoding="utf-8")
        analysis.write_text(_refresh_aims_section(prev, aims), encoding="utf-8")
        analysis_action = "refreshed_aims"

    return {
        "arms": str(ap),
        "spine": [a["id"] for a in spine],
        "trail": [t["id"] for t in trail],
        "analysis": str(analysis),
        "analysis_action": analysis_action,
    }
