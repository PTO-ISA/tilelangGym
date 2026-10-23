"""Load benches/<st>/report/arms.yaml and resolve report paths."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


SUMMARY_COLUMNS = [
    "arm",
    "path",
    "VF",
    "el_cyc",
    "mid_ipc_W",
    "mid_in_el_cyc",
    "mid_out_el_cyc",
    "roof_name",
    "roof_value",
    "pct_overall",
    "pct_mid",
    "LDS_width",
    "status",
    "source",
]


def repo_root() -> Path:
    # scripts/st_report/registry.py → repo root
    return Path(__file__).resolve().parents[2]


def st_dir(st: str) -> Path:
    return repo_root() / "benches" / st


def report_dir(st: str) -> Path:
    return st_dir(st) / "report"


def analysis_path(st: str) -> Path:
    return st_dir(st) / "ANALYSIS.md"


def arms_path(st: str) -> Path:
    return report_dir(st) / "arms.yaml"


def summary_path(st: str) -> Path:
    return report_dir(st) / "SUMMARY.tsv"


def gaps_path(st: str) -> Path:
    return report_dir(st) / "gaps.md"


def plots_dir(st: str) -> Path:
    return report_dir(st) / "plots"


def load_arms(st: str) -> dict[str, Any]:
    path = arms_path(st)
    if not path.is_file():
        raise FileNotFoundError(f"missing arms.yaml: {path}")
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = _minimal_yaml_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"arms.yaml must be a mapping: {path}")
    spine = data.get("spine") or []
    trail = data.get("trail") or []
    if not isinstance(spine, list) or not isinstance(trail, list):
        raise ValueError("spine/trail must be lists")
    data["spine"] = spine
    data["trail"] = trail
    data["st"] = data.get("st") or st
    return data


def spine_ids(arms: dict[str, Any]) -> list[str]:
    return [a["id"] for a in arms["spine"]]


def trail_ids(arms: dict[str, Any]) -> list[str]:
    return [a["id"] for a in arms["trail"]]


def read_summary(st: str) -> list[dict[str, str]]:
    path = summary_path(st)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in reader]


def write_summary(st: str, rows: list[dict[str, str]]) -> Path:
    path = summary_path(st)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = {c: row.get(c, "") or "" for c in SUMMARY_COLUMNS}
            writer.writerow(out)
    return path


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    """Tiny subset parser for arms.yaml if PyYAML is missing."""
    # Prefer PyYAML; this fallback handles the SV5 seed shape only.
    import re

    data: dict[str, Any] = {"spine": [], "trail": []}
    section = None
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current and section in ("spine", "trail"):
            data[section].append(current)
        current = None

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if re.match(r"^st:\s*", line):
            flush()
            data["st"] = line.split(":", 1)[1].strip().strip('"')
            section = None
            continue
        if line.startswith("aims:"):
            flush()
            section = "aims"
            rest = line.split(":", 1)[1].strip()
            if rest in (">-", "|", ">"):
                data["aims"] = ""
            else:
                data["aims"] = rest.strip('"')
            continue
        if section == "aims" and (line.startswith("  ") or line.startswith("\t")):
            data["aims"] = (data.get("aims") or "") + (" " if data.get("aims") else "") + line.strip()
            continue
        if line.strip() == "spine:":
            flush()
            section = "spine"
            continue
        if line.strip() == "trail:":
            flush()
            section = "trail"
            continue
        m = re.match(r"^\s+-\s+id:\s*(.+)$", line)
        if m and section in ("spine", "trail"):
            flush()
            current = {"id": m.group(1).strip().strip('"')}
            continue
        m = re.match(r"^\s+(\w+):\s*(.*)$", line)
        if m and current is not None:
            key, val = m.group(1), m.group(2).strip()
            if val in ("null", "~", ""):
                current[key] = None
            elif (val.startswith('"') and val.endswith('"')) or (
                val.startswith("'") and val.endswith("'")
            ):
                current[key] = val[1:-1]
            else:
                try:
                    if "." in val:
                        current[key] = float(val)
                    else:
                        current[key] = int(val)
                except ValueError:
                    current[key] = val
            continue
    flush()
    return data
