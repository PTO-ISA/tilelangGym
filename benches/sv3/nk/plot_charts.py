"""SV3 charts for the contiguous-K runs."""

import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

CANN = "/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/python/site-packages"
sys.path.insert(0, CANN)
from cannsim.core.public.instr_decoder import decode_instr_logs_from_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "plots"
MN = re.compile(r"\(ID:\s*(\d+)\)\s+([A-Z0-9_]+)")
PALETTE = [
    "#1f4e79", "#c45c26", "#1a7a7a", "#e07a1f", "#1b5e20", "#5c4e78",
    "#3d5a80", "#7cb342", "#6b4c7a", "#8fb4d4", "#888888", "#b5b5b5",
]
OFF_ROWS = {
    "SIMT_IADD", "SIMT_IADD_I", "SIMT_LEA", "SIMT_S2R", "SIMT_SHFI",
    "SIMT_BRANCH", "SIMT_ISETP", "SIMT_ISETP_I", "SIMT_END", "SIMT_MOVI",
    "SIMT_MOV", "SIMT_PLOP3", "SIMT_FSETP", "SIMT_LOP3",
    "RV_SLDI", "RV_SSTI", "RV_VMOV", "RV_SMOV", "RV_SMOVI", "RV_SMOVK",
    "RV_PSET", "RV_SEND", "RV_VCI", "RV_VLOOP",
    "RV_SADD", "RV_SSHLI", "RV_SZEROEXT",
}


def opcode(dec):
    match = MN.search(dec or "")
    return match.group(2) if match else None


def newest_bin(case: Path):
    bins = sorted(case.glob("**/instr.bin"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not bins:
        raise FileNotFoundError(case)
    return bins[0]


def windows(path: Path, vf_name: str):
    rows = [row for row in decode_instr_logs_from_file(path) if row.core == 0 and row.sub == 1]
    issues, dones = {}, {}
    for row in rows:
        op = opcode(row.dec)
        if op != vf_name:
            continue
        key = (row.pc, op)
        if row.is_popped == 1 and key not in issues:
            issues[key] = row.time
        elif row.is_popped == 0:
            dones[key] = row.time
    spans = []
    for key, start in issues.items():
        if key in dones:
            spans.append((start, dones[key]))
    spans.sort()
    ops = []
    cursor = 0
    for start, done in spans:
        for row in rows:
            if row.is_popped != 1:
                continue
            op = opcode(row.dec)
            if op is None or op == vf_name or not (start <= row.time <= done):
                continue
            ops.append((cursor + row.time - start, op))
        cursor += done - start
    return cursor, ops


def draw_instr(case: Path, title: str, out_name: str, kind: str):
    vf_name = "VF" if kind == "simd" else "VF_SIMT"
    ticks, ops = windows(newest_bin(case), vf_name)
    counts = Counter(op for _, op in ops)
    shown = [
        (op.removeprefix("SIMT_").removeprefix("RV_"), {op})
        for op in sorted(counts, key=lambda name: (-counts[name], name))
        if op not in OFF_ROWS
    ]
    fig = plt.figure(figsize=(10.6, 0.34 * max(len(shown), 1) + 2.1), dpi=200, facecolor="white")
    fig.suptitle(title, x=0.01, ha="left", y=0.98, fontsize=13, fontweight="semibold")
    grid = GridSpec(2, 1, figure=fig, height_ratios=[2.15, 0.95], hspace=0.08, top=0.88, bottom=0.12, left=0.18, right=0.90)
    top, bot = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    n = len(shown)
    for i in range(n):
        top.axhspan(i - 0.5, i + 0.5, color="#f7f8f9" if i % 2 == 0 else "white", linewidth=0)
    for i, (label, names) in enumerate(shown):
        y = n - 1 - i
        ts = [t for t, name in ops if name in names]
        top.vlines(ts, y - 0.34, y + 0.34, color=PALETTE[i % len(PALETTE)], linewidth=0.7)
        top.text(ticks * 1.015, y, f"×{len(ts)}", va="center", ha="left", fontsize=8.5, color="#444", clip_on=False)
    top.set_yticks(range(n), [label for label, _ in reversed(shown)])
    top.set_xlim(0, ticks)
    top.set_ylim(-0.55, n - 0.45)
    top.tick_params(axis="x", labelbottom=False, length=0)
    width = 20
    bins = max((ticks + width - 1) // width, 1)
    bucket = [0] * bins
    for t, _ in ops:
        if 0 <= t < ticks:
            bucket[t // width] += 1
    xs, ys = [], []
    for i, count in enumerate(bucket):
        lo, hi = i * width, min(ticks, (i + 1) * width)
        xs.append(lo + (hi - lo) / 2)
        ys.append(count / (hi - lo) if hi > lo else 0)
    bot.fill_between(xs, ys, color="#5c4e78", alpha=0.14, linewidth=0)
    bot.plot(xs, ys, color="#5c4e78", linewidth=1.15)
    bot.set_xlim(0, ticks)
    bot.set_ylim(0, max(4, (max(ys) if ys else 1) + 0.4))
    bot.set_ylabel("IPC")
    bot.set_xlabel("tick inside the VF window")
    bot.yaxis.grid(True, color="#eef0f2", linewidth=0.7)
    bot.set_axisbelow(True)
    for ax in (top, bot):
        ax.tick_params(length=0, labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    out = OUT / out_name
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"VF {ticks}  issues {len(ops)}  IPC {len(ops) / ticks:.2f}  -> {out.name}")


def draw_n():
    ns = [128, 256, 512, 1024, 2048]
    thread = [19308, 36210, 75455, 155646, 570802]
    warp = [12461, 23708, 46810, 89806, 177125]
    fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=200)
    ax.plot(ns, thread, marker="o", color="#1f4e79", label="Thread")
    ax.plot(ns, warp, marker="o", color="#c45c26", label="Warp")
    ax.set_xticks(ns)
    ax.set_xlabel("N")
    ax.set_ylabel("VF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = OUT / "n_vf.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"-> {out.name}")


def main():
    OUT.mkdir(exist_ok=True)
    for stale in ("n_vf.png", "simd_instr.png", "simt_first_launch.png"):
        path = OUT / stale
        if path.exists():
            path.unlink()
    draw_n()
    draw_instr(
        HERE / "runs" / "simd-n4096-k32",
        "SIMD, N = 4096, K = 32",
        "simd_instr.png",
        "simd",
    )
    draw_instr(
        HERE / "runs" / "ub-n4096-k32",
        "Buffer SIMT, N = 4096, K = 32, 256 threads",
        "simt_first_launch.png",
        "simt",
    )


if __name__ == "__main__":
    main()
