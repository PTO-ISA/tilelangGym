"""SV9 charts. Instruction rows from the E=512 keep traces."""

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

import re

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
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
    "RV_PSET", "RV_SEND", "RV_VCI",
}


def opcode(dec):
    match = MN.search(dec or "")
    return match.group(2) if match else None


def newest_vf_bin(case: Path, vf_name):
    candidates = sorted(case.glob("**/instr.bin"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        rows = [row for row in decode_instr_logs_from_file(path) if row.core == 0 and row.sub == 1]
        if any(row.is_popped == 1 and opcode(row.dec) == vf_name for row in rows):
            return path
    raise FileNotFoundError(f"no {vf_name} trace under {case}")


def window(path: Path, vf_name):
    rows = [row for row in decode_instr_logs_from_file(path) if row.core == 0 and row.sub == 1]
    issues = [row.time for row in rows if row.is_popped == 1 and opcode(row.dec) == vf_name]
    dones = [row.time for row in rows if row.is_popped == 0 and opcode(row.dec) == vf_name]
    issue, done = issues[0], dones[-1]
    ticks = done - issue
    ops = []
    for row in rows:
        if row.is_popped != 1:
            continue
        op = opcode(row.dec)
        if op is None or op == vf_name:
            continue
        if issue <= row.time <= done:
            ops.append((row.time - issue, op))
    return ticks, ops


def draw_instr(case: str, title: str, out_name: str, kind="simt"):
    vf_name = "VF" if kind == "simd" else "VF_SIMT"
    ticks, ops = window(newest_vf_bin(HERE / case, vf_name), vf_name)
    counts = Counter(op for _, op in ops)
    shown = [(op.removeprefix("SIMT_").removeprefix("RV_"), {op})
             for op in sorted(counts, key=lambda name: (-counts[name], name))
             if op not in OFF_ROWS]
    fig = plt.figure(figsize=(10.6, 0.34 * len(shown) + 2.1), dpi=200, facecolor="white")
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
    bins = (ticks + width - 1) // width
    bucket = [0] * bins
    for t, _ in ops:
        if 0 <= t < ticks:
            bucket[t // width] += 1
    xs, ys = [], []
    for i, count in enumerate(bucket):
        lo, hi = i * width, min(ticks, (i + 1) * width)
        xs.append(lo + (hi - lo) / 2)
        ys.append(count / (hi - lo))
    bot.fill_between(xs, ys, color="#5c4e78", alpha=0.14, linewidth=0)
    bot.plot(xs, ys, color="#5c4e78", linewidth=1.15)
    bot.set_xlim(0, ticks)
    bot.set_ylim(0, max(4, max(ys) + 0.4))
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


def main():
    OUT.mkdir(exist_ok=True)
    draw_instr(
        "runs/simt/keep-e512-k8-t32",
        "SIMT keep, E = 512, K = 8, 32 threads",
        "simt_e512_instr.png",
    )
    draw_instr(
        "runs/simd/resident/e512-k8",
        "SIMD keep, E = 512, K = 8",
        "simd_e512_instr.png",
        kind="simd",
    )


if __name__ == "__main__":
    main()
