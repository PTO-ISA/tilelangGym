"""IPC and instruction charts for the TileLang buffer GEMV and the CCE GEMV."""

import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

NK = Path(__file__).resolve().parents[1] / "nk"
sys.path.insert(0, str(NK))
import plot_charts as pc

OUT = Path(__file__).resolve().parents[1] / "plots"
CCE_BIN = pc.newest_bin(Path(__file__).resolve().parent)
TL_BIN = pc.newest_bin(NK / "runs" / "ub-n4096-k32")


def shown_rows(ops):
    counts = Counter(op for _, op in ops)
    return [
        (op.removeprefix("SIMT_"), {op})
        for op in sorted(counts, key=lambda name: (-counts[name], name))
        if op not in pc.OFF_ROWS
    ]


def ipc(ticks, ops, width=20):
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
    return xs, ys


def first_launch(path):
    rows = [row for row in pc.decode_instr_logs_from_file(path) if row.core == 0 and row.sub == 1]
    issues, dones = {}, {}
    for row in rows:
        op = pc.opcode(row.dec)
        if op != "VF_SIMT":
            continue
        key = (row.pc, op)
        if row.is_popped == 1 and key not in issues:
            issues[key] = row.time
        elif row.is_popped == 0:
            dones[key] = row.time
    spans = sorted(dones[key] - start for key, start in issues.items() if key in dones)
    return spans[0]


def draw(path, title, out_name, xlim=None):
    ticks, ops = pc.windows(path, "VF_SIMT")
    rows = shown_rows(ops)
    view = ticks if xlim is None else min(ticks, xlim)
    fig = plt.figure(figsize=(10.6, 0.34 * max(len(rows), 1) + 2.1), dpi=160, facecolor="white")
    fig.suptitle(title, x=0.01, ha="left", y=0.98, fontsize=13, fontweight="semibold")
    grid = GridSpec(2, 1, figure=fig, height_ratios=[2.15, 0.95], hspace=0.08, top=0.88, bottom=0.12, left=0.16, right=0.90)
    top, bot = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    n = len(rows)
    for i in range(n):
        top.axhspan(i - 0.5, i + 0.5, color="#f7f8f9" if i % 2 == 0 else "white", linewidth=0)
    for i, (label, names) in enumerate(rows):
        y = n - 1 - i
        ts = [t for t, name in ops if name in names and t < view]
        top.vlines(ts, y - 0.34, y + 0.34, color=pc.PALETTE[i % len(pc.PALETTE)], linewidth=0.8)
        top.text(view * 1.01, y, f"×{len(ts)}", va="center", ha="left", fontsize=8.5, color="#444", clip_on=False)
    top.set_yticks(range(n), [label for label, _ in reversed(rows)])
    top.set_xlim(0, view)
    top.set_ylim(-0.55, n - 0.45)
    top.tick_params(axis="x", labelbottom=False, length=0)
    xs, ys = ipc(ticks, ops)
    xs = [x for x in xs if x <= view]
    ys = ys[: len(xs)]
    bot.fill_between(xs, ys, color="#5c4e78", alpha=0.14, linewidth=0)
    bot.plot(xs, ys, color="#5c4e78", linewidth=1.15)
    bot.set_xlim(0, view)
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
    fig.savefig(out, dpi=160, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"{out}  VF {ticks}  IPC {len(ops) / ticks:.2f}")


def main():
    OUT.mkdir(exist_ok=True)
    draw(TL_BIN, "TileLang buffer, N = 4096, K = 32, 256 threads, VF 132303", "cmp_tilelang.png")
    draw(CCE_BIN, "CCE, N = 4096, K = 32, 256 threads, VF 55100", "cmp_cce.png")
    tl_first = first_launch(TL_BIN)
    cce_first = first_launch(CCE_BIN)
    draw(TL_BIN, f"TileLang buffer, first launch, {tl_first} ticks", "cmp_tilelang_zoom.png", xlim=tl_first)
    draw(CCE_BIN, f"CCE, first launch, {cce_first} ticks", "cmp_cce_zoom.png", xlim=cce_first)


if __name__ == "__main__":
    main()
