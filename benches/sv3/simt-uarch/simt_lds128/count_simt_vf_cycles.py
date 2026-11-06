#!/usr/bin/env python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------
"""Count SIMT VF cycles and SIMT instruction mix from camodel veccore logs.

simt_vf_cycles = retire_cycle - pop_cycle

retire_cycle is the [CYCLE] on "PUSHQ ... VF_SIMT" in instr_log.dump.
pop_cycle is the [CYCLE] on "PUSHQ ... VF_SIMT" in instr_popped_log.dump.
The two lines are paired by instruction ID.

SIMT instruction counts come from instr_popped_log.dump, between that VF's
pop cycle and retire cycle. Each issue is paired with instr_log.dump by ID.

cycles is the sum of retire_cycle - pop_cycle. Warps are counted separately,
so that total is larger than simt_vf_cycles. avg is that sum divided by the
issue count.

busy is the length of the union of [pop, retire) intervals. Overlapping warps
of the same instruction count once. The busy total merges every SIMT_*
interval, so different instructions in flight at the same time also count once.
VF_SIMT is left out of that total because its interval is the whole VF.

real is the overlap-adjusted execution cycles of each SIMT instruction. A cycle
is split across the warps in flight, in proportion to how many of those warps
are executing that instruction. The real total sums those shares and matches
the busy total.

bw is bytes per cycle for SIMT_LDS and SIMT_STS only, divided by that
instruction's real cycles. This kernel copies one tile: SIMT_LDS reads
64 * 128 * 4 bytes and SIMT_STS writes 64 * 128 * 4 bytes.

ipc is issue count divided by real cycles, and only for compute
instructions. This kernel is a pure copy, so every ipc is -.

Usage:
    python3 count_simt_vf_cycles.py [log_dir]

Default log_dir is <this_script>/build/bin.
"""

import re
import sys
from collections import Counter
from pathlib import Path

# Bytes moved by each memory instruction. One float 64x128 tile, copied as
# 128-bit values. SIMT_LDS is LDS.b128. The byte count is still the tile size,
# not issue_count * 16.
TILE_BYTES = 64 * 128 * 4
BW_BYTES = {"SIMT_LDS": TILE_BYTES, "SIMT_STS": TILE_BYTES}
COMPUTE_INSTRS = set()

VF_LINE = re.compile(r"\[(\d+)\].*PUSHQ.*\(ID:\s*(\d+)\).*VF_SIMT")
INSTR_LINE = re.compile(r"\[(\d+)\].*\(ID:\s*(\d+)\)\s+((?:SIMT_\S+|VF_SIMT))")


def vf_records(path):
    records = []
    for line in path.read_text(errors="replace").splitlines():
        match = VF_LINE.search(line)
        if match:
            records.append((int(match.group(2)), int(match.group(1))))
    return records


def instr_records(path):
    records = {}
    for line in path.read_text(errors="replace").splitlines():
        match = INSTR_LINE.search(line)
        if match:
            records[int(match.group(2))] = (int(match.group(1)), match.group(3))
    return records


def union_cycles(intervals):
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    start, end = ordered[0]
    for nxt, stop in ordered[1:]:
        if nxt <= end:
            end = max(end, stop)
        else:
            total += end - start
            start, end = nxt, stop
    return total + end - start


def real_cycles(intervals):
    events = []
    for name, spans in intervals.items():
        if name == "VF_SIMT":
            continue
        for start, end in spans:
            events.append((start, 1, name))
            events.append((end, -1, name))
    events.sort(key=lambda item: (item[0], 0 if item[1] < 0 else 1))

    active = Counter()
    real = Counter()
    index = 0
    prev = None
    while index < len(events):
        tick = events[index][0]
        if prev is not None and tick > prev and active:
            duration = tick - prev
            warps = sum(active.values())
            for name, inflight in active.items():
                real[name] += duration * inflight / warps
        while index < len(events) and events[index][0] == tick:
            _, delta, name = events[index]
            active[name] += delta
            if active[name] == 0:
                del active[name]
            index += 1
        prev = tick
    return real


def simt_stats(pop_recs, retire_recs, pop_cycle, retire_cycle):
    counts = Counter()
    cycles = Counter()
    intervals = {}
    for instr_id, (cycle, name) in pop_recs.items():
        if not pop_cycle <= cycle <= retire_cycle:
            continue
        if instr_id not in retire_recs:
            raise KeyError(instr_id)
        retire = retire_recs[instr_id][0]
        counts[name] += 1
        cycles[name] += retire - cycle
        intervals.setdefault(name, []).append((cycle, retire))
    return counts, cycles, intervals


def print_table(counts, cycles, intervals):
    busy = {name: union_cycles(spans) for name, spans in intervals.items()}
    real = real_cycles(intervals)
    real_total = sum(real.values())
    merged_total = union_cycles(
        [
            span
            for name, spans in intervals.items()
            if name != "VF_SIMT"
            for span in spans
        ]
    )
    name_width = max(
        [len("instruction"), *(len(name) for name in counts)] or [len("instruction")]
    )
    count_width = max(
        len("count"),
        *(len(str(n)) for n in counts.values()),
        len(str(sum(counts.values()))),
    )
    cycle_vals = [cycles[name] for name in counts]
    cycle_width = max(
        len("cycles"), *(len(str(n)) for n in cycle_vals), len(str(sum(cycle_vals)))
    )
    busy_vals = [busy[name] for name in counts]
    busy_width = max(
        len("busy"), *(len(str(n)) for n in busy_vals), len(str(merged_total))
    )
    issue_count = sum(counts.values())
    avg_by_name = {name: cycles[name] / counts[name] for name in counts}
    avg_total = sum(cycle_vals) / issue_count
    avg_texts = [f"{avg_by_name[name]:.2f}" for name in counts]
    avg_total_text = f"{avg_total:.2f}"
    avg_width = max(len("avg"), *(len(text) for text in avg_texts), len(avg_total_text))
    real_by_name = {
        name: float(busy[name]) if name == "VF_SIMT" else real[name] for name in counts
    }
    real_texts = [f"{real_by_name[name]:.2f}" for name in counts]
    real_total_text = f"{real_total:.2f}"
    real_width = max(
        len("real"), *(len(text) for text in real_texts), len(real_total_text)
    )

    def bw_text(name):
        if name not in BW_BYTES or real_by_name[name] == 0:
            return "-"
        return f"{BW_BYTES[name] / real_by_name[name]:.2f}"

    bw_texts = [bw_text(name) for name in counts]
    bw_width = max(len("bw"), *(len(text) for text in bw_texts), 1)

    def ipc_text(name, count):
        if name not in COMPUTE_INSTRS or real_by_name[name] == 0:
            return "-"
        return f"{count / real_by_name[name]:.2f}"

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    bw_by_name = {name: bw_text(name) for name, _count in ordered}
    ipc_by_name = {name: ipc_text(name, count) for name, count in ordered}
    ipc_width = max(len("ipc"), *(len(text) for text in ipc_by_name.values()), 1)
    header = (
        f"{'instruction':<{name_width}}  {'count':>{count_width}}  "
        f"{'cycles':>{cycle_width}}  {'avg':>{avg_width}}  "
        f"{'busy':>{busy_width}}  {'real':>{real_width}}  "
        f"{'bw':>{bw_width}}  {'ipc':>{ipc_width}}"
    )
    rule = (
        f"{'-' * name_width}  {'-' * count_width}  {'-' * cycle_width}  "
        f"{'-' * avg_width}  {'-' * busy_width}  {'-' * real_width}  "
        f"{'-' * bw_width}  {'-' * ipc_width}"
    )
    print(header)
    print(rule)
    for name, count in ordered:
        print(
            f"{name:<{name_width}}  {count:>{count_width}}  {cycles[name]:>{cycle_width}}  "
            f"{avg_by_name[name]:>{avg_width}.2f}  {busy[name]:>{busy_width}}  "
            f"{real_by_name[name]:>{real_width}.2f}  {bw_by_name[name]:>{bw_width}}  "
            f"{ipc_by_name[name]:>{ipc_width}}"
        )
    print(rule)
    print(
        f"{'total':<{name_width}}  {issue_count:>{count_width}}  "
        f"{sum(cycle_vals):>{cycle_width}}  {avg_total:>{avg_width}.2f}  "
        f"{merged_total:>{busy_width}}  {real_total:>{real_width}.2f}  "
        f"{'-':>{bw_width}}  {'-':>{ipc_width}}"
    )


def main(argv):
    root = Path(__file__).resolve().parent
    log_dir = Path(argv[1]) if len(argv) > 1 else root / "build" / "bin"
    if len(argv) > 2:
        print(f"Usage: {argv[0]} [log_dir]", file=sys.stderr)
        return 1
    if not log_dir.is_dir():
        print(f"log dir not found: {log_dir}", file=sys.stderr)
        return 1

    retire_logs = sorted(log_dir.glob("core*.veccore*.instr_log.dump"))
    if not retire_logs:
        print(f"no core*.veccore*.instr_log.dump under {log_dir}", file=sys.stderr)
        return 1

    found = False
    for retire_log in retire_logs:
        pop_log = retire_log.with_name(
            retire_log.name.replace("instr_log.dump", "instr_popped_log.dump")
        )
        if not pop_log.is_file():
            print(f"missing popped log: {pop_log}", file=sys.stderr)
            return 1

        core = retire_log.name[: -len(".instr_log.dump")]
        pop_by_id = dict(vf_records(pop_log))
        retire_rows = vf_records(retire_log)
        if not pop_by_id and not retire_rows:
            continue

        pop_recs = instr_records(pop_log)
        retire_recs = instr_records(retire_log)
        for instr_id, retire_cycle in retire_rows:
            if instr_id not in pop_by_id:
                print(
                    f"{core}: ID {instr_id} retired at {retire_cycle} but has no PUSHQ VF_SIMT in {pop_log.name}",
                    file=sys.stderr,
                )
                return 1
            pop_cycle = pop_by_id[instr_id]
            try:
                counts, cycles, intervals = simt_stats(
                    pop_recs, retire_recs, pop_cycle, retire_cycle
                )
            except KeyError as exc:
                print(
                    f"{core}: ID {exc.args[0]} popped inside VF {instr_id} but has no retire line in {retire_log.name}",
                    file=sys.stderr,
                )
                return 1
            vf_cycles = retire_cycle - pop_cycle
            print(
                f"{core} ID {instr_id}  pop {pop_cycle}  retire {retire_cycle}  simt_vf_cycles {vf_cycles}"
            )
            print()
            print_table(counts, cycles, intervals)
            print()
            found = True

    if not found:
        print(f"no PUSHQ VF_SIMT lines under {log_dir}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
