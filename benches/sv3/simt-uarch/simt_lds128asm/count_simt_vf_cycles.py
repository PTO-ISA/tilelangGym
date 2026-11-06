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
"""Count SIMT VF cycles and auto-derive SIMT_LDS / SIMT_STS bw from camodel logs.

simt_vf_cycles = retire_cycle - pop_cycle of matching PUSHQ VF_SIMT IDs.

Instruction table (count / cycles / avg / busy / real / bw / ipc):
  cycles  = sum over issues of (retire - pop); overlapping warps counted separately
  busy    = union length of [pop, retire) per opcode
  real    = overlap-adjusted share of each cycle across in-flight warps
  bw      = moved_bytes / real   (SIMT_LDS and SIMT_STS only)
  ipc     = count / real for compute ops (none in this pure-copy kernel)

LDS/STS moved_bytes are derived from the log (no hard-coded tile size):

1. Prefer camodel data-register lists
   - SIMT_LDS: [Rd][Rd1]… → nregs * 4 B/lane  (b32→1, b64→2, b128→4)
   - SIMT_STS: [Rs][Rs1]… likewise
2. Else Binary bits [35:32]
   - RVECLD (LDS): 8→4B, 10→8B, 12→16B
   - RVECST (STS): 4→4B,  6→8B,  8→16B

    moved_bytes = sum over issues of (lane_bytes * WARP_SIZE)

For a single full-tile b128 copy that equals 64*128*4. Do not use
`issues * 16` (misses the 32 lanes).

Usage:
    python3 count_simt_vf_cycles.py [log_dir]

Default log_dir is <this_script>/build/bin.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

WARP_SIZE = 32
COMPUTE_INSTRS = set()
MEM_NAMES = ("SIMT_LDS", "SIMT_STS")

VF_LINE = re.compile(r"\[(\d+)\].*PUSHQ.*\(ID:\s*(\d+)\).*VF_SIMT")
INSTR_LINE = re.compile(
    r"\[(\d+)\].*Binary:\s*(0x[0-9a-fA-F]+).*\(ID:\s*(\d+)\)\s+((?:SIMT_\S+|VF_SIMT))"
)
INSTR_LINE_NO_BIN = re.compile(r"\[(\d+)\].*\(ID:\s*(\d+)\)\s+((?:SIMT_\S+|VF_SIMT))")
DATA_REGS_LDS = re.compile(r"\[Rd\d*:\d+\|R\]")
DATA_REGS_STS = re.compile(r"\[Rs\d*:\d+\|R\]")

# bits [35:32] of the 64-bit Binary field
LDS_ACCESS_BYTES = {0: 1, 2: 1, 4: 2, 6: 2, 8: 4, 10: 8, 12: 16}
STS_ACCESS_BYTES = {4: 4, 6: 8, 8: 16}  # RVECST coding ≠ RVECLD


def vf_records(path: Path):
    records = []
    for line in path.read_text(errors="replace").splitlines():
        match = VF_LINE.search(line)
        if match:
            records.append((int(match.group(2)), int(match.group(1))))
    return records


def lane_bytes(name: str, line: str, binary: int | None) -> int:
    """Bytes moved per lane for one SIMT_LDS / SIMT_STS issue."""
    if name == "SIMT_LDS":
        nregs = len(DATA_REGS_LDS.findall(line))
        if nregs:
            return nregs * 4
        if binary is not None:
            code = (binary >> 32) & 0xF
            if code in LDS_ACCESS_BYTES:
                return LDS_ACCESS_BYTES[code]
    elif name == "SIMT_STS":
        nregs = len(DATA_REGS_STS.findall(line))
        if nregs:
            return nregs * 4
        if binary is not None:
            code = (binary >> 32) & 0xF
            if code in STS_ACCESS_BYTES:
                return STS_ACCESS_BYTES[code]
    return 0


def instr_records(path: Path):
    """id -> (cycle, name, lane_bytes)."""
    records = {}
    for line in path.read_text(errors="replace").splitlines():
        match = INSTR_LINE.search(line)
        if match:
            cycle = int(match.group(1))
            binary = int(match.group(2), 16)
            instr_id = int(match.group(3))
            name = match.group(4)
            records[instr_id] = (cycle, name, lane_bytes(name, line, binary))
            continue
        match = INSTR_LINE_NO_BIN.search(line)
        if match:
            cycle = int(match.group(1))
            instr_id = int(match.group(2))
            name = match.group(3)
            records[instr_id] = (cycle, name, lane_bytes(name, line, None))
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
    mem_bytes = Counter()
    lane_hist = {name: Counter() for name in MEM_NAMES}
    for instr_id, (cycle, name, lb) in pop_recs.items():
        if not pop_cycle <= cycle <= retire_cycle:
            continue
        if instr_id not in retire_recs:
            raise KeyError(instr_id)
        retire = retire_recs[instr_id][0]
        counts[name] += 1
        cycles[name] += retire - cycle
        intervals.setdefault(name, []).append((cycle, retire))
        if name in MEM_NAMES and lb:
            mem_bytes[name] += lb * WARP_SIZE
            lane_hist[name][lb] += 1
    return counts, cycles, intervals, mem_bytes, lane_hist


def format_width_hist(hist: Counter) -> str:
    if not hist:
        return "?"
    return ", ".join(f"{b}B×{n}" for b, n in sorted(hist.items()))


def print_table(counts, cycles, intervals, mem_bytes):
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
    bw_bytes = {name: mem_bytes[name] for name in MEM_NAMES if mem_bytes.get(name)}
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
        if name not in bw_bytes or real_by_name[name] == 0:
            return "-"
        return f"{bw_bytes[name] / real_by_name[name]:.2f}"

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
                counts, cycles, intervals, mem_bytes, lane_hist = simt_stats(
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
            if mem_bytes:
                parts = []
                for name in MEM_NAMES:
                    if name not in mem_bytes:
                        continue
                    nbytes = mem_bytes[name]
                    n = counts[name]
                    parts.append(
                        f"{name} {nbytes} B "
                        f"[{format_width_hist(lane_hist[name])}] "
                        f"({nbytes // n} B/issue)"
                    )
                print(f"moved: {', '.join(parts)}")
            print()
            print_table(counts, cycles, intervals, mem_bytes)
            print()
            found = True

    if not found:
        print(f"no PUSHQ VF_SIMT lines under {log_dir}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
