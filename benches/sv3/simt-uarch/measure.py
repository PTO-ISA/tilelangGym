"""Bytes per cycle for one opcode inside the VF window.

Usage: measure.py <instr.bin> simd|simt <opcode> <useful-bytes>
Issue span is the last issue minus the first issue.
Active cycles are completion minus issue for one instruction. Popped 1 is the issue, popped 0 is the completion.
"""

import re
import sys
from pathlib import Path

from cannsim.core.public.instr_decoder import decode_instr_logs_from_file

OPCODE = re.compile(r"\(ID:\s*(\d+)\)\s+([A-Z0-9_]+)")


def main():
    path, kind, focus, nbytes = sys.argv[1:]
    nbytes = int(nbytes)
    vf_name = "VF" if kind == "simd" else "VF_SIMT"
    entries = [e for e in decode_instr_logs_from_file(Path(path)) if e.core == 0 and e.sub == 1]
    events = []
    for e in entries:
        match = OPCODE.search(e.dec or "")
        if match:
            events.append((e.time, e.is_popped, e.pc, int(match.group(1)), match.group(2)))
    issues, dones = {}, {}
    for time, popped, pc, ident, op in events:
        key = (pc, ident, op)
        if popped == 1 and key not in issues:
            issues[key] = time
        elif popped == 0:
            dones[key] = time
    windows = sorted((issues[key], dones[key]) for key in issues if key[2] == vf_name and key in dones)

    def inside(time):
        return any(start <= time <= done for start, done in windows)

    times = []
    active = []
    for time, popped, pc, ident, op in events:
        if op != focus or not inside(time):
            continue
        key = (pc, ident, op)
        if popped == 1:
            times.append(time)
        elif key in issues:
            active.append(dones[key] - issues[key])
    if len(times) < 2 or not active:
        print(f"{focus} count={len(times)} NO_SPAN")
        return 1
    span = max(times) - min(times)
    per = nbytes / len(times)
    active.sort()
    mid = active[len(active) // 2]
    print(
        f"{focus} count={len(times)} span={span} "
        f"B/issue={per:.1f} B/cyc={nbytes / span:.2f} issue/cyc={len(times) / span:.3f} "
        f"active={mid} B/active={per / mid:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
