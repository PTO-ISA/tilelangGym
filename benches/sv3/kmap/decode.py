"""Decode one instr.bin. Prints VF, IPC, el/cyc, mac/cyc, and spill counts."""

import re
import sys
from pathlib import Path

from cannsim.core.public.instr_decoder import decode_instr_logs_from_file

OPCODE = re.compile(r"\(ID:\s*(\d+)\)\s+([A-Z0-9_]+)")
SPILL = ("SIMT_LDK", "SIMT_STK", "RV_SLDI", "RV_SSTI", "RV_VMOV")


def main():
    path, rows, cols, K, kind = sys.argv[1:]
    rows, cols, K = int(rows), int(cols), int(K)
    entries = [
        e
        for e in decode_instr_logs_from_file(Path(path))
        if e.core == 0 and e.sub == 1
    ]
    events = []
    for e in entries:
        match = OPCODE.search(e.dec)
        if match is None:
            continue
        events.append((e.time, e.is_popped, e.pc, int(match.group(1)), match.group(2)))

    name = "VF" if kind == "simd" else "VF_SIMT"
    issues = {}
    dones = {}
    for time, popped, pc, ident, op in events:
        key = (pc, ident, op)
        if popped == 1 and key not in issues:
            issues[key] = time
        elif popped == 0:
            dones[key] = time

    windows = []
    for key, start in issues.items():
        if key[2] != name or key not in dones:
            continue
        windows.append((start, dones[key]))
    if not windows:
        print("NO_VF")
        return 1
    vf = sum(done - start for start, done in windows)
    inside = 0
    spills = {op: 0 for op in SPILL}
    for time, popped, _pc, _ident, op in events:
        if popped != 1:
            continue
        if any(start <= time <= done for start, done in windows) and op != name:
            inside += 1
        if op in spills:
            spills[op] += 1
    cells = rows * cols
    ipc = inside / vf if vf else 0
    print(
        f"launches={len(windows)} VF={vf} el/cyc={cells / vf:.2f} "
        f"mac/cyc={cells * K / vf:.2f} IPC={ipc:.2f} "
        + " ".join(f"{op}={spills[op]}" for op in SPILL)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
