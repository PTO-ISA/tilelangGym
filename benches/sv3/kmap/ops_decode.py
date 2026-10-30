"""Decode one ops instr.bin. Prints whole-kernel VF and the operator window.

Usage: ops_decode.py instr.bin load|ffma|sts
"""

import re
import sys
from pathlib import Path

from cannsim.core.public.instr_decoder import decode_instr_logs_from_file

from ops import COLS, K, ROWS

OPCODE = re.compile(r"\(ID:\s*(\d+)\)\s+([A-Z0-9_]+)")
OP = {
    "load": {"SIMT_LDG"},
    "ffma": {"SIMT_FFMA", "SIMT_FFMA_I"},
    "sts": {"SIMT_STS"},
}
# Elements credited to the operator window.
ELEM = {
    "load": ROWS * COLS * K,  # one A load per cell per k
    "ffma": ROWS * COLS * K,  # one FFMA per cell per k
    "sts": ROWS * COLS,  # one store per cell
}


def main():
    path, name = sys.argv[1], sys.argv[2]
    targets = OP[name]
    label = next(iter(targets)).removeprefix("SIMT_").split("_")[0]
    if name == "ffma":
        label = "FFMA"
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

    issues, dones = {}, {}
    for time, popped, pc, ident, op in events:
        key = (pc, ident, op)
        if popped == 1 and key not in issues:
            issues[key] = time
        elif popped == 0:
            dones[key] = time
    windows = []
    for key, start in issues.items():
        if key[2] != "VF_SIMT" or key not in dones:
            continue
        windows.append((start, dones[key]))
    if not windows:
        print("NO_VF")
        return 1
    start, done = min(windows)
    vf = done - start
    inside = 0
    op_times = []
    counts = {}
    for time, popped, _pc, _ident, op in events:
        if popped != 1:
            continue
        if not (start <= time <= done) or op == "VF_SIMT":
            continue
        inside += 1
        counts[op] = counts.get(op, 0) + 1
        if op in targets:
            op_times.append(time)
    cells = ROWS * COLS
    ipc = inside / vf if vf else 0
    el = cells / vf if vf else 0
    key_ops = ", ".join(f"{n} {op.removeprefix('SIMT_')}" for op, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if op.startswith("SIMT_"))
    print(f"VF={vf} el/cyc={el:.2f} IPC={ipc:.2f} opcodes={key_ops}")
    if not op_times:
        print(f"WINDOW {label}=none")
        return 0
    lo, hi = min(op_times), max(op_times)
    span = hi - lo if hi > lo else 1
    win_el = ELEM[name] / span
    print(f"WINDOW {label}={lo - start}-{hi - start} el/cyc={win_el:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
