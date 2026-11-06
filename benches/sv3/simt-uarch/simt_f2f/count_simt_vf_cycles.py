import sys
from pathlib import Path

code = (Path(__file__).resolve().parents[1] / "simt_add" / "count_simt_vf_cycles.py").read_text()
code = code.replace(
    'TILE_BYTES = 64 * 128 * 4\nBW_BYTES = {"SIMT_LDS": TILE_BYTES * 2, "SIMT_STS": TILE_BYTES}\nCOMPUTE_INSTRS = {"SIMT_FADD"}',
    'HALF_BYTES = 64 * 128 * 2\nFLOAT_BYTES = 64 * 128 * 4\nBW_BYTES = {"SIMT_LDS": HALF_BYTES, "SIMT_STS": FLOAT_BYTES}\nCOMPUTE_INSTRS = {"SIMT_F2F"}',
)
exec(compile(code, "count_simt_vf_cycles.py", "exec"), {"__name__": "__main__", "__file__": __file__})
