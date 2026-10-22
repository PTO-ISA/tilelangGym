# VMI DSL VL=128 G=8 VAND — primary numbers

**Date:** 2026-10-22 Asia/Hong_Kong  
See [`ANALYSIS.md`](ANALYSIS.md).

| arm | el/cyc | VF |
|---|---:|---:|
| **VL128 G=8 VAND** | **33.44** | 245 |
| prior VL128 VMUL abs | 20.79 | 394 |
| FA dual64 VAND | 11.89 | 689 |

Abs: i32 `vinterpret_cast` + `vand(…, 0x7FFFFFFF)` + cast-back. No RV_VMUL in mix.
