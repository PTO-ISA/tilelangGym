# SoftLib dual64 absmax VAND — mid-point rolling IPC

**Style:** W=20, left-open right-closed `(t−W, t]` (same as reduce-max / absmax-VMUL).

| | |
|---|---:|
| **midpoint IPC** (`t_mid=2743`) | **2.50** |
| **midpoint input el/cyc** | **41.6** (`#VLD·64 / 20`) |
| mid VAND/cyc | 0.60 |
| mid VCGMAX/cyc | 0.65 |
| mean IPC body | **2.41** |
| mean el_L over VLD-win | **43.3** |
| peak el_L | **83.2** @2673 |
| firstVLD → lastVLD → lastVST | 2654 → 2832 → 2871 |
| whole-kernel el/cyc | **31.39** (VF 261) |

Compare: reduce-max mid **51.2** el/cyc · absmax-VMUL mid **25.6** · **VAND mid 41.6**.
