# Operator kernels

16×1024, 1024 threads. The same shape as the head-to-head SIMT row (VF 1690, 9.69 el/cyc). Each kernel is one operator, plus the load or store that operator needs so the compiler keeps it. el/cyc is 16384/VF.

| kernel | VF | el/cyc | IPC | what is inside |
| --- | ---: | ---: | ---: | --- |
| load | 1496 | 10.95 | 1.48 | 512 LDS, 512 STS. The 16 values are written back. |
| max | 879 | 18.64 | 2.00 | 512 LDS, 512 max, 32 STS. One max per column. |
| fdiv | 629 | 26.05 | 0.61 | 32 LDS, 32 FDIV, 32 STS. One inverse per column. |
| fmul | 1551 | 10.56 | 1.75 | 512 LDS, 512 FMUL, 512 STS. |
| sts | 773 | 21.20 | 1.61 | 512 STS of the constant 1. No LDS. |

The 26.05 on `fdiv` divides the whole tile by a kernel that only computes 1024 inverses. Those inverses alone are 1024/629 = 1.63 per cycle.

Issue window of the named operator, in ticks from the start of the VF. The rate is the elements that operator issues, divided by the window.

| kernel | operator | ticks | el/cyc in that window |
| --- | --- | --- | ---: |
| load | LDS | 55–1224 | 14.02 |
| max | max | 209–846 | 25.72 |
| fdiv | FDIV | 89–511 | 2.43 |
| fmul | FMUL | 183–1497 | 12.47 |
| sts | STS | 88–747 | 24.86 |
