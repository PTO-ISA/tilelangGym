# Operator kernels

N=16384, 1024 threads, 16 rows. The head-to-head SIMT row is VF 2363, 6.93 el/cyc. el/cyc below is 16384/VF.

`max` and `fdiv` start from an LDS. The max loads the 16 rows. The divide loads the 16 rows and divides each by the column's first value. `redux` loads the first row and takes the max across the 1024 threads.

| kernel | VF | el/cyc | IPC | inside |
| --- | ---: | ---: | ---: | --- |
| load | 1496 | 10.95 | 1.48 | 512 LDS, 512 STS |
| max | 879 | 18.64 | 2.00 | 512 LDS, 512 max, 32 STS |
| redux | 1026 | 15.97 | 1.03 | 64 REDUX, 128 BAR, 96 LDS |
| fdiv | 1572 | 10.42 | 1.75 | 512 LDS, 512 FDIV, 512 STS |
| sts | 773 | 21.20 | 1.61 | 512 STS, no LDS |

| kernel | operator | ticks | el/cyc |
| --- | --- | --- | ---: |
| load | LDS | 55–1224 | 14.02 |
| max | max | 209–846 | 25.72 |
| redux | REDUX | 98–624 | 1.95 |
| fdiv | FDIV | 187–1488 | 12.59 |
| sts | STS | 88–747 | 24.86 |
