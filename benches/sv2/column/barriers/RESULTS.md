# SV2.A schedule barriers

16×1024, 1024 threads. Every launch passed, maxabs 1.192e-07.

An empty `asm volatile("" ::: "memory")` inside this VF is deleted. Those six sources compiled to one `kernel.so`, so the first table was six runs of one kernel.

Each cut below is its own kernel. The cut flushes that stage's values through a volatile access, which this compiler keeps. The flush is local memory traffic: `LDK` and `STK`. The vector mix does not change. `LDS`, `STS`, `FMUL`, and `FMNMX` stay 512 / 512 / 512 / 480.

| kernel | cut after | VF | el/cyc | IPC | issues | LDK | STK |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fused | stock loop, no cut | 1690 | 9.69 | 1.95 | 3296 | 0 | 0 |
| sep | five stages, no cut | 1725 | 9.50 | 1.91 | 3296 | 0 | 0 |
| cut-load | the 16 loaded values | 7694 | 2.13 | 0.63 | 4832 | 512 | 1024 |
| cut-max | the max | 1991 | 8.23 | 1.70 | 3392 | 32 | 64 |
| cut-inv | the inverse | 1971 | 8.31 | 1.74 | 3424 | 32 | 64 |
| cut-mul | the 16 products | 6403 | 2.56 | 0.75 | 4832 | 512 | 1024 |
| cut-mem | the loads and the products | 9972 | 1.64 | 0.64 | 6368 | 1024 | 2048 |
| cut-all | load, max, inverse, multiply | 11187 | 1.46 | 0.59 | 6592 | 1088 | 2176 |

`LDK` is one per flushed value per warp. `STK` is two per flushed value per warp. Flushing one value, the max or the inverse, costs about 300 ticks. Flushing 16 values costs thousands.
