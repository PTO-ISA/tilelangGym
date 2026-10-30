# Schedule fence

N=16384, 1024 threads, 16 rows. Every row is the full block scale. `fused` and `sep` are one binary (`72e7c4c33d57b0a49d24367ea587e07e`). The cut is a volatile flush of that stage's values. All eight launches PASS, maxabs 5.960e-08.

`LDS` 576, `STS` 576, `FDIV` 512, `FMNMX` 480, `FMNMX_I` 64, `REDUX` 64, `BAR` 128 on every row. `LDK` and `STK` are the flush.

| fence | after | VF | el/cyc | IPC | issues | LDK | STK |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fused |  | 2360 | 6.94 | 1.69 | 4000 | 0 | 0 |
| sep |  | 2360 | 6.94 | 1.69 | 4000 | 0 | 0 |
| load | the 16 loads | 8355 | 1.96 | 0.66 | 5536 | 512 | 1024 |
| max | the max | 3179 | 5.15 | 1.29 | 4096 | 32 | 64 |
| redux | the block scale | 2368 | 6.92 | 1.73 | 4096 | 32 | 64 |
| div | the 16 quotients | 7158 | 2.29 | 0.77 | 5536 | 512 | 1024 |
| mem | the loads and the quotients | 14212 | 1.15 | 0.50 | 7072 | 1024 | 2048 |
| all | load, max, redux, and divide | 15181 | 1.08 | 0.48 | 7264 | 1088 | 2176 |
