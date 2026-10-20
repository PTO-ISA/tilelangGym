| tag | temps | VF_cyc | el/cyc | wall_us | LDS | STS | FADD | IADD | opsim | layout |
|-----|-------|-------:|-------:|--------:|----:|----:|-----:|-----:|-------|--------|
| sv1_e8192_t128_frag | frag | 4680 | 1.7504 | 2.6 | 128 | 64 | 256 | 192 | PASS maxabs=0.0 | RF:float t1[64],float t2[64],float t3[64] |
| sv1_e8192_t128_shared | shared | 7450 | 1.0996 | 4.14 | 128 | 256 | 256 | 384 | PASS maxabs=0.0 | no_RF_arrays; UB-pointer temps |
| sv1_e8192_t512_frag | frag | 3405 | 2.4059 | 1.89 | 128 | 64 | 256 | 192 | PASS maxabs=0.0 | RF:float t1[16],float t2[16],float t3[16] |
| sv1_e8192_t512_shared | shared | 6840 | 1.1977 | 3.8 | 128 | 256 | 256 | 384 | PASS maxabs=0.0 | no_RF_arrays; UB-pointer temps |
| sv1_e16384_t128_frag | frag | 9031 | 1.8142 | 5.02 | 256 | 128 | 512 | 384 | PASS maxabs=0.0 | RF:float t1[128],float t2[128],float t3[128] |
| sv1_e16384_t128_shared | shared |  |  |  | 0 | 0 | 0 | 0 | FAIL maxabs=492.489990234375 | no_RF_arrays; UB-pointer temps |
