# SV1 float2 multi-TILE @ T=512 (2026-10-20 HKT)

**Pins:** cann_92b2 · Ascend950PR_9599 · \ST_VF_MODE=simt-asc\ · stock float2 (\max_vector_bits=64\, LDS.64-class) · kernel \sv1t_tiled_add.py\ (serial tiles × Parallel(TILE), add-in-shared).

**Naming:** **TILE8192** = E=8192, TILE=E (formerly called "untiled" / Parallel(E)). Same config as prior stream point T512=**3.715** / VF=**2205**.

## §5.1 primary — E=8192 TILE @ T=512

| TILE | VF_cyc | el/cyc | wall µs | LDS:FADD:STS | MRF live (unique Rn × T × 4B) |
|-----:|-------:|-------:|--------:|-------------:|------------------------------:|
| 2048 | 2205 | **3.715** | 1.23 | 256:256:128 | **11 × 512 × 4 = 22.0 KB / 17.2%** |
| 4096 | 2205 | **3.715** | 1.23 | 256:256:128 | **11 × 512 × 4 = 22.0 KB / 17.2%** |
| **8192** (TILE8192) | **2205** | **3.715** | **1.23** | 256:256:128 | **22.0 KB / 17.2%** |

Flat vs TILE — under float2 E8192@T512, tile size does not move el/cyc or MRF.

## Compare — E=16384 TILE @ T=512 (same float2 pack)

| TILE | VF_cyc | el/cyc | MRF live |
|-----:|-------:|-------:|---------:|
| 2048 | 4356 | 3.761 | 13×512×4 = 26.0 KB / 20.3% |
| 4096 | 4345 | 3.771 | 26.0 KB / 20.3% |
| 8192 | 4349 | 3.767 | 26.0 KB / 20.3% |
| 16384 | 4433 | 3.696 | 11×512×4 = 22.0 KB / 17.2% |

## MRF method

\llvm-objdump -d\ opsim \icore_binary.o\ · unique \R\\d+\ in \simt_vf\ / \ector_simt_entry\ body · KB = n × T × 4 / 1024 · % of **128 KB** MRF. SoT: \harvest_float2_mrf.tsv\. SoftLib A+B LDS.b32 T512 remains **52 → 104 KB / 81.2%** (prior harvest).

OUT: \/tmp/sv1_float2_tile_t512_20261020/\
