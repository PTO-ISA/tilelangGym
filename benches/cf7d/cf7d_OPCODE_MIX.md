# CF7d opcode mix + ISA-mix harvest (from lean dumps)

Source: `opsim_cf7d_lean/*/core0.veccore0_instr_exe.csv` (synced from pto-b10 `/tmp/st_simtvf_parallel/ptodsl_cf7d/`).
Body excludes `VF`/`VF_SIMT` wrappers and framing (`SET_FLAG`/`WAIT_FLAG`/`END*`/`DC_PRELOAD`/`NOP`).
N=32 · E=256 · units=E/N=8 · IPC=body_calls/VF_cyc · el/cyc=E/VF_cyc.

## A ≡ B opcode identity

| G | p | A body | B body | opcode Counter equal? |
|--:|--:|-------:|-------:|:---------------------:|
| 0 | 25 | 204 | 204 | YES |
| 0 | 50 | 204 | 204 | YES |
| 0 | 95 | 204 | 204 | YES |
| 16 | 25 | 716 | 716 | YES |
| 16 | 50 | 716 | 716 | YES |
| 16 | 95 | 716 | 716 | YES |
| 64 | 25 | 2252 | 2252 | YES |
| 64 | 50 | 2252 | 2252 | YES |
| 64 | 95 | 2252 | 2252 | YES |

**All A≡B opcode Counters identical:** YES

## Per-cell opcode call_count (balanced)

### cf7d_e256_b8_g0_vl64_pgate25_armA
VF=125.0 · body_instr=204 · IPC=1.6320 · el/cyc=2.0480

| opcode | call_count |
|--------|----------:|
| `RV_VSEL` | 64 |
| `RV_VCMP_GT` | 40 |
| `RV_VLDI` | 20 |
| `MOV_XD_IMM` | 17 |
| `RV_VDUP` | 16 |
| `MOVK` | 11 |
| `MOV_XD_SPR` | 6 |
| `MOV_SRC_TO_DST_ALIGNv2` | 4 |
| `RV_VSTI` | 4 |
| `ADD_IMM` | 3 |
| `INSERT_XD_XN` | 3 |
| `RV_SEND` | 3 |
| `ADD` | 2 |
| `LD_XD_XN_IMM` | 2 |
| `RV_VDUPS` | 2 |
| `SHL` | 2 |
| `AND` | 1 |
| `LDP_XI_XJ_XN` | 1 |
| `MADD` | 1 |
| `RV_PSET` | 1 |
| `SHR` | 1 |
| **SUM** | **204** |
Balance: SUM==body_instr → **PASS**

### cf7d_e256_b8_g0_vl64_pgate25_armB
VF=125.0 · body_instr=204 · IPC=1.6320 · el/cyc=2.0480

| opcode | call_count |
|--------|----------:|
| `RV_VSEL` | 64 |
| `RV_VCMP_GT` | 40 |
| `RV_VLDI` | 20 |
| `MOV_XD_IMM` | 17 |
| `RV_VDUP` | 16 |
| `MOVK` | 11 |
| `MOV_XD_SPR` | 6 |
| `MOV_SRC_TO_DST_ALIGNv2` | 4 |
| `RV_VSTI` | 4 |
| `ADD_IMM` | 3 |
| `INSERT_XD_XN` | 3 |
| `RV_SEND` | 3 |
| `ADD` | 2 |
| `LD_XD_XN_IMM` | 2 |
| `RV_VDUPS` | 2 |
| `SHL` | 2 |
| `AND` | 1 |
| `LDP_XI_XJ_XN` | 1 |
| `MADD` | 1 |
| `RV_PSET` | 1 |
| `SHR` | 1 |
| **SUM** | **204** |
Balance: SUM==body_instr → **PASS**

### cf7d_e256_b8_g0_vl64_pgate25_armC
VF=69.0 · body_instr=83 · IPC=1.2029 · el/cyc=3.7101

| opcode | call_count |
|--------|----------:|
| `MOV_XD_IMM` | 15 |
| `MOVK` | 10 |
| `RV_VCMP_GT` | 8 |
| `RV_VSEL` | 8 |
| `MOV_XD_SPR` | 6 |
| `RV_VLDI` | 6 |
| `MOV_SRC_TO_DST_ALIGNv2` | 4 |
| `RV_VSTI` | 4 |
| `ADD_IMM` | 3 |
| `RV_SEND` | 3 |
| `ADD` | 2 |
| `INSERT_XD_XN` | 2 |
| `LD_XD_XN_IMM` | 2 |
| `RV_VDUP` | 2 |
| `SHL` | 2 |
| `AND` | 1 |
| `LDP_XI_XJ_XN` | 1 |
| `MADD` | 1 |
| `RV_PSET` | 1 |
| `RV_VDUPS` | 1 |
| `SHR` | 1 |
| **SUM** | **83** |
Balance: SUM==body_instr → **PASS**

### cf7d_e256_b8_g16_vl64_pgate25_armA
VF=643.0 · body_instr=716 · IPC=1.1135 · el/cyc=0.3981

| opcode | call_count |
|--------|----------:|
| `RV_VMUL` | 512 |
| `RV_VSEL` | 64 |
| `RV_VCMP_GT` | 40 |
| `RV_VLDI` | 20 |
| `MOV_XD_IMM` | 16 |
| `RV_VDUP` | 16 |
| `MOVK` | 11 |
| `MOV_XD_SPR` | 6 |
| `MOV_SRC_TO_DST_ALIGNv2` | 4 |
| `RV_VSTI` | 4 |
| `ADD_IMM` | 3 |
| `INSERT_XD_XN` | 3 |
| `RV_SEND` | 3 |
| `RV_VDUPS` | 3 |
| `ADD` | 2 |
| `LD_XD_XN_IMM` | 2 |
| `SHL` | 2 |
| `AND` | 1 |
| `LDP_XI_XJ_XN` | 1 |
| `MADD` | 1 |
| `RV_PSET` | 1 |
| `SHR` | 1 |
| **SUM** | **716** |
Balance: SUM==body_instr → **PASS**

### cf7d_e256_b8_g16_vl64_pgate25_armC
VF=189.0 · body_instr=215 · IPC=1.1376 · el/cyc=1.3545

| opcode | call_count |
|--------|----------:|
| `RV_VMUL` | 128 |
| `MOV_XD_IMM` | 16 |
| `MOVK` | 11 |
| `RV_VCMP_GT` | 8 |
| `RV_VSEL` | 8 |
| `MOV_XD_SPR` | 6 |
| `RV_VLDI` | 6 |
| `MOV_SRC_TO_DST_ALIGNv2` | 4 |
| `RV_VSTI` | 4 |
| `ADD_IMM` | 3 |
| `INSERT_XD_XN` | 3 |
| `RV_SEND` | 3 |
| `ADD` | 2 |
| `LD_XD_XN_IMM` | 2 |
| `RV_VDUP` | 2 |
| `RV_VDUPS` | 2 |
| `SHL` | 2 |
| `AND` | 1 |
| `LDP_XI_XJ_XN` | 1 |
| `MADD` | 1 |
| `RV_PSET` | 1 |
| `SHR` | 1 |
| **SUM** | **215** |
Balance: SUM==body_instr → **PASS**

### cf7d_e256_b8_g16_vl64_pgate95_armC
VF=634.0 · body_instr=659 · IPC=1.0394 · el/cyc=0.4038

| opcode | call_count |
|--------|----------:|
| `RV_VMUL` | 512 |
| `RV_VCMP_GT` | 32 |
| `RV_VSEL` | 32 |
| `MOV_XD_IMM` | 16 |
| `RV_VLDI` | 12 |
| `MOVK` | 11 |
| `RV_VDUP` | 8 |
| `MOV_XD_SPR` | 6 |
| `MOV_SRC_TO_DST_ALIGNv2` | 4 |
| `RV_VSTI` | 4 |
| `ADD_IMM` | 3 |
| `INSERT_XD_XN` | 3 |
| `RV_SEND` | 3 |
| `ADD` | 2 |
| `LD_XD_XN_IMM` | 2 |
| `RV_VDUPS` | 2 |
| `SHL` | 2 |
| `AND` | 1 |
| `LDP_XI_XJ_XN` | 1 |
| `MADD` | 1 |
| `RV_PSET` | 1 |
| `SHR` | 1 |
| **SUM** | **659** |
Balance: SUM==body_instr → **PASS**

### cf7d_e256_b8_g64_vl64_pgate25_armA
VF=2161.0 · body_instr=2252 · IPC=1.0421 · el/cyc=0.1185

| opcode | call_count |
|--------|----------:|
| `RV_VMUL` | 2048 |
| `RV_VSEL` | 64 |
| `RV_VCMP_GT` | 40 |
| `RV_VLDI` | 20 |
| `MOV_XD_IMM` | 16 |
| `RV_VDUP` | 16 |
| `MOVK` | 11 |
| `MOV_XD_SPR` | 6 |
| `MOV_SRC_TO_DST_ALIGNv2` | 4 |
| `RV_VSTI` | 4 |
| `ADD_IMM` | 3 |
| `INSERT_XD_XN` | 3 |
| `RV_SEND` | 3 |
| `RV_VDUPS` | 3 |
| `ADD` | 2 |
| `LD_XD_XN_IMM` | 2 |
| `SHL` | 2 |
| `AND` | 1 |
| `LDP_XI_XJ_XN` | 1 |
| `MADD` | 1 |
| `RV_PSET` | 1 |
| `SHR` | 1 |
| **SUM** | **2252** |
Balance: SUM==body_instr → **PASS**

### cf7d_e256_b8_g64_vl64_pgate25_armC
VF=573.0 · body_instr=599 · IPC=1.0454 · el/cyc=0.4468

| opcode | call_count |
|--------|----------:|
| `RV_VMUL` | 512 |
| `MOV_XD_IMM` | 16 |
| `MOVK` | 11 |
| `RV_VCMP_GT` | 8 |
| `RV_VSEL` | 8 |
| `MOV_XD_SPR` | 6 |
| `RV_VLDI` | 6 |
| `MOV_SRC_TO_DST_ALIGNv2` | 4 |
| `RV_VSTI` | 4 |
| `ADD_IMM` | 3 |
| `INSERT_XD_XN` | 3 |
| `RV_SEND` | 3 |
| `ADD` | 2 |
| `LD_XD_XN_IMM` | 2 |
| `RV_VDUP` | 2 |
| `RV_VDUPS` | 2 |
| `SHL` | 2 |
| `AND` | 1 |
| `LDP_XI_XJ_XN` | 1 |
| `MADD` | 1 |
| `RV_PSET` | 1 |
| `SHR` | 1 |
| **SUM** | **599** |
Balance: SUM==body_instr → **PASS**

### cf7d_e256_b8_g64_vl64_pgate95_armA
VF=2161.0 · body_instr=2252 · IPC=1.0421 · el/cyc=0.1185

| opcode | call_count |
|--------|----------:|
| `RV_VMUL` | 2048 |
| `RV_VSEL` | 64 |
| `RV_VCMP_GT` | 40 |
| `RV_VLDI` | 20 |
| `MOV_XD_IMM` | 16 |
| `RV_VDUP` | 16 |
| `MOVK` | 11 |
| `MOV_XD_SPR` | 6 |
| `MOV_SRC_TO_DST_ALIGNv2` | 4 |
| `RV_VSTI` | 4 |
| `ADD_IMM` | 3 |
| `INSERT_XD_XN` | 3 |
| `RV_SEND` | 3 |
| `RV_VDUPS` | 3 |
| `ADD` | 2 |
| `LD_XD_XN_IMM` | 2 |
| `SHL` | 2 |
| `AND` | 1 |
| `LDP_XI_XJ_XN` | 1 |
| `MADD` | 1 |
| `RV_PSET` | 1 |
| `SHR` | 1 |
| **SUM** | **2252** |
Balance: SUM==body_instr → **PASS**

### cf7d_e256_b8_g64_vl64_pgate95_armC
VF=2171.0 · body_instr=2195 · IPC=1.0111 · el/cyc=0.1179

| opcode | call_count |
|--------|----------:|
| `RV_VMUL` | 2048 |
| `RV_VCMP_GT` | 32 |
| `RV_VSEL` | 32 |
| `MOV_XD_IMM` | 16 |
| `RV_VLDI` | 12 |
| `MOVK` | 11 |
| `RV_VDUP` | 8 |
| `MOV_XD_SPR` | 6 |
| `MOV_SRC_TO_DST_ALIGNv2` | 4 |
| `RV_VSTI` | 4 |
| `ADD_IMM` | 3 |
| `INSERT_XD_XN` | 3 |
| `RV_SEND` | 3 |
| `ADD` | 2 |
| `LD_XD_XN_IMM` | 2 |
| `RV_VDUPS` | 2 |
| `SHL` | 2 |
| `AND` | 1 |
| `LDP_XI_XJ_XN` | 1 |
| `MADD` | 1 |
| `RV_PSET` | 1 |
| `SHR` | 1 |
| **SUM** | **2195** |
Balance: SUM==body_instr → **PASS**

## ISA-mix families (ops/N) — key cells

| tag | VF | body | IPC | el/cyc | load | store | pred_select | eltwise | bcast | send | scalar_tax | mte | other | ops/N tot |
|----|---:|-----:|----:|------:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cf7d_e256_b8_g0_vl64_pgate25_armA` | 125.0 | 204 | 1.632 | 2.048 | 2.500 | 0.500 | 13.125 | 0.000 | 2.250 | 0.375 | 3.750 | 0.000 | 3.000 | 25.500 |
| `cf7d_e256_b8_g0_vl64_pgate25_armB` | 125.0 | 204 | 1.632 | 2.048 | 2.500 | 0.500 | 13.125 | 0.000 | 2.250 | 0.375 | 3.750 | 0.000 | 3.000 | 25.500 |
| `cf7d_e256_b8_g0_vl64_pgate25_armC` | 69.0 | 83 | 1.203 | 3.710 | 0.750 | 0.500 | 2.125 | 0.000 | 0.375 | 0.375 | 3.500 | 0.000 | 2.750 | 10.375 |
| `cf7d_e256_b8_g16_vl64_pgate25_armA` | 643.0 | 716 | 1.114 | 0.398 | 2.500 | 0.500 | 13.125 | 64.000 | 2.375 | 0.375 | 3.625 | 0.000 | 3.000 | 89.500 |
| `cf7d_e256_b8_g16_vl64_pgate25_armC` | 189.0 | 215 | 1.138 | 1.354 | 0.750 | 0.500 | 2.125 | 16.000 | 0.500 | 0.375 | 3.625 | 0.000 | 3.000 | 26.875 |
| `cf7d_e256_b8_g16_vl64_pgate95_armC` | 634.0 | 659 | 1.039 | 0.404 | 1.500 | 0.500 | 8.125 | 64.000 | 1.250 | 0.375 | 3.625 | 0.000 | 3.000 | 82.375 |
| `cf7d_e256_b8_g64_vl64_pgate25_armA` | 2161.0 | 2252 | 1.042 | 0.118 | 2.500 | 0.500 | 13.125 | 256.000 | 2.375 | 0.375 | 3.625 | 0.000 | 3.000 | 281.500 |
| `cf7d_e256_b8_g64_vl64_pgate25_armC` | 573.0 | 599 | 1.045 | 0.447 | 0.750 | 0.500 | 2.125 | 64.000 | 0.500 | 0.375 | 3.625 | 0.000 | 3.000 | 74.875 |
| `cf7d_e256_b8_g64_vl64_pgate95_armA` | 2161.0 | 2252 | 1.042 | 0.118 | 2.500 | 0.500 | 13.125 | 256.000 | 2.375 | 0.375 | 3.625 | 0.000 | 3.000 | 281.500 |
| `cf7d_e256_b8_g64_vl64_pgate95_armC` | 2171.0 | 2195 | 1.011 | 0.118 | 1.500 | 0.500 | 8.125 | 256.000 | 1.250 | 0.375 | 3.625 | 0.000 | 3.000 | 274.375 |

## ISA-mix → el/cyc bridge (intra-*d A vs C)

Unit: N=32 · E=256 · IPC=body/VF. `pred_el/cyc = IPC × (N / ops_per_N_total) = E/VF` (tautology on total ops); useful when family mix differs.

### G=0 p=25
- A: VF=125.0 body=204 IPC=1.632 el/cyc=2.0480 ops/N=25.500
- C: VF=69.0 body=83 IPC=1.203 el/cyc=3.7101 ops/N=10.375
- measured el/cyc ratio A/C = 0.5520
- pred_ratio (IPC×ops) = 0.5520 · |pred/meas−1| = 0.0000
- pred_select ops/N: A=13.125 C=2.125
- eltwise ops/N: A=0.000 C=0.000

### G=16 p=25
- A: VF=643.0 body=716 IPC=1.114 el/cyc=0.3981 ops/N=89.500
- C: VF=189.0 body=215 IPC=1.138 el/cyc=1.3545 ops/N=26.875
- measured el/cyc ratio A/C = 0.2939
- pred_ratio (IPC×ops) = 0.2939 · |pred/meas−1| = 0.0000
- pred_select ops/N: A=13.125 C=2.125
- eltwise ops/N: A=64.000 C=16.000

### G=64 p=25
- A: VF=2161.0 body=2252 IPC=1.042 el/cyc=0.1185 ops/N=281.500
- C: VF=573.0 body=599 IPC=1.045 el/cyc=0.4468 ops/N=74.875
- measured el/cyc ratio A/C = 0.2652
- pred_ratio (IPC×ops) = 0.2652 · |pred/meas−1| = 0.0000
- pred_select ops/N: A=13.125 C=2.125
- eltwise ops/N: A=256.000 C=64.000

### G=64 p=95
- A: VF=2161.0 body=2252 IPC=1.042 el/cyc=0.1185 ops/N=281.500
- C: VF=2171.0 body=2195 IPC=1.011 el/cyc=0.1179 ops/N=274.375
- measured el/cyc ratio A/C = 1.0046
- pred_ratio (IPC×ops) = 1.0046 · |pred/meas−1| = 0.0000
- pred_select ops/N: A=13.125 C=8.125
- eltwise ops/N: A=256.000 C=256.000

