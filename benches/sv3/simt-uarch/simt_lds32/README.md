# simt_lds32

SIMT LDS/STS bandwidth test. The tile is 64×128 (8192 floats). The kernel has one input and one output:

1. `TLOAD` moves `src` from GM to UB.
2. A SIMT vector function copies UB to UB: `dst[i] = src[i]` (`SIMT_LDS` on `src`, `SIMT_STS` on `dst`).
3. `TSTORE` moves `dst` from UB to GM.

The launch is `threadX=1024` by `threadY=1` (1024 threads). Each thread copies 8 elements. Element `k` of linear thread `tid` is `tid + k * 1024`, so a warp still touches consecutive floats. Golden data is a copy of `input1.bin`.

## Run

```bash
bash run.sh                         # A5 (default)
bash run.sh --a6                    # A6: dav-920r1-vec / dav_9201
bash run.sh -c case1_float_64_128   # one case; suite prefix is optional
```

Cmake: `-DPTO_ARCH=a5|a6`. A5 uses `--cce-aicore-arch=dav-c310-vec`; A6 uses `dav-920r1-vec`. Build dirs: `build_a5` / `build_a6`.

A short name such as `case1_float_64_128` is passed to gtest as `SIMTLds32Test.case1_float_64_128`. Current case:

- `case1_float_64_128`

## Thread shape search

The tile is 64×128 (8192 floats). Each candidate launches `threadX` by `threadY` with `threadX * threadY <= 1024` and `threadX` a multiple of 32. Element `k` of linear thread `tid` is `tid + k * nthreads`, so a warp still touches consecutive floats. `simt_vf_cycles` is the retire cycle of `PUSHQ VF_SIMT` in `instr_log.dump` minus the pop cycle of the same instruction in `instr_popped_log.dump`. Every shape matched golden data.

| threadX | threadY | threads | elements/thread | simt_vf_cycles |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 1 | 1024 | 8 | 957 |
| 32 | 32 | 1024 | 8 | 989 |
| 512 | 1 | 512 | 16 | 990 |
| 128 | 8 | 1024 | 8 | 993 |
| 256 | 4 | 1024 | 8 | 994 |
| 64 | 16 | 1024 | 8 | 995 |
| 512 | 2 | 1024 | 8 | 997 |
| 32 | 16 | 512 | 16 | 1009 |
| 64 | 8 | 512 | 16 | 1009 |
| 128 | 4 | 512 | 16 | 1009 |
| 256 | 2 | 512 | 16 | 1027 |
| 128 | 2 | 256 | 32 | 2311 |
| 32 | 8 | 256 | 32 | 2312 |
| 64 | 4 | 256 | 32 | 2313 |
| 256 | 1 | 256 | 32 | 2417 |
| 128 | 1 | 128 | 64 | 4382 |
| 64 | 2 | 128 | 64 | 4399 |
| 32 | 4 | 128 | 64 | 4400 |
| 64 | 1 | 64 | 128 | 8636 |
| 32 | 2 | 64 | 128 | 8662 |
| 32 | 1 | 32 | 256 | 17205 |

`threadX=1024`, `threadY=1` is the shortest at 957 cycles. A later standalone run of that kernel measured 976 cycles.

## Count SIMT VF cycles

`count_simt_vf_cycles.py` reads the camodel veccore logs written under `build_a5/bin` (or `build_a6/bin`).

`simt_vf_cycles` is the retire cycle of `PUSHQ VF_SIMT` in `instr_log.dump` minus the pop cycle of the same instruction in `instr_popped_log.dump`, matched by instruction ID.

Each SIMT issue inside that window is paired the same way:

- `count` is the number of issues.
- `cycles` is the sum of `retire_cycle - pop_cycle`. Warps are counted separately.
- `avg` is `cycles / count`.
- `busy` is the merged length of `[pop, retire)` intervals, so overlapping warps of that instruction count once. The `busy` total also merges different instructions that are in flight at the same time. `VF_SIMT` is left out of that total.

```bash
python3 count_simt_vf_cycles.py build_a5/bin
# or build_a6/bin
```

`case1_float_64_128` on the Ascend950 simulator:

```text
core0.veccore0 ID 56  pop 2658  retire 3634  simt_vf_cycles 976

instruction  count  cycles     avg  busy
-----------  -----  ------  ------  ----
SIMT_IADD      448    3584    8.00   779
SIMT_LEA       288    2784    9.67   772
SIMT_LDS       256    9046   35.34   874
SIMT_STS       256    7849   30.66   862
SIMT_MOVI      224    1568    7.00   681
SIMT_S2R        64     640   10.00   402
SIMT_END        32     288    9.00   223
SIMT_IADD_I     32     256    8.00   243
SIMT_SHFI       32     224    7.00   221
VF_SIMT          1     976  976.00   976
-----------  -----  ------  ------  ----
total         1633   27215   16.67   950
```

`SIMT_LDS` and `SIMT_STS` are each 256 issues: 8192 elements / 32 lanes in a warp.

## Dump AI Core assembly

`dump_aicore_asm.sh` extracts the device object embedded in a kernel shared library and disassembles it. Pass the path of `xxx_kernel.so`. The script writes `xxx_kernel.o` and `xxx_kernel.S` in the same directory as that `.so`.

It runs these two steps with `llvm-objcopy` and `llvm-objdump` from `/home/chenjinlin/cann9.3.0-1014/cann-9.3.0`. The script sets `ASCEND_HOME_PATH` to that tree and sources its `set_env.sh`.

```bash
llvm-objcopy -O binary -j .aicore_binary xxx_kernel.so xxx_kernel.o
llvm-objdump -d xxx_kernel.o > xxx_kernel.S
```

Build the kernel first with `bash run.sh`, then dump it:

```bash
bash dump_aicore_asm.sh build/lib/libsimt_lds32_kernel.so
```

`libsimt_lds32_kernel.S` is the `elf64-hiipu` disassembly of `.text` inside `.aicore_binary`, including `runSimtLds32`.
