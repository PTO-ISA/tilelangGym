# simt_add

## Run

```bash
bash run.sh                         # A5 (default)
bash run.sh --a6                    # A6: dav-920r1-vec / dav_9201
bash run.sh -c case1_float_64_128   # one case; suite prefix is optional
```

Cmake: `-DPTO_ARCH=a5|a6`. A5 uses `--cce-aicore-arch=dav-c310-vec`; A6 uses `dav-920r1-vec`. Build dirs: `build_a5` / `build_a6`.

A short name such as `case1_float_64_128` is passed to gtest as `SIMTAddTest.case1_float_64_128`. The kernel launches `threadX=256` by `threadY=4` (1024 threads), and each thread adds 8 elements. Current case:

- `case1_float_64_128`

## Thread shape search

The tile is 64×128 (8192 floats). Each candidate launches `threadX` by `threadY` with `threadX * threadY <= 1024` and `threadX` a multiple of 32. Element `k` of linear thread `tid` is `tid + k * nthreads`, so a warp still reads consecutive floats. `simt_vf_cycles` is the retire cycle of `PUSHQ VF_SIMT` in `instr_log.dump` minus the pop cycle of the same instruction in `instr_popped_log.dump`. Every shape matched golden data, and each logged 512 `SIMT_LDS` issues.

| threadX | threadY | threads | elements/thread | simt_vf_cycles |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 4 | 1024 | 8 | 1369 |
| 1024 | 1 | 1024 | 8 | 1382 |
| 64 | 16 | 1024 | 8 | 1387 |
| 128 | 8 | 1024 | 8 | 1387 |
| 512 | 2 | 1024 | 8 | 1398 |
| 32 | 32 | 1024 | 8 | 1403 |
| 512 | 1 | 512 | 16 | 1672 |
| 32 | 16 | 512 | 16 | 1684 |
| 256 | 2 | 512 | 16 | 1704 |
| 64 | 8 | 512 | 16 | 1739 |
| 128 | 4 | 512 | 16 | 1739 |
| 32 | 8 | 256 | 32 | 2663 |
| 64 | 4 | 256 | 32 | 2663 |
| 128 | 2 | 256 | 32 | 2663 |
| 256 | 1 | 256 | 32 | 2815 |
| 128 | 1 | 128 | 64 | 5027 |
| 32 | 4 | 128 | 64 | 5040 |
| 64 | 2 | 128 | 64 | 5040 |
| 64 | 1 | 64 | 128 | 9916 |
| 32 | 2 | 64 | 128 | 9942 |
| 32 | 1 | 32 | 256 | 19769 |

`threadX=256`, `threadY=4` is the shortest. A later standalone run of that kernel measured 1419 cycles with the same 2497 SIMT instructions.

## Dump AI Core assembly

`dump_aicore_asm.sh` extracts the device object embedded in a kernel shared
library and disassembles it. Pass the path of `xxx_kernel.so`. The script
writes `xxx_kernel.o` and `xxx_kernel.S` in the same directory as that `.so`.

It runs these two steps with `llvm-objcopy` and `llvm-objdump` from
`/home/chenjinlin/cann9.3.0-1014/cann-9.3.0`. The script sets
`ASCEND_HOME_PATH` to that tree and sources its `set_env.sh`.

```bash
llvm-objcopy -O binary -j .aicore_binary xxx_kernel.so xxx_kernel.o
llvm-objdump -d xxx_kernel.o > xxx_kernel.S
```

Build the kernel first with `bash run.sh`, then dump it:

```bash
bash dump_aicore_asm.sh build/lib/libsimt_add_kernel.so
```

`libsimt_add_kernel.S` is the `elf64-hiipu` disassembly of `.text` inside
`.aicore_binary`, including `runSimtAdd`.
