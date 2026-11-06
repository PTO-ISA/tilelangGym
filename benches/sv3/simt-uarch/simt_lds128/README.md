# simt_lds128

SIMT `LDS.b128` bandwidth test. The tile is the same 64×128 float buffer as `simt_lds32` (8192 floats, 32768 bytes). The SIMT function reinterprets it as 128-bit values and copies it:

1. `TLOAD` moves `src` from GM to UB.
2. Each thread copies 2 128-bit values — either `dst[i] = src[i]` (default) or explicit `LDS.b128`/`STS.b128` inline asm.
3. `TSTORE` moves `dst` from UB to GM.

The launch is `threadX=1024`, `threadY=1`. Element `k` of thread `tid` is `tid + k * 1024`, so a warp touches consecutive 16-byte words. Golden data is a copy of `input1.bin`.

## Run

```bash
bash run.sh                          # A5, C assign (default)
bash run.sh --a6                     # A6: dav-920r1-vec / dav_9201
bash run.sh --asm                    # inline LDS.b128 / STS.b128 (A5)
bash run.sh -c case1_float_64_128
```

Cmake knobs: `-DPTO_ARCH=a5|a6`, `-DSIMT_LDS128_USE_INLINE_ASM=ON|OFF`.
A5 uses `--cce-aicore-arch=dav-c310-vec`; A6 uses `dav-920r1-vec`. Build dirs are
`build_a5` / `build_a6`.

## Count SIMT VF cycles

```bash
python3 count_simt_vf_cycles.py build_a5/bin
# or build_a6/bin
```

`bw` for `SIMT_LDS` and `SIMT_STS` is `64*128*4 / real`.
