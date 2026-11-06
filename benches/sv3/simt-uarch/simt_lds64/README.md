# simt_lds64

SIMT `LDS.b64` bandwidth test. The tile is the same 64×128 float buffer as `simt_lds32` (8192 floats, 32768 bytes). The SIMT function reinterprets it as `uint64_t` and copies it:

1. `TLOAD` moves `src` from GM to UB.
2. Each thread copies 4 `uint64_t` values with `dst[i] = src[i]`, so the UB load is `LDS.b64` (8 bytes) and the store is `STS.b64`.
3. `TSTORE` moves `dst` from UB to GM.

The launch is `threadX=1024`, `threadY=1`. Element `k` of thread `tid` is `tid + k * 1024`, so a warp touches consecutive 8-byte words. Golden data is a copy of `input1.bin`.

## Run

```bash
bash run.sh                         # A5 (default)
bash run.sh --a6                    # A6: dav-920r1-vec / dav_9201
bash run.sh -c case1_float_64_128
```

Cmake: `-DPTO_ARCH=a5|a6`. Build dirs: `build_a5` / `build_a6`.

## Count SIMT VF cycles

```bash
python3 count_simt_vf_cycles.py build_a5/bin
# or build_a6/bin
```

`bw` for `SIMT_LDS` and `SIMT_STS` is `64*128*4 / real`.
