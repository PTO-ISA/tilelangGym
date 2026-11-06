# simt_lds128asm

Same bandwidth shape as `simt_lds128` (64×128 float tile, 1024 threads × 2×
`__uint128_t`), but the SIMT copy uses **inline asm** so `LDS.b128` /
`STS.b128` are explicit:

```text
LDS.b128  tmp, [src]  ws:0 rs:0 wait:0b0000000 stall:2
STS.b128  tmp, [dst]  rs:0 wait:0b0000001 stall:2
```

C reference (`simt_lds128`) is `dst[i] = src[i]`. Here the opcodes are forced
through `asm volatile`. `__uint128_t` cannot bind to `"R"` (ccec: *Unsupported
vector registor types for inline*), so the 128-bit payload uses hard GPR
**R20** (same register class as the C lowering); only the address pointers use
`"R"`.

## Run

```bash
bash run.sh                         # A5 (default)
bash run.sh --a6                    # A6 (may not assemble native LDS.b128)
bash run.sh -c case1_float_64_128
```

Cmake: `-DPTO_ARCH=a5|a6`. Build dirs: `build_a5` / `build_a6`.

## Dump device asm

```bash
bash dump_aicore_asm.sh build_a5/lib/libsimt_lds128asm_kernel.so
grep -E 'LDS\.b128|STS\.b128' build_a5/lib/libsimt_lds128asm_kernel.S
```

## Count SIMT VF cycles / LDS bw

```bash
python3 count_simt_vf_cycles.py build_a5/bin
# or build_a6/bin
```

`moved` / `bw` are auto-derived from the camodel log (Rd/Rs count, else Binary
bits \[35:32\]): each `LDS.b128` / `STS.b128` issue is 16 B/lane × 32 lanes;
for a full-tile copy that totals `64*128*4` bytes. `bw = moved_bytes / real`.
