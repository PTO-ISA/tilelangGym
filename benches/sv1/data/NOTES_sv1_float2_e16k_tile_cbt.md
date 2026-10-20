# SV1 float2 E16K TILE@T512 copy-between-tiles (anti-fuse) — 2026-10-20

Pins: cann_92b2 · Ascend950PR_9599 · `ST_VF_MODE=simt-asc` · stock float2 (`max_vector_bits=64`) · B64 lib.

## Structural change vs fused baseline

**Fused (prior `/tmp/sv1_float2_e16k_tile_t512_20261020`):** full-E `T.copy` outside, then `with vf_region: for t in serial(E/TILE): Parallel(TILE) add` → **VF_SIMT call_count=1** for all TILE, VF≈4350 flat.

**This CBT kernel:** each serial trip does GM→UB `T.copy` (TILE) → `vf_region` Parallel(TILE) add → UB→GM `T.copy`, so MTE copies sit **between** tile trips and should prevent fusing into one VF.

Compiled mode used: **tile_ub**

- `tile_ub`: TILE-sized UB + `T.copy(A[t*TILE:(t+1)*TILE], a_ub)`
- `e_ub_slice`: E-sized UB + slice copy into UB window (fallback)

Tag naming unchanged: `sv1tt_e{E}_tile{TILE}_t{T}`. See SUMMARY for VF_SIMT call_count vs trips (=E/TILE).

## Sweep

E=16384 TILE ∈ {2048,4096,8192,16384} @ T=512 (+ E8192 if cheap).

Do NOT update PR ANALYSIS yet — measure first.


## Results (E=16384 @ T=512, mode=tile_ub)

| TILE | trips | VF_SIMT call_count | VF_cyc | el/cyc | wall_us | PASS |
|------|------:|-------------------:|-------:|-------:|--------:|:----:|
| 2048 | 8 | 8 | 4203 | 3.8982 | 2.34 | PASS |
| 4096 | 4 | 4 | 4273 | 3.8343 | 2.37 | PASS |
| 8192 | 2 | 2 | 4253 | 3.8523 | 2.37 | PASS |
| 16384 | 1 | 1 | 4433 | 3.6959 | 2.46 | PASS |

**Antifuse:** YES — VF_SIMT call_count matches trips (E/TILE) for every arm.
**el/cyc:** mild differentiation (3.90 -> 3.70); TILE2048 slightly best; still near fused baseline ~3.76 / VF~4350.
**LDS/STS/FADD:** flat 512/256/512 across TILE (total work same; split across N VF calls).
**BRANCH/ISETP:** 0 (no serial loop inside VF body — loop is outside between MTE copies).
**float2:** yes (float4=0). Fallback e_ub_slice not needed.

Optional E8192 ladder skipped (CBT_MODE_USED lost across compile subshell); E16K focus complete.
