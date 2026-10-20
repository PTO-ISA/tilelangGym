# SV1 float2 multi-TILE @ T=512 — E=16384 primary (2026-10-20)

Pins: cann_92b2 · Ascend950PR_9599 · `ST_VF_MODE=simt-asc` · stock float2 (`max_vector_bits=64`) · kernel `sv1t_tiled_add.py` (serial tiles × Parallel(TILE), add-in-shared).

## §5.1 primary = E=16384 TILE∈{2048,4096,8192,16384} @ T=512

| TILE | serial | VF_cyc | el/cyc (=E/VF) | IADD | BRANCH | MRF live |
|-----:|-------:|-------:|---------------:|-----:|-------:|----------|
| 2048 | 8 | 4356 | 3.761 | 912 | 128 | 26.0 KB / 20.3% |
| 4096 | 4 | 4345 | **3.771** | 848 | 64 | 26.0 / 20.3% |
| 8192 | 2 | 4349 | 3.767 | 816 | 32 | 26.0 / 20.3% |
| 16384 | 1 | 4433 | 3.696 | 768 | 0 | 22.0 / 17.2% |

el/cyc always **E/VF** (16384/VF), never TILE/VF. Peak tile hub = **TILE4096 = 3.771**.

## Why E8192 TILE ladder looked flat (not used as §5.1)

Prior primary wrongly used E=8192 TILE∈{2048,4096,8192} — all VF=2205 / el=3.715.
**Not a launch bug:** TIR shows correct `range(4)×Parallel(2048)` / `range(2)×Parallel(4096)` / `range(1)×Parallel(8192)`; source md5s differ; tags carry correct E/TILE.
**Root cause:** at E=8192 the compiler **fully unrolls** the small serial tile trip counts → identical opsim mix `LDS:FADD:STS:IADD = 256:256:128:384` and **no** BRANCH/ISETP → identical VF.
At E=16384, BRANCH/ISETP scale with serial trip count and VF **does** move (4356→4345→4349→4433).

E8192 TILE=E (**TILE8192**) remains the §5.2 T-dim hub (peak T1024=3.846) — that is Parallel(E)/TILE=E, **not** a multi-TILE ladder.

## Artifacts

- Opsim: `/tmp/sv1_float2_e16k_tile_t512_20261020/` (promoted from prior mixed OUT; no re-opsim needed — E16K arms already PASS)
- Prior mixed (E8192+E16K): `/tmp/sv1_float2_tile_t512_20261020/`
- MRF: `harvest_float2_mrf.tsv` (llvm-objdump unique Rn × T × 4B / 128 KB)
