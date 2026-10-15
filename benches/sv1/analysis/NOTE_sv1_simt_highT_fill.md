# SimtVF untiled E=16K — warps to fill Body4 issue holes (T sweep)

**Date:** 2026-10-15 HKT · pack **.64** · Body4=LDS+STS+IADD+FADD · mix-cap **4** · TILE=E=**16384**

## Verdict

1. **Mid-median saturates early; mid-peak still rises slowly.** At T=2048 mid-peak Body4 IPC **1.50** (37.5% of cap 4) but mid-median stuck at **0.467** since T=256. Overall IPC **0.489**. Cap 4 is not reachable by T alone.

2. Ignoring setup/teardown: max idle bubble shrinks 244→28, but **~68% idle remains in mid** — not “only head/tail overhead left.”

3. Occupancy helps **peak bursts** (more co-issue windows); it does **not** fix typical mid-body holes (per-warp RAW + IADD tax + resources).

4. Crude Little estimate from same-warp LDS→FADD gap ≈2.8 cyc → ~**3 warps/sch** ≈ **T≈362** to hide that gap alone (does not guarantee approaching cap 4).

5. Log-fit extrapolate mid-peak→2: T≈8788 (275 warps); →4: T≈1698939 (fit is rough — trust measured points more).

## Table (body = 1st LDS → last STS)

| T | warps | w/sch | VF_cyc | Body4 IPC | %cap4 | mid med | mid peak | %idle | maxBub | FADD%wait | el/cyc |
|--:|------:|------:|-------:|----------:|------:|--------:|---------:|------:|-------:|----------:|-------:|
| 128 | 4 | 1.0 | 5700 | 0.376 | 9.4 | 0.400 | 0.500 | 69.4 | 244 | 50.0 | 2.874 |
| 256 | 8 | 2.0 | 4864 | 0.435 | 10.9 | 0.467 | 0.533 | 72.0 | 57 | 50.0 | 3.368 |
| 512 | 16 | 4.0 | 4415 | 0.468 | 11.7 | 0.467 | 0.900 | 67.6 | 32 | 50.0 | 3.711 |
| 1024 | 32 | 8.0 | 4319 | 0.484 | 12.1 | 0.467 | 1.167 | 67.9 | 35 | 50.0 | 3.793 |
| 2048 | 64 | 16.0 | 4279 | 0.489 | 12.2 | 0.467 | 1.500 | 67.7 | 28 | 50.0 | 3.829 |

### Issue-width hist (% of body cycles)

| T | w=0 | w=1 | w=2 | w=3 | w=4 |
|--:|----:|----:|----:|----:|----:|
| 128 | 69.4 | 23.9 | 6.7 | 0.1 | 0.0 |
| 256 | 72.0 | 15.1 | 10.3 | 2.4 | 0.0 |
| 512 | 67.6 | 21.0 | 8.8 | 2.3 | 0.3 |
| 1024 | 67.9 | 19.7 | 9.4 | 2.4 | 0.7 |
| 2048 | 67.7 | 19.9 | 9.1 | 2.6 | 0.7 |

## Reading

- **Head/tail:** max idle bubble shrinks a lot as T↑ (fill/drain). That is setup/teardown occupancy, not mid-body pipe fill.
- **Mid-body:** 30-cyc peak/median Body4 IPC is the right metric for “are holes filled if we ignore setup/teardown?”
- **FADD waitBitMask≠0 ~50%** stays flat with T → per-warp RAW depth unchanged; T only overlaps other warps into bubbles.
- **Mix-cap 4** = 4 wrap schedulers. FADD share ~27.5% → FADD pipe still clamps to 1.0 once Body4 IPC approaches 4; we never get near that here.

## Plots

- `plots/sv1_simt_highT_fill_sweep.png`
- `plots/sv1_simt_highT_mid_series.png`

## Method

Opsim keeps with `PTOAS_KEEP_MSPROF_STAGING=1`. Body4 issue IPC from chrome-trace. Mid = body span with 15% trim each end. 30-cyc sliding window step 1.


## Answer to “how many warps to fill all holes?”

| Question | Answer |
|----------|--------|
| Fill **mid-median** Body4 IPC? | Already flat by **T=256–512** at **~0.47** (~12% of cap 4). More warps do **not** raise typical mid-body issue rate. |
| Fill **mid-peak** (best 30c)? | Still climbing slowly: T128 **0.50** → T512 **0.90** → T1024 **1.17** → T2048 **1.50** (37.5% of 4). Occupancy helps bursts, not the average. |
| Fill toward **mix-cap 4**? | **No with T alone.** Log-fit to mid-peak=4 is millions of threads — structural (per-warp RAW + IADD tax + resources), not missing occupancy. |
| Ignore head/tail only? | Max idle bubble 244→28 (good). Mid-median still ~0.47 / ~68% idle → **mid is NOT a filled pipe with only head/tail tax left.** |

Crude Little from LDS→FADD gap ≈2.8c → ~3 warps/sch ≈ T≈360 already surpassed at T512; further T mostly overlaps into rare peak windows.

## Related: T512 FADD→STS ~53c

See `NOTE_sv1_simt_FADD_STS_T512_scheduler_delay.md` — **scheduler return delay** (STS waitBitMask=0), warps balanced, not longer RAW / not bad pairing.
