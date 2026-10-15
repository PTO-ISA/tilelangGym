# AICore code size vs I-cache (SimtVF T=128 schedule)

**Date:** 2026-10-15 HKT · SOC Ascend950PR_9599

## Method

- Kernel size = `aicore_binary.o` `.text` from opsim dump (NOT host `.so` ~350KB glue).
- I-cache = `core0.veccore0.ifu.icache.dump` (line size **128B** from PC stride `0x80`).
- Miss proxy: `prefetch MISS` / `linefill update tagRam` / `send Read biu Req`.
- `early_MISS` = MISS in first 200 dump lines (cold fill); `late_MISS` = rest (pressure).

## Sizes

See `SUMMARY_icache_codesize.tsv`. Max aicore `.text` in sweep: **6000 B (5.86 KB)**.

## Limit (conservative)

Official I$ capacity not printed in dumps. Treat **16–32 KB** as typical AICore I$ band.
Flag if aicore `.text` approaches **16 KB+** or `late_MISS` grows with unroll depth.

ILP16 at **~6 KB** sits well under that band if geometry is ≥16KB.
