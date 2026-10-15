# VERDICT — SV1

## Highlight

**SIMD DSL** hits ~**85%** of ~**44** el/cyc UB roof with a small UB ld/st loop — **no per-thread index**, **no instruction scheduling**.

SimtVF best ~**4.08** el/cyc (**25%** of roof **16**); schedule todos remain.

## Verdict

Use handwritten SIMD DSL for stream eltwise when possible. Full write-up: [`ANALYSIS.md`](ANALYSIS.md). Suite table: [`reports/PASS_FAIL_PERF.md`](../../reports/PASS_FAIL_PERF.md).
