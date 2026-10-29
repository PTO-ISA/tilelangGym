# SV8 — Case-3 quant e2e bcast (G16 gate)

**Era:** `case3-quant-bcast` (fp16 in/out · reduce + recip + bcast mul).  
**Do not merge** el/cyc with SV5–7 `era=fp32-absmax`.

## Shape (gate)
- R=64, C=128, G=16, T=32 · E=R×C=8192
- Labels: `reduce_axis=inner` · SoftLib_axis=`1xG`

## Kernels
| file | arm |
|------|-----|
| `sv8_case3_bcast.py` | SIMT-ASC SimtVF (`live` / `spill_dist`) |
| `sv8v_case3_bcast.py` | SIMD VMI twin — **COMPILE FAIL** this tip (LowerParallel stride=0/0) |
| `sv8d_case3_bcast.py` | Handwritten PTO-DSL SIMD (measured live) |
| `config.toml` | pins + case list |

## How to run (pto-b10 / pack)
```bash
# From pack root (box mirror or flux):
./oneshot_sv8_simt.sh          # simt-asc live + spill
./oneshot_sv8v_simd.sh         # sv8v (expect compile fail until LowerParallel fixed)
./oneshot_sv8d_dsl.sh          # sv8d DSL live
# Or: ./run_all.sh
```

## Report
- **ANALYSIS.md** — ST spine (Aims → Conclusion); truth = `RESULT.md`
- **RESULT.md** / **RESULT.tsv** — measured numbers only (no invented el/cyc)
- SoftLib SIMT A5: blocker in ANALYSIS §6 (not measured this pack)
