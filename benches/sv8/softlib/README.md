# SV8 — Case-3 quant e2e bcast (G32 tip · t737u)

- **era=`case3-quant-bcast`** · `reduce_axis=inner` · SoftLib_axis=`1xG`
- Tip gold **G=32** (R64×C128 T32 E=8192) — SIMD-worst vs SIMT-best gap explore
- **ANALYSIS.md** — ST spine; truth = `RESULT.md` / `RESULT.tsv`
- Kernels: `sv8_case3_bcast.py` (simt-asc), `sv8d_vl_resident_case3.py` (VL-resident SIMD), `sv8d_case3_bcast.py` / `sv8v_case3_bcast.py` (G16-era / blocked)
- **Do not merge** PR without Lok yes
