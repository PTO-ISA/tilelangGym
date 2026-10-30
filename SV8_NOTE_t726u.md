# SV8 prep note (Lok t726u) — does **not** block PR #8 ANALYSIS

**Tip path:** `benches/sv8/` on PR #8 tip (`tilelangGym-pr8-tip`)

| File | Role |
|------|------|
| `config.toml` | id=`sv8` · title **Case-3 quant e2e bcast** · G=16 · R=64 · C=128 · T=32 |
| `sv8_case3_bcast.py` | SoftLib / TileLang SimtVF Case-3: absmax → recip → bcast mul · fp16 IO |
| `sv8v_case3_bcast.py` | VMI twin |
| `sv8d_case3_bcast.py` | handwritten SIMD DSL twin |
| `VERDICT.md` | gate simt-asc PASS (live + spill_dist); simd-pto mostly red historically; SoftLib simt-pto **deferred** |

**Exists today:** tiny **G=16** Case-3 **e2e quant** path (R64×C128) with live + spill_dist arms — **not** Case-1 absmax-only; includes recip epilogue (`era` ≠ fp32-absmax Case-1).

**Not ready as SV5–7-style 3-arm absmax SoT:** needs remeasure under updated harness §21/§22 if Lok wants SIMD vs SIMT whole e2e quant compare with axis labels. Parent will dispatch measure separately.

**Oneshoot scripts on tip:** `scripts/oneshot/oneshot_sv5_sv6_sv8.sh`, `oneshot_sv8d_retry.sh`, laptop `run_via_laptop_sv5_sv6_sv8.ps1`.
