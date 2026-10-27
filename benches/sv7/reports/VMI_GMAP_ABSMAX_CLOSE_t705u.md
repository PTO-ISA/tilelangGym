# VMI Gmap absmax close — t699u/t705u (era=fp32-absmax)

**Date:** 2026-10-27 Asia/Hong_Kong · E=8192 · Case-1 · no recip  
**Skill:** `tilelanggym-vmi-gmap-close` · companions: simd-per-g-mapping · reduce-dual-issue-recipes · st-semantics-gate  
**Live pack:** `/mnt/fluxdata/happybot/vmi_absmax_gmap_t699u_20261027/` · `reports/RESULT_absmax_vmi.tsv`  
**Dig SoT:** `sv_simd_regress_semantics_20261026` (G16/G32)

## Layer summary

| G | Functional | Tried (inventory) | Best-sweep | SoT el/cyc | vs SoftLib CCE |
|--:|------------|-------------------|------------|-----------:|----------------|
| 16 | PASS dig VL128/NG8 VAND | VL128/NG8 VAND (+ prior abs paths) | **DONE** | **33.44** (VF 245) | SoftLib 31.39 — VMI best |
| 32 | PASS dig R32×C256 VL256/NG8 VAND; live R64×C128 also PASS | dig storeB; live vand+vabs R64×C128 | **DONE** (dig SoT; live does not beat dig) | **22.14** dig (VF 370); live 21.90 VAND / 22.02 VABS | SoftLib 21.28 — VMI best |
| 64 | PASS live VCMAX VL64 | vand + vabs | **DONE** | **30.68** VABS (VF 267); VAND 30.57 | SoftLib **30.68** — **tied** |
| 128 | PASS live VMAX→VCMAX + whole-VL VCMAX | vmaxfold_vand + vcmax_vl128_vand (same VF) | **DONE** | **32.25** (VF 254) | SoftLib **32.38** — SoftLib +0.4% |

## Dig-vs-live delta (G32)
| Check | Dig SoT | Live t699u |
|-------|---------|------------|
| abs-path | VAND | VAND + VABS_FP |
| shape | **R32×C256** | **R64×C128** |
| epilogue | absmax-only | absmax-only |
| VL/NG | VL256 / NG8 | VL256 / NG8 |
| el/cyc | **22.14** | 21.90 / 22.02 |
→ Keep dig as SoT; live is shape peer, not superseding.

## Instr mix (live t699u opsim)
| tag | mix (counts) |
|-----|--------------|
| g64_vcmax_vl64_vand | VCMAX:128 VAND:128 VLD:128 VST:128 |
| g64_vcmax_vl64_vabs | VCMAX:128 VABS:128 VLD:128 VST:128 |
| g128_*_vand (both) | VCMAX:64 VMAX:64 VAND:128 VLD:128 VST:64 |
| g32_vcg_vl256_ng8_vand | VCGMAX:32 VAND:128 VMAX:96 VDINTLV:64 VLD:64 VST:32 |
| g32_vcg_vl256_ng8_vabs | VCGMAX:32 VABS:128 VMAX:96 VDINTLV:64 VLD:64 VST:32 |

## SEMANTICS_GATE
All six live tags: gold PASS · era=fp32-absmax · VF from `vf_real_execute_time` sum · opcode matches lowering (VCGMAX @ G32; VCMAX @ G64; VMAX+VCMAX @ G128). **PASS**.

## End condition
**Not UNMEASURED-only.** Required explore set covered per G (VCGMAX mid; whole-VL VCMAX large; VMAX-fold large). Optional split-ILP not motivated (SoftLib within ~0.4% at G128; loop-guard noise). SoftLib CCE optional when VMI ≥ SoftLib (G16/G32); at G64 tied; at G128 SoftLib peer retained in table.
