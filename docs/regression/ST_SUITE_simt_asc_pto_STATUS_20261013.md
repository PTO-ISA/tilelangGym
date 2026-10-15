# t_parallel / SimtVF ST suite — simt-asc vs simt-pto (fact report)

**As of:** 2026-10-13 ~14:20 HKT  
**Hops:** LAPTOP-C8QEKIFC + DESKTOP-V7R2A6I both **offline** at report time → fresh full-suite re-run on remat tip **not completed**.

## Checkout / tip / env

| Item | Value |
|------|-------|
| Path | `/mnt/fluxdata/happybot/projects/tilelang-pto-isa-st` |
| Branch (live) | `fix/simtvf-remat-idx-evalset-vec` |
| Tip SHA | `a3012088cc61a7fac36c50ae19caf2524d5bf804` — `[Ascend] Fix SimtVF remat_idx EvalSet / vector gather pack` |
| Suite tree | also on `feat/st-simtvf-parallel-sv1-sv9` @ `593a8f0c` (PR #2); restored into remat working tree for harness |
| CANN | `/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2` (9.2.0-beta.2) |
| PTOAS (simt-pto) | nightly **0.67** in `/mnt/fluxdata/happybot/projects/.camodel_deps_nightly067` |
| SOC | Ascend950PR_9599 |

**PR refs:** [PTO-ISA/tilelang#2](https://github.com/PTO-ISA/tilelang/pull/2) (ST suite; body still stamps 2026-10-10), [PTO-ISA/tilelang#5](https://github.com/PTO-ISA/tilelang/pull/5) (remat fix; evidence = SV4 only).

## Newest FULL-suite OUT dirs (still tip `31483438`, **before** remat fix)

| Mode | OUT | Tip | Stamp |
|------|-----|-----|-------|
| **simt-asc** (+ failed simt-pto column) | `/mnt/fluxdata/happybot/st_out_cann92b2_20261010_104824` | `31483438` | 2026-10-10T10:48:24+08 |
| **simt-pto** (nightly 0.67) | `/mnt/fluxdata/happybot/st_out_nightly067_simt_pto_20261010_170837` | `31483438` | 2026-10-10T17:08:37+08 |
| **simd-pto** | `/mnt/fluxdata/happybot/st_out_simd_pto_20261010_163238` | (same stack) | 2026-10-10 |

Tables: `$OUT/VF_MODE_PASS_TABLE.txt`.

### Counts (from those table footers — do not treat as post-remat)

- **simt-asc:** **59 PASS / 5 FAIL** (64 tags in cann92b2 dual table; PR #2 still says 56/61 for an earlier stamp)
- **simt-pto (nightly 0.67):** **33 PASS / 29 FAIL**
- **simd-pto:** **0 PASS / 61 FAIL** (`Unsupported storage scope ... local` / SimdVF→PTO)
- **PTO-DSL twins (`kernels_ptodsl/` *d):** not part of `oneshot_vf_mode_regress.sh`; PR #2 still records 21/21 Simt + 21/21 SIMD PTO-DSL product PASS on Wenbo tip — **simd-pto VF-mode twins still 0/61**

## 1) Is simt-asc ALL PASS after Lok’s fix (incl. sv6/sv7)?

**Not confirmed.** No completed full-suite re-run on tip `a3012088`.

### Post-fix evidence that *does* exist (2026-10-12)

OUT: `/tmp/remat_idx_evalset_fix_20261012` (simt-asc / cann_92b2)

| tag | result |
|-----|--------|
| `sv4_e256_b8_t32_keep_idx` | PASS |
| `sv4_e256_b8_t32_remat_idx` | **PASS** (was COMPILE_FAIL `eval_vec_`) |
| `sv4_e256_b8_t32_remat_idx_calc` | PASS |

`sv4_e256_b16_t32_remat_idx` was **not** re-proven in that artifact.

### sv6 / sv7 (Lok: “previously MISSING / not run”)

On the **2026-10-10 simt-asc** table they were **not** all MISSING:

| tag | simt-asc 2026-10-10 | notes |
|-----|---------------------|-------|
| `sv6_r64_c128_g32_t32_keep_in_warp` | **PASS** | also re-harvested PASS 2026-10-12 in `.../simt-asc/SUMMARY_sv6_sv7_harvest_20261012.txt` |
| `sv7_r64_c128_g64_t32_multiwarp_ub` | **PASS** | same harvest |
| `sv7_r64_c128_g128_t32_multiwarp_ub` | **PASS** | same harvest |
| `sv6_r64_c128_g32_t32` | **FAIL compile** | bare / leftover arm |
| `sv7_r64_c128_g64_t32` | **FAIL compile** | bare / leftover arm |
| `sv7_r64_c128_g128_t32` | **FAIL compile** | bare / leftover arm |

**MISSING** in the cann92b2 dual table was on the **simt-pto** column for the keep_in_warp / multiwarp rows (those modes weren’t populated there), not “sv6/sv7 never run on simt-asc”.

→ Lok’s claim “**all** simt-asc PASS including sv6/sv7” does **not** hold on the last full table (3 bare sv6/sv7 arms still FAIL compile), and is **unverified** for a post-remat full matrix. Remat fix does not, by itself, clear those bare-arm FAILs.

### simt-asc remaining FAILs on newest full table (pre-remat tip `31483438`)

1. `sv4_e256_b8_t32_remat_idx` — FAIL compile *(cleared on tip a3012088 for this e256_b8 arm only)*  
2. `sv4_e256_b16_t32_remat_idx` — FAIL compile *(not re-run post-fix)*  
3. `sv6_r64_c128_g32_t32` — FAIL compile  
4. `sv7_r64_c128_g64_t32` — FAIL compile  
5. `sv7_r64_c128_g128_t32` — FAIL compile  

## 2) Every simt-pto case that did NOT pass (nightly067, tip `31483438`)

Source: `/mnt/fluxdata/happybot/st_out_nightly067_simt_pto_20261010_170837/VF_MODE_PASS_TABLE.txt`  
**29 / 62** non-PASS (footer: PASS 33 FAIL 29):

1. `sv2_r32_c32_t32_frag_live` — FAIL compile (`Unsupported PTO cast: float16 -> float32`)  
2. `sv2_r32_c32_t32_reload` — FAIL compile (same)  
3. `sv3_m24_vl64_k16_t32_keep` — FAIL compile (same)  
4. `sv3_m32_vl64_k16_t32_keep` — FAIL compile (same)  
5. `sv3_m32_vl64_k16_t32_split_cm16` — FAIL compile (same)  
6. `sv4_e256_b8_t32_keep_idx` — FAIL compile (PTO lowering / llvm)  
7. `sv4_e256_b8_t32_remat_idx` — FAIL compile (`eval_vec_`)  
8. `sv4_e256_b16_t32_keep_idx` — FAIL compile (PTO lowering / llvm)  
9. `sv4_e256_b16_t32_remat_idx` — FAIL compile (`eval_vec_`)  
10. `sv4_e512_b8_t32_keep_idx` — FAIL compile  
11. `sv4_e1024_b8_t16_keep_idx` — FAIL compile  
12. `sv4_e1024_b8_t32_keep_idx` — FAIL compile  
13. `sv4_e1024_b16_t32_keep_idx` — FAIL compile  
14. `sv4_e1536_b8_t32_keep_idx` — FAIL compile  
15. `sv4_e2048_b8_t32_keep_idx` — FAIL compile  
16. `sv5_r64_c128_g16_t32_keep_reg` — FAIL compile (f16→f32)  
17. `sv5_r64_c128_g16_t32_ub_reload` — FAIL compile (f16→f32)  
18. `sv6_r64_c128_g32_t32` — FAIL compile  
19. `sv6_r64_c128_g32_t32_keep_in_warp` — FAIL no-pass-line (compiled, no PASS line)  
20. `sv7_r64_c128_g64_t32` — FAIL compile (f16→f32)  
21. `sv7_r64_c128_g128_t32` — FAIL compile (f16→f32)  
22. `sv8_r64_c128_g16_t32_live` — FAIL compile (f16→f32)  
23. `sv8_r64_c128_g16_t32_spill_dist` — FAIL compile (f16→f32)  
24. `sv9_e256_k1_t32_keep` — FAIL compile (`store()` arity)  
25. `sv9_e256_k8_t32_keep` — FAIL compile (same)  
26. `sv9_e256_k8_t32_remat_scores` — FAIL compile (same)  
27. `sp3_m32_h128_g32_t32_e8m0` — FAIL compile (vreg scalar op)  
28. `sp5_n64_h128_g32_qg32_t32_interleave` — FAIL compile (IndentationError / parse)  
29. `sp5_n64_h128_g32_qg32_t32_sideband` — FAIL compile (same)  

Remat tip would at most clear the two `eval_vec_` remat_idx rows among these; **not** re-measured. Dominant buckets remain f16→f32 EmitInlineSimtVF, keep_idx llvm.store, SV9 store arity, SP5/SP3.

## 3) Fresh-run attempt (blocked)

1. Started OUT `/mnt/fluxdata/happybot/st_out_rematfix_simt_asc_pto_20261013_140343` but live `build/lib/libtilelang.so` had been rolled back to **Oct 9** pre-remat binary (remat `.o`s from Oct 12 still present; `libtvm_compiler.so` Oct 12).  
2. Killed that run; started `ninja tilelang` rebuild (pulled a large tvm rebuild after CMake re-check).  
3. Both hops dropped → cannot finish rebuild or re-launch harness.

**When hops return:** rebuild/link `libtilelang.so` on tip `a3012088`, then  
`ST_VF_MODES="simt-asc simt-pto" bash oneshot_vf_mode_regress.sh`  
with cann_92b2 + nightly067; publish new `$OUT/VF_MODE_PASS_TABLE.txt`.

## Bottom line for Lok / PTO-ISA

| Question | Answer |
|----------|--------|
| All simt-asc PASS after fix? | **No proof.** Only SV4 remat e256_b8 proven. Full table still has 5 FAIL pre-fix; 3 bare sv6/sv7 FAILs unrelated to remat. |
| sv6/sv7? | Good arms (`keep_in_warp` / `multiwarp_ub`) already **PASS** on simt-asc Oct 10+12. Bare arms still **FAIL compile**. Not “MISSING because not run” on simt-asc. |
| simt-pto? | Still **33/61-ish PASS** on last full run; **29 named FAILs** listed above. Remat does not clear the bulk. |
| PTO-DSL / simd-pto twins? | Hand PTO-DSL *d path still claimed green in PR #2; **simd-pto VF-mode still 0/61**. |
