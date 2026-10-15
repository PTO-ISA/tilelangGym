# simt-pto ST FAIL classification (29 cases)

**Question (Lok):** for the 29 simt-pto FAILs, mainly TileLang PTO codegen vs real PTOAS?

**Answer:** **Yes — mainly TileLang→PTO codegen (27/29).** None of the 29 is a clean “valid PTO accepted then PTOAS/runtime dies” miss. Two are harness/bookkeeping (`sv6` bare + `keep_in_warp` no-pass-line).

| Primary fault | Count |
|---|---:|
| **TileLang→PTO codegen** | **27** |
| **Harness/kernel** | **2** |
| **PTOAS** | **0** |
| **Ambiguous** | **0** |

**OUT (authoritative):** `/mnt/fluxdata/happybot/st_out_nightly067_simt_pto_20261010_170837/`  
**Table:** `VF_MODE_PASS_TABLE.txt` → simt-pto **33 PASS / 29 FAIL**  
**Env:** PTOAS nightly **0.67** @ commit `1b36b6f5` / generated 2026-10-09 (Actions `37994539624`), deps `.camodel_deps_nightly067`, TileLang tip at OUT time **`31483438`**, cann_92b2  
**Logs root:** `…/simt-pto/logs/`

---

## Bucket summary

### A. TileLang→PTO codegen — Unsupported cast f16→f32 (11)
`codegen_pto.cc` Fatal / `InternalError: Unsupported PTO cast: float16 -> float32` during EmitInlineSimtVF / BuildTileLangPTO.

| Case | Evidence (path) |
|---|---|
| `sv2_r32_c32_t32_frag_live` | `codegen_pto.cc:4147: Fatal: Unsupported PTO cast: float16 -> float32` — `simt-pto/logs/compile_sv2_r32_c32_t32_frag_live.log` |
| `sv2_r32_c32_t32_reload` | same — `…/compile_sv2_r32_c32_t32_reload.log` |
| `sv3_m24_vl64_k16_t32_keep` | same — `…/compile_sv3_m24_vl64_k16_t32_keep.log` |
| `sv3_m32_vl64_k16_t32_keep` | same — `…/compile_sv3_m32_vl64_k16_t32_keep.log` |
| `sv3_m32_vl64_k16_t32_split_cm16` | same — `…/compile_sv3_m32_vl64_k16_t32_split_cm16.log` |
| `sv5_r64_c128_g16_t32_keep_reg` | same — `…/compile_sv5_r64_c128_g16_t32_keep_reg.log` |
| `sv5_r64_c128_g16_t32_ub_reload` | same — `…/compile_sv5_r64_c128_g16_t32_ub_reload.log` |
| `sv7_r64_c128_g64_t32` | `InternalError: Unsupported PTO cast: float16 -> float32` — `…/compile_fail_sv7_r64_c128_g64_t32_multiwarp_ub.txt` |
| `sv7_r64_c128_g128_t32` | same — `…/compile_fail_sv7_r64_c128_g128_t32_multiwarp_ub.txt` |
| `sv8_r64_c128_g16_t32_live` | `codegen_pto.cc:4147: Fatal: Unsupported PTO cast…` — `…/compile_sv8_r64_c128_g16_t32_live.log` |
| `sv8_r64_c128_g16_t32_spill_dist` | same — `…/compile_sv8_r64_c128_g16_t32_spill_dist.log` |

### B. TileLang→PTO codegen — `llvm.store` illegal after emit (8)
TileLang emits `kernel.pto`; VPTO legalize then rejects `llvm.store` (symptom in PTOAS pipeline, root cause = emitted PTO/memory pattern from codegen). Parent label list treats this as codegen.

| Case | Evidence (path) |
|---|---|
| `sv4_e256_b8_t32_keep_idx` | `failed to legalize operation 'llvm.store' that was explicitly marked illegal` / `Failed to lower VPTO to LLVM modules.` — `…/compile_fail_sv4_e256_b8_t32_keep_idx.txt` |
| `sv4_e256_b16_t32_keep_idx` | same pattern — `…/compile_fail_sv4_e256_b16_t32_keep_idx.txt` |
| `sv4_e512_b8_t32_keep_idx` | same — `…/compile_fail_sv4_e512_b8_t32_keep_idx.txt` |
| `sv4_e1024_b8_t16_keep_idx` | same — `…/compile_fail_sv4_e1024_b8_t16_keep_idx.txt` |
| `sv4_e1024_b8_t32_keep_idx` | same — `…/compile_fail_sv4_e1024_b8_t32_keep_idx.txt` |
| `sv4_e1024_b16_t32_keep_idx` | same — `…/compile_fail_sv4_e1024_b16_t32_keep_idx.txt` |
| `sv4_e1536_b8_t32_keep_idx` | same — `…/compile_fail_sv4_e1536_b8_t32_keep_idx.txt` |
| `sv4_e2048_b8_t32_keep_idx` | same — `…/compile_fail_sv4_e2048_b8_t32_keep_idx.txt` |

### C. TileLang→PTO codegen — AscendThreadSync `eval_vec_` (2)
Fails in TileLang pass `AscendThreadSync("shared.dyn")` / `SimtVFThreadSyncMutator` before PTO emit.

| Case | Evidence (path) |
|---|---|
| `sv4_e256_b8_t32_remat_idx` | `TVM_FFI_ICHECK(eval_vec_);` → `InternalError: Check failed: (eval_vec_) is false:` — `…/compile_fail_sv4_e256_b8_t32_remat_idx.txt` |
| `sv4_e256_b16_t32_remat_idx` | same — `…/compile_fail_sv4_e256_b16_t32_remat_idx.txt` |

### D. TileLang→PTO codegen — PTODSL `store()` arity (3)
Generated `kernel.ptodsl.py` calls `pto.store` with 4 positional args.

| Case | Evidence (path) |
|---|---|
| `sv9_e256_k1_t32_keep` | `pto.store(..., idxs, i * 2)` → `TypeError: store() takes from 2 to 3 positional arguments but 4 were given` — `…/compile_fail_sv9_e256_k1_t32_keep.txt` |
| `sv9_e256_k8_t32_keep` | same — `…/compile_fail_sv9_e256_k8_t32_keep.txt` |
| `sv9_e256_k8_t32_remat_scores` | same — `…/compile_fail_sv9_e256_k8_t32_remat_scores.txt` |

### E. TileLang→PTO codegen — bad emitted Python indent (2)

| Case | Evidence (path) |
|---|---|
| `sp5_n64_h128_g32_qg32_t32_interleave` | `IndentationError: expected an indented block after 'for' statement on line 37` — `…/compile_fail_sp5_n64_h128_g32_qg32_t32_interleave.txt` |
| `sp5_n64_h128_g32_qg32_t32_sideband` | `IndentationError: … on line 38` — `…/compile_fail_sp5_n64_h128_g32_qg32_t32_sideband.txt` |

### F. TileLang→PTO codegen — e8m0 emit uses vreg as Vec init scalar (1)

| Case | Evidence (path) |
|---|---|
| `sp3_m32_h128_g32_t32_e8m0` | `TypeError: runtime scalar operators only support index/int/float values, got !pto.vreg<64xf32>` in `pto.Vec(..., init=…)` — `…/compile_fail_sp3_m32_h128_g32_t32_e8m0.txt` |

### G. Harness/kernel (2)

| Case | Evidence (path) |
|---|---|
| `sv6_r64_c128_g32_t32` | `COMPILE_FAIL sv6_r64_c128_g32_t32 rc=0` with **no** TileLang/PTOAS error snippet; compile log only builds `keep_in_warp` SO — `oneshot_vf_mode_regress.log` + `…/logs/compile_sv6_r64_c128_g32_t32.log`; `FAIL_REASONS.txt`: `COMPILE_FAIL (no snippet)` |
| `sv6_r64_c128_g32_t32_keep_in_warp` | `COMPILE_OK` + `.so` present, but table **FAIL no-pass-line**; report status **MISSING** / `(no instr csv)`; no `opsim_sv6_…` dir — `…/reports/sv6_r64_c128_g32_t32_keep_in_warp.md`, `VF_MODE_PASS_TABLE.txt` |

---

## Per-case primary label (all 29)

| # | Case | Primary | One-line evidence |
|---|---|---|---|
| 1 | `sv2_r32_c32_t32_frag_live` | TileLang→PTO codegen | Unsupported PTO cast f16→f32 (`codegen_pto.cc:4147`) |
| 2 | `sv2_r32_c32_t32_reload` | TileLang→PTO codegen | same |
| 3 | `sv3_m24_vl64_k16_t32_keep` | TileLang→PTO codegen | same |
| 4 | `sv3_m32_vl64_k16_t32_keep` | TileLang→PTO codegen | same |
| 5 | `sv3_m32_vl64_k16_t32_split_cm16` | TileLang→PTO codegen | same |
| 6 | `sv4_e256_b8_t32_keep_idx` | TileLang→PTO codegen | illegal `llvm.store` / Failed to lower VPTO |
| 7 | `sv4_e256_b8_t32_remat_idx` | TileLang→PTO codegen | AscendThreadSync `eval_vec_` ICHECK |
| 8 | `sv4_e256_b16_t32_keep_idx` | TileLang→PTO codegen | illegal `llvm.store` |
| 9 | `sv4_e256_b16_t32_remat_idx` | TileLang→PTO codegen | AscendThreadSync `eval_vec_` |
| 10 | `sv4_e512_b8_t32_keep_idx` | TileLang→PTO codegen | illegal `llvm.store` |
| 11 | `sv4_e1024_b8_t16_keep_idx` | TileLang→PTO codegen | illegal `llvm.store` |
| 12 | `sv4_e1024_b8_t32_keep_idx` | TileLang→PTO codegen | illegal `llvm.store` |
| 13 | `sv4_e1024_b16_t32_keep_idx` | TileLang→PTO codegen | illegal `llvm.store` |
| 14 | `sv4_e1536_b8_t32_keep_idx` | TileLang→PTO codegen | illegal `llvm.store` |
| 15 | `sv4_e2048_b8_t32_keep_idx` | TileLang→PTO codegen | illegal `llvm.store` |
| 16 | `sv5_r64_c128_g16_t32_keep_reg` | TileLang→PTO codegen | Unsupported PTO cast f16→f32 |
| 17 | `sv5_r64_c128_g16_t32_ub_reload` | TileLang→PTO codegen | same |
| 18 | `sv6_r64_c128_g32_t32` | Harness/kernel | COMPILE_FAIL rc=0, no error snippet |
| 19 | `sv6_r64_c128_g32_t32_keep_in_warp` | Harness/kernel | COMPILE_OK but FAIL no-pass-line / MISSING |
| 20 | `sv7_r64_c128_g64_t32` | TileLang→PTO codegen | Unsupported PTO cast f16→f32 |
| 21 | `sv7_r64_c128_g128_t32` | TileLang→PTO codegen | same |
| 22 | `sv8_r64_c128_g16_t32_live` | TileLang→PTO codegen | same |
| 23 | `sv8_r64_c128_g16_t32_spill_dist` | TileLang→PTO codegen | same |
| 24 | `sv9_e256_k1_t32_keep` | TileLang→PTO codegen | `store()` arity 4 vs 2–3 |
| 25 | `sv9_e256_k8_t32_keep` | TileLang→PTO codegen | same |
| 26 | `sv9_e256_k8_t32_remat_scores` | TileLang→PTO codegen | same |
| 27 | `sp3_m32_h128_g32_t32_e8m0` | TileLang→PTO codegen | Vec init got `!pto.vreg<64xf32>` |
| 28 | `sp5_n64_h128_g32_qg32_t32_interleave` | TileLang→PTO codegen | IndentationError after `for` |
| 29 | `sp5_n64_h128_g32_qg32_t32_sideband` | TileLang→PTO codegen | IndentationError after `for` |

---

## Notes

- **Not mainly PTOAS:** every compile-path miss dies in TileLang emit/passes or in PTODSL produced by TileLang; the `llvm.store` bucket surfaces inside VPTO legalize but is driven by the emitted `kernel.pto`, not a post-good-emit simulator crash.
- **Contrast:** `sv4_*_remat_idx_calc` and `sv9_e256_k8_t32_remat_idx` **PASS** on the same OUT — keep_idx / remat_idx / remat_scores paths diverge inside TileLang, not a blanket PTOAS outage.
- **Checkout tip on hop now** may be newer (`a3012088` remat fix); this classification is strictly from the Oct 10 OUT at tip **`31483438`**.
