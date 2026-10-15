# ST suite probe: sv6 harness + sv4 keep_idx llvm.store (2026-10-13)

## Env
- **Host:** pto-b10 (edgexpert-59a6)
- **Tip SHA:** `a3012088cc61a7fac36c50ae19caf2524d5bf804` (`fix/simtvf-remat-idx-evalset-vec`)
- **libtilelang.so mtime:** 2026-10-13 14:24:50 +0800
- **ST_VF_MODE:** simt-pto
- **CANN:** `/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2`
- **PTOAS / camodel:** `/mnt/fluxdata/happybot/projects/.camodel_deps_nightly067` (nightly **0.67**, commit `1b36b6f5`)
- **OUT:** `/mnt/fluxdata/happybot/st_out_sv6_simt_pto_probe_20261013_162726`
- **Prior OUT (ref):** `/mnt/fluxdata/happybot/st_out_nightly067_simt_pto_20261010_170837/`
- **Probe log:** `/mnt/fluxdata/happybot/st_out_sv6_simt_pto_probe_20261013_162726/probe.log`

---

## Task 1 — sv6 simt-pto harness-only cases

### Cases
| Case | Prior claim (Oct 10) | Fresh probe (tip a3012088) |
|------|----------------------|----------------------------|
| `sv6_r64_c128_g32_t32` (oneshot bare TAG) | COMPILE_FAIL rc=0, no snippet | **Harness FALSE FAIL** |
| `sv6_r64_c128_g32_t32_keep_in_warp` | COMPILE_OK, FAIL no-pass-line / MISSING opsim | **PASS** (compile + opsim) |

### Evidence — bare TAG is a harness tag mismatch
Oneshot invokes:
```bash
compile_run_generic ... sv6_r64_c128_g32_t32 64 128 32 32   # no arm arg
```
Kernel (`sv6_reduce_mid_eltwise.py`) defaults `arm=keep_in_warp` and writes:
```text
tag = f"sv6_r{R}_c{C}_g{G}_t{threads}_{arm}"  # → ..._keep_in_warp
```
so the `.so` lands as `sv6_r64_c128_g32_t32_keep_in_warp.so`, while oneshot checks `so/sv6_r64_c128_g32_t32.so`.

Fresh repro (`.../logs/compile_sv6_r64_c128_g32_t32.log`):
```text
oneshot_style: rc=0 expected_so=.../so/sv6_r64_c128_g32_t32.so exists=no
VERDICT_HARNESS_BARE: COMPILE_FAIL rc=0 is FALSE FAIL — kernel default arm tags *_keep_in_warp.so but oneshot expects sv6_r64_c128_g32_t32.so
```
**Verdict `sv6_r64_c128_g32_t32`:** harness false FAIL (not a real TileLang/PTOAS compile failure). Fix oneshot TAG to include `_keep_in_warp` (or pass arm explicitly and match TAG).

### Evidence — keep_in_warp is real PASS when opsim runs
- Compile: `COMPILE_OK` → `.../so/sv6_r64_c128_g32_t32_keep_in_warp.so` (35872 bytes)
- Opsim: `PASSLINE_keep_in_warp: PASS maxabs=5.960464477539063e-08 R=64 C=128 G=32 CG=4`
- Log: `.../opsim_sv6_r64_c128_g32_t32_keep_in_warp.log`

Prior Oct 10 `FAIL no-pass-line` / report `MISSING` was because oneshot treated the bare TAG as COMPILE_FAIL and **skipped opsim**; harvest still saw the `*_keep_in_warp.so` and marked MISSING. Not a kernel failure.

**Verdict `sv6_r64_c128_g32_t32_keep_in_warp`:** now PASS (was harness false FAIL / missing opsim).

---

## Task 2 — 8× sv4 `*_keep_idx` `llvm.store`: TileLang emit vs PTOAS

### Cases (all same root)
`sv4_e256_b8_t32_keep_idx`, `sv4_e256_b16_t32_keep_idx`, `sv4_e512_b8_t32_keep_idx`, `sv4_e1024_b8_t16_keep_idx`, `sv4_e1024_b8_t32_keep_idx`, `sv4_e1024_b16_t32_keep_idx`, `sv4_e1536_b8_t32_keep_idx`, `sv4_e2048_b8_t32_keep_idx`

Prior + fresh: every case fails at `kernel.pto:68:11` with:
```text
error: failed to legalize operation 'llvm.store' that was explicitly marked illegal
Error: Failed to lower VPTO to LLVM modules.
```

### Method
1. Re-emit representative `sv4_e256_b8_t32_keep_idx` with `TILELANG_CLEANUP_TEMP_FILES=0`; capture `kernel.ptodsl.py` + `kernel.pto`.
2. Contrast PASS sibling `sv4_e256_b8_t32_remat_idx_calc` (same shapes, remat path).
3. Re-run `ptoas --pto-arch=a5 --pto-backend=vpto` alone on captured `kernel.pto`.

Artifacts under OUT:
- `pto_capture/sv4_e256_b8_t32_keep_idx/{kernel.ptodsl.py,kernel.pto,ptoas_stderr.txt}`
- `pto_capture/sv4_e256_b8_t32_remat_idx_calc/{kernel.ptodsl.py,kernel.pto}` (ptoas_rc=0, `.so` built)

### keep_idx PTODSL (TileLang emit) — keeps idxs RF buffer
```python
with pto.simt(32, 1, 1):
  idxs = pto.alloc_buffer((8,), pto.i32)   # <-- keep_idx only
  acc = pto.alloc_buffer((8,), pto.f32)
  ...
  for i in pto.range(0, 4, 1, unroll="enable"):
    pto.store(pto.load(..., contiguous=2), idxs, i * 2)  # vector store into i32 buffer
```

### keep_idx MLIR (before VPTO legalize) — illegal op at line 68
```mlir
%32 = llvm.alloca %31 x i32 : (i32) -> !llvm.ptr
...
%66 = pto.load %64[%65] : !pto.ptr<vector<2xsi32>, ub> -> vector<2xsi32>
...
%72 = llvm.getelementptr %32[%71] : (!llvm.ptr, i64) -> !llvm.ptr, i8
llvm.store %66, %72 : vector<2xsi32>, !llvm.ptr    // line 68 — VPTO illegal
```
`ptoas` alone on this file → rc=1, same legalize error (not a harness/wrapper issue).

### remat_idx_calc sibling — no idxs buffer; PASS
```python
with pto.simt(32, 1, 1):
  acc = pto.alloc_buffer((8,), pto.f32)   # no idxs
  ...
  # index recomputed inline; no i32 vector store into alloca
```
Still emits `llvm.store ... vector<2xf32>` for `acc` (legal under VPTO) → **ptoas_rc=0**, `COMPILE_OK` `.so`.

### Sameness of all 8
All eight prior `compile_fail_*.txt` cite `kernel.pto:68:11` + illegal `llvm.store`. Same keep_idx codegen shape (idxs RF + contiguous i32 vector store).

### Verdict
**TileLang emit (keep_idx path) — not a clean PTOAS false-reject of valid PTO.**

- The captured input already contains VPTO-illegal `llvm.store vector<2xsi32>`.
- PASS sibling proves the same toolchain accepts `llvm.store vector<2xf32>` and succeeds when idxs are rematerialized instead of kept.
- Root cause for the ST FAIL bucket: **codegen_pto / keep_idx lowering** produces an i32 RF (`alloc_buffer` + contiguous store) that PTODSL→MLIR maps to a VPTO-illegal `llvm.store`.
- Secondary note: PTOAS/VPTO could add i32-vector store legalization, but that is not required to explain the suite FAILs; remat already shows a TileLang-side legal pattern.

**Fix ownership:** TileLang (keep_idx on target=pto) — scalarize i32 stores, emit a VPTO-legal buffer op, or route keep_idx through remat-equivalent emit. Not a PTOAS-only ticket without first changing the emitted IR.

---

## Summary for Lok / PTO-ISA
1. **sv6 bare:** harness false FAIL (TAG vs default-arm `.so` name).
2. **sv6 keep_in_warp:** real **PASS** on simt-pto once opsim runs.
3. **8× sv4 keep_idx llvm.store:** **TileLang emit** (shared root); PTOAS correctly rejects illegal op present in emitted `kernel.pto`. Fix in TileLang keep_idx→PTO path.
