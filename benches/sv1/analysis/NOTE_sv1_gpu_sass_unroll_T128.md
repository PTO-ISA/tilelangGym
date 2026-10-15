# GPU (CUDA) T≈128 + unroll SASS vs Ascend SimtVF SV1

**Date:** 2026-10-15 (HKT)  
**Machine:** pto-b10 GB10 (sm_121), nvcc/ptxas 13.1  
**Question:** Does GPU SASS **reschedule** and use **float4 (.128)** for elementwise add with blockDim≈128 + loop unroll?

## Short answer

| Question | Answer | Evidence |
|----------|--------|----------|
| Uses float4 / **LDG.E.128** when source is float4? | **YES** | Hand `float4` kernels + TILE2048 recreate → `LDG.E.128` / `STG.E.128` / `LDS.128` / `STS.128` |
| Compiler **reschedules** (ILP / load-hoist before use)? | **YES** | ILP4: **8× LDG.128** then **16× FADD** then **4× STG.128**; recreate interleaves LDS/FADD/STS |
| TileLang shipped SV1 cubin always .128? | **NO (caveat)** | Existing `tcopy_E16384.sass` is **LDG.E.64 / STS.64** despite float4 in CUDA — still reschedules (6 LDG before first FADD) |
| SimtVF match? | **NO** | SimtVF stays **LDS/STS .64**, IADD-heavy, FADD `waitBitMask≠0` ~50%, Body4 IPC ~0.37 |

**Bottom line:** GPU **does** use float4→`.128` and **does** software-pipeline / reschedule independent loads ahead of FADD when the kernel is written (or lowered) that way. Ascend SimtVF SV1 does **not** get an equivalent pack/schedule win today — that matches the “fix schedule/unroll at fixed T, don’t pile warps” lever.

---

## 1. Kernels examined

### A. Hand CUDA — `vec_add_f4_unroll` / `vec_add_f4_ilp4` (blockDim=128)

- `float4*` global LD/ST, `#pragma unroll 4`, and an ILP-4 variant (4 independent float4s per iter).
- Compiled: `nvcc -O3 -arch=sm_121` → cubin/object; SASS via `cuobjdump -sass` / `nvdisasm` (13.1).
- Artifacts: `/tmp/sv1_gpu_sass_unroll_T128/vec_f4.sm_121.o`, `vec_f4.sm_121.sass`, `vec_add_f4_dev.cu`.

### B. Recreated TileLang TILE=2048 THREADS=128 float4 body

- Matches `kernel_tcopy_E16384_T2048.cu.txt` structure: `#pragma unroll` ×4 float4 gmem↔smem, float4 add, float4 store.
- Artifacts: `tcopy_T2048_recreate.cu` / `.ptx` / `.sass`.

### C. Prior TileLang SV1 GPU dump (already on disk)

- CUDA source **does** use float4 + `#pragma unroll` (TILE≥512 @ T128).
- Shipped SASS `sass_mix/tcopy_E16384.sass`: **`.64` only** (see caveat below).
- Perf: TILE2048@T128 ≈ **1.035 el/cyc** (vs Ascend SimtVF TILE2048@T128 ≈ **3.22 el/cyc** on camodel — different roofs/ISAs; not apples-to-apples absolute).

---

## 2. Float4 / `.128` — YES (when float4 is actually in the ISA path)

### Hand ILP4 SASS (sm_121) — census

| Opcode | Count |
|--------|------:|
| **LDG.E.128** | 18 |
| **STG.E.128** | 9 |
| **FADD** | 36 |

### Quoted schedule — independent loads hoisted, then FADD, then store

```text
/*0100*/  LDG.E.128.CONSTANT R4,  desc[UR6][R36.64] ;
/*0120*/  LDG.E.128.CONSTANT R12, desc[UR6][R36.64+0x10] ;
/*0130*/  LDG.E.128.CONSTANT R8,  desc[UR6][R32.64] ;
/*0140*/  LDG.E.128.CONSTANT R16, desc[UR6][R32.64+0x10] ;
/*0150*/  LDG.E.128.CONSTANT R24, desc[UR6][R32.64+0x20] ;
/*0160*/  LDG.E.128.CONSTANT R20, desc[UR6][R36.64+0x20] ;
/*0170*/  LDG.E.128.CONSTANT R28, desc[UR6][R36.64+0x30] ;
/*0180*/  LDG.E.128.CONSTANT R32, desc[UR6][R32.64+0x30] ;
/*0190*/  FADD R4,  R4,  R8 ;
/*01a0*/  FADD R5,  R5,  R9 ;
/*01c0*/  FADD R6,  R6,  R10 ;
/*01d0*/  FADD R7,  R7,  R11 ;
… (more FADD on other float4 lanes) …
/*02c0*/  STG.E.128 desc[UR6][R8.64], R4 ;
/*02e0*/  STG.E.128 desc[UR6][R8.64+0x10], R12 ;
/*02f0*/  STG.E.128 desc[UR6][R8.64+0x20], R20 ;
/*0300*/  STG.E.128 desc[UR6][R8.64+0x30], R28 ;
```

**Reschedule read:** eight `.128` loads (A/B × 4 float4s) issue **before** any FADD that consumes them — classic ptxas latency-hiding / ILP, not “LD; FADD; ST” per element.

### PTX confirms v4

```text
ld.global.nc.v4.b32  {%r13..%r16}, [%rd7];
ld.global.nc.v4.b32  {%r17..%r20}, [%rd9];
ld.global.nc.v4.b32  {%r21..%r24}, [%rd7+16];
… (8× v4 loads) …
add.f32 …
st.global.v4.b32 …
```

### TILE2048 recreate SASS — pack `.128` end-to-end

| Opcode | Count |
|--------|------:|
| **LDG.E.128** | 8 |
| **STS.128** | 12 |
| **LDS.128** | 12 |
| **STG.E.128** | 4 |
| **FADD** | 16 |

```text
/*0150*/  LDG.E.128 R8,  desc[UR8][R36.64+-0x1000] ;
/*0160*/  LDG.E.128 R12, desc[UR8][R36.64+-0x800] ;
/*0170*/  LDG.E.128 R4,  desc[UR8][R36.64] ;
… more LDG.E.128 …
/*01b0*/  STS.128 [R3], R8 ;
… STS interleaved with more LDG …
/*02a0*/  LDS.128 R12, [R3] ;
/*02b0*/  LDS.128 R16, [R2] ;
… LDS burst …
/*0310*/  FADD R12, R12, R16 ;
/*0320*/  FADD R13, R13, R17 ;
/*0350*/  LDS.128 R16, [R3+0x1000] ;   ← more LDS between FADDs
/*0360*/  FADD R4, R4, R8 ;
/*03a0*/  STS.128 [R0], R12 ;
```

**Reschedule read:** gmem LD burst → smem ST; after barrier region, **LDS.128 burst**, FADD, with **extra LDS/STS mixed in** — not a rigid LDS→FADD→STS same-element chain.

### TileLang CUDA source (TILE2048 @ T128) — float4 + unroll in C++

```text
27:  for (int t = 0; t < 8; ++t) {
29:    #pragma unroll
30:    for (int i = 0; i < 4; ++i) {
31:      *(float4*)(A_shared + …) = *(float4*)(A + …);
…
37:    #pragma unroll
38:    for (int i_2 = 0; i_2 < 4; ++i_2) {
40:        float4 v_ = *(float4*)(A_shared + …);
41:        float4 v__1 = *(float4*)(B_shared + …);
42:        *(float2*)(&(__1.x)) = tl::add2(…);  // compute as add2×2
43:        *(float2*)(&(__1.z)) = tl::add2(…);
```

MIX note: first float4 mem path at **TILE=512**; TILE2048 = float4(mem)+add2×2, **1.035 el/cyc**.

---

## 3. Caveat — shipped TileLang `tcopy_E16384.sass` is `.64`

Static SASS census for existing dump:

| Opcode | Count |
|--------|------:|
| LDG.E.64.CONSTANT | 128 |
| STS.64 | 192 |
| FADD | 128 |
| **LDG.E.128** | **0** |

Still shows **reschedule**:

```text
/*0080*/  LDG.E.64 … R8 …
/*0090*/  LDG.E.64 … R10 …
/*00a0*/  LDG.E.64 … R12 …
/*00b0*/  LDG.E.64 … R14 …
/*00c0*/  LDG.E.64 … R4 …
/*00d0*/  LDG.E.64 … R2 …
/*0170*/  STS.64 [R252], R8 ;
/*0180*/  FADD R6, R8, R10 ;     ← 6 LDG before first FADD
/*0190*/  FADD R7, R9, R11 ;
/*01a0*/  STS.64 [R252+0x400], R10 ;
/*01b0*/  STS.64 [R252+0x800], R6 ;
```

Later: **huge LDG.64 burst** (dozens of independent offsets) before consuming FADDs — software pipeline across many elements.

**Why .64 not .128 here?** Likely TileLang/ptxas path on this cubin (descriptor/CONSTANT, `tl::add2`, or that particular specialized shape) split float4 CUDA into `.64` SASS. Hand float4 + clean recreate **do** emit `.128`. So: **source float4 ≠ guaranteed SASS .128**, but **GPU can and does emit .128** for the same algorithmic pattern.

---

## 4. Side-by-side vs Ascend SimtVF Body4

| | **GPU CUDA (float4 unroll / ILP)** | **Ascend SimtVF SV1** |
|--|--|--|
| Pack | **`.128`** (hand/recreate); TL dump sometimes `.64` | **`.64` only** (LDS/STS); never `.128` on path |
| Addr / index tax | Low in hand path (IMAD/SHF setup); TL dump has UIADD/IADD for tiling | **IADD ~31%** of Body4 |
| Dep / wait | Compiler hoists independent LDGs; multi-elem ILP | Same-warp LDS→FADD gap ~2.5c; **FADD waitBitMask≠0 ~50%** |
| Issue mix | LDG/STG/FADD (+ LDS/STS if shared) | Body4: LDS 27.5% / STS 13.8% / IADD 31.2% / **FADD 27.5%** (all `.64`) |
| Mid Body IPC | N/A same metric; GB10 TILE2048@T128 ~**1.04 el/cyc** | Mid-median Body4 IPC **~0.37** (~9% of mix-cap 4) → ~**3.2 el/cyc** |
| T↑ fill | Helps modestly (thr 256–512) | Mid-median flats by T=256–512; only rare mid-peak rises |
| Lever | Unroll + float4 + let ptxas reschedule | **Schedule/unroll at fixed T**; don’t rely on wrap count |

SimtVF Body4 mix (TILE2048@T128, pack **`.64` only** — not `.128`):

```text
LDS.64  27.5%   STS.64  13.8%   IADD  31.2%   FADD  27.5%
FADD share × mix-cap4 → clamped by FADD pipe to 1.0 → roof 32 el/cyc
Measured FADD-IPC ~0.10 → ~3.2 el/cyc (~10% of 32); consistent with mix + stall
```

---

## 5. Takeaways for SimtVF

1. **GPU proves the schedule story:** with T=128 and unroll, SASS issues many independent wide loads before FADD — latency hidden without needing huge T.
2. **float4/.128 is real on GPU** for this pattern; SimtVF stuck at `.64` leaves 2× bytes/op on the table vs GPU’s best path.
3. **IADD tax on SimtVF** has no analog in the clean GPU float4 ILP kernel (index math folded into IMAD.WIDE / imm offsets).
4. Actionable on Ascend: unroll + imm-offset LDS/STS (cut IADD), interleave independent elems, pursue **`.128` pack** — same direction GPU already takes.

---

## 6. Artifact paths

On pto-b10:

```text
/tmp/sv1_gpu_sass_unroll_T128/
  vec_add_f4_dev.cu
  vec_f4.sm_121.o / .sass / .ptx
  tcopy_T2048_recreate.cu / .ptx / .sass
  log4.txt
/tmp/sv1_gpu_sm1/
  kernel_tcopy_E16384_T2048.cu.txt
  sass_mix/tcopy_E16384.sass
  MIX_sv1_gpu_tile16k.txt
```

Box / note:

```text
/workspace/sv1_sv_analysis/data/NOTE_sv1_gpu_sass_unroll_T128.md
/workspace/sv1_sv_analysis/data/COMPARE_sv1_simt_vs_gpu_tile16k.txt
```
