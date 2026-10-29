# SV8 analysis — Case-3 quant e2e bcast G=32 (inner 1×G) · t737u

**Date:** 2026-10-29 Asia/Hong_Kong · ST-spine · pack **t737u** (simt-asc G32 + VL-resident SIMD)  
**Format:** [`docs/ST_REPORT_FORMAT.md`](../../docs/ST_REPORT_FORMAT.md) + `tilelanggym-st-report` · companions: `tilelanggym-st-semantics-gate` · softlib-simt-schedule (A5 blocker)  
**Primary metric:** **el/cyc = E / VF_cyc** · **E = R×C = 8192** · **`era=case3-quant-bcast`** (Case-3 · reduce + recip + bcast mul).  
**Dtype:** SIMT-ASC **fp16** X/Out; VL-resident SIMD **f32** X/Out (footnote — same split as G16 tip).  
**SOC / CANN:** Ascend950PR_9599 camodel · simt **cann_92b2** · VL-res opsim **cann_91b3**.  
**Host:** pto-b10.  
**Truth source:** pack `RESULT.md` (authoritative). **Never invent measured numbers.**  
**Never merge** with SV5–7 **`era=fp32-absmax`** SoT (Harness §13). SoftLib CCE = off main / N/A (t646). **Do not merge this PR** until Lok says yes.

**ST map:** SV8 = Case-3 e2e · tip gold **G=32** (t737u: SIMD-worst vs SIMT-best to explore gap) · `reduce_axis=inner` SoftLib_axis=`1xG`. Shape **R64×C128 G32 T32** (E=8192 kept). Prior G16 tip = COMPARE only.

### Headline — `reduce_axis=inner` · `era=case3-quant-bcast` · G=32

| # | Path | Best on this tip | Era |
|---|------|------------------|-----|
| 1 | **SIMD PTO-DSL VL-resident** | KEEP X · `vcmax(group=1)×NG` · **4.1711** (VF **1964**) | case3-quant-bcast |
| 2 | **SIMT-ASC stock** | `__simt_vf__` live · **0.1444** (VF **56714**) | case3-quant-bcast |
| 3 | **SIMT SoftLib hand-craft** | **not measured** — A5 blocker + next-cmd (§6) | — |

> SoftLib CCE: **N/A / off main** (t646).  
> simd-pto **sv8v**: historical COMPILE FAIL — not re-opened this tip.  
> **Ratio (measured):** VL-res SIMD÷SIMT el/cyc = 4.1711 / 0.1444 ≈ **28.9×** (SIMD faster). H2 ~2× **exceeded** with KEEP-X motif.

---

## 1. Aims

| Id | Aim |
|----|-----|
| **A1** | Measure SV8 **G32** Case-3 e2e SIMT-ASC vs VL-resident SIMD on R64×C128 G32 T32 E=8192 |
| **A2** | Re-test H2 with explicit VL-resident X: SIMD ~2× SIMT (el/cyc) |
| **A3** | Quantify membar/SMEM_BAR vs VF on VL-resident SIMD (vs G16 size-1 scratch) |
| **A4** | Classify bound: outer UB-BW vs EX/schedule vs VL-resident VF body |
| **A5** | SoftLib SIMT hand-craft Case-3: measured or blocker+next-cmd |
| **A6** | Document layout/schedule + X residency for both G32 arms |
| **A7** | COMPARE vs prior G16 tip (size-1 SIMD vs VL-resident) — gap direction |

---

## 2. Test-case design

| Axis | Choice |
|------|--------|
| Shape | R=64 · C=128 · **G=32** · T=32 · **E=8192** |
| Why G32 | t737u: SIMD-worst vs SIMT-best group size to explore gap |
| Datatype | SIMT **fp16** X/Out · VL-res **f32** · f32 scale · **`era=case3-quant-bcast`** |
| Gold | e2e `Out = X * (1/max(absmax_group, 1e-6))` |
| Reduce axis | **`reduce_axis=inner`** contiguous 1×G (§22) |
| SoftLib axis | **`axis=1xG`** (§21) |
| SIMT-ASC | `sv8_case3_bcast.py` · live \| spill_dist · G=32 args |
| SIMD VL-res | `sv8d_vl_resident_case3.py` · VL=64 · KEEP X · per-group `vcmax(group=1)` (NG=2 compact illegal) |
| SoftLib SIMT | **deferred** — A5 |
| SoftLib CCE | **N/A** (t646) |

**Sources:** `RESULT.md` · kernels under `benches/sv8/` · `config.toml`.

---

## 3. Layout and schedule (SIMT and SIMD)

### 3.1 SIMT-ASC G32 (`sv8_case3_bcast.py`) — A6

- **Layout:** `x_ub`/`out_ub` shared fp16; `sf_inv[R, CG]` fragment f32 (`CG=C/G=4`).
- **live:** Parallel absmax+recip → `sf_inv` fragment; consumer mul reloads **X from `x_ub`**.
- **spill_dist:** expand scales to shared then reload (layout-transform spill).
- **RF:** scale RF-resident; **X not RF-resident** (same pattern as G16).

### 3.2 SIMD VL-resident G32 (`sv8d_vl_resident_case3.py`) — A6

- **Motif:** load X VL=64 → i32-VAND abs → for each of NG=2 groups: mask other lanes to 0 → `vcmax(..., group=1)` → size-1 eps/recip → `vbrc` + `vsel` paint scale into VL → **`vmul` with KEPT X** → store Out. **No second X load.**
- **Why not `vcmax(group=NG=2)`:** PTOAS VMI residual / rejects `vreg<2xf32>` (also size-2 vmax/vdiv). Gmap SoT for absmax-only G32 is VL=256 NG=8 — not this Case-3 e2e shape path.
- **Instr (call_count):** `RV_VLDS=0` · `RV_VCMAX=256` · `RV_VDIV=256` · `RV_VSEL=1024` · `RV_VMUL=128` · `RV_VAND=128` · **no `RV_SMEM_BAR`**.

### 3.3 SoftLib CCE

**N/A** this pack (t646).

---

## 4. SIMD PTO-DSL performance (VL-resident)

| tag | status | VF_cyc | VF_us | wall_us | el/cyc | instr / schedule | gate |
|-----|--------|-------:|------:|--------:|-------:|------------------|------|
| **vl_res G32 live** | **PASS** maxabs=1.192e-07 | **1964** | 1.09 | 1.92 | **4.1711** | KEEP X; VCMAX g=1×NG; VLDS=0 | § gate |

**SEMANTICS_GATE:** PASS as VL-resident Case-3 motif (KEEP X, VLDS=0); INCONCLUSIVE as inner compact VCGMAX/NG=2 SoT — PTOAS blocks `vreg<2xf32>`; used `vcmax(group=1)×NG`; not Gmap VL256 NG=8.

**IPC / figures:** **MISSING** this push — next-cmd: harvest OPPROF → `python -m scripts.st_report plot --st sv8` (or pack-local) for VL-res + simt G32 arms; embed under `benches/sv8/plots/`.

---

## 5. SIMT (Ascend default) lowering performance

| tag | status | VF_cyc | VF_us | wall_us | el/cyc | notes | gate |
|-----|--------|-------:|------:|--------:|-------:|-------|------|
| **live G32** | **PASS** | **56714** | 31.51 | 32.77 | **0.1444** | VF_SIMT; cann_92b2 | § gate |
| spill_dist G32 | **PASS** | **42819** | 23.79 | 25.04 | **0.1913** | spill faster than live this tip | same era |

**SEMANTICS_GATE:** PASS — gold G32 Case-3 e2e measured; SoftLib_axis=1xG; era=case3-quant-bcast; not claiming SoftLib hand-craft / multi-motif SoT.

**IPC / figures:** **MISSING** — same next-cmd as §4.

---

## 6. SIMT hand-craft (SoftLib) analysis

| Aim | A5 SoftLib SIMT hand-craft Case-3 |
| Measured | **none** this tip |
| Blocker | SoftLib Case-3 motif matrix not staged on G32 |
| Next-cmd | `# SoftLib SIMT Case-3 staging (when A5 opens): bash scripts/softlib_simt_stage.sh` |

**SEMANTICS_GATE:** INCONCLUSIVE — SoftLib Case-3 hand-craft not measured this pack (A5 blocker).

---

## 7. Running IPC + figures

| Arm | Running IPC / PR9 figures |
|-----|---------------------------|
| VL-resident G32 | **MISSING** + next-cmd §4 |
| SIMT-ASC G32 | **MISSING** + next-cmd §4 |
| SoftLib SIMT | N/A (unmeasured) |

---

## 8. Conclusion

| Arm | el/cyc | Verdict |
|-----|-------:|---------|
| **SIMD VL-resident G32** | **4.1711** | Primary SIMD tip — KEEP X works; ~29× SIMT; not compact-VCGMAX SoT |
| **SIMT-ASC G32 live** | **0.1444** | Stock SimtVF; X from UB; gold PASS |
| **SoftLib SIMT** | — | A5 blocker |

**Ratios (measured only):** SIMD÷SIMT ≈ **28.9×**. H2 ~2× **exceeded** with VL-resident motif.

**COMPARE (A7):** G16 size-1 SIMD was **slower** than SIMT (SIMT÷SIMD≈4.50×). G32 VL-resident **flips** the gap. Motif (KEEP X vs size-1+scratch), not G alone, drives direction.

**Bound class (A4):** outer UB-BW **not** binder; SIMT = EX/schedule + X reload; VL-res = VSEL/VDIV/VCMAX body without membar expand.

**Do not merge** until Lok review.
