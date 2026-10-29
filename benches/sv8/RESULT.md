# SV8 G32 Case-3 e2e RESULT (t737u steer — SIMD worst vs SIMT best)

**Host:** pto-b10 / Ascend950PR_9599 camodel  
**When:** 2026-10-29T14:15:00+08:00 Asia/Hong_Kong (G32 live PASS)  
**Pack:** `/mnt/fluxdata/happybot/sv8_g16_e2e_t726u/` · box `/workspace/sv8_g16_e2e_t726u/`  
**Labels:** `reduce_axis=inner` SoftLib_axis=`1xG` `era=case3-quant-bcast` §22 → absmax expects VC/VCGMAX family  
**Shape:** R=64 C=128 **G=32** T=32  E=R*C=8192 (E kept; G retargeted from G16 per t737u)  
**Env:**
- **simt-asc:** TILELANG=`tilelang-pto-isa-st` + **cann_92b2** (`cann-9.2.0-beta.2`)
- **sv8d VL-resident:** PTO-DSL handwritten (`sv8d_vl_resident_case3.py`) + **cann_91b3** opsim; OPPROF under `out/dsl_vlres/opsim_sv8d_vlres_r64_c128_g32_t32_vl_resident/`

**Dtype footnote (era=case3-quant-bcast):** SIMT-ASC = **fp16** X/Out + f32 scale; VL-resident SIMD = **f32** X/Out (same split as G16 sv8d vs simt). Ratios are same-shape Case-3 e2e, not cross-dtype microbench SoT.

---

## 1. Analysis (hypothesis first)

### H1 — UB-BW is the G-size drawback on this e2e
**Not the live bottleneck on R64×C128.** Outer GM↔UB ≈32–64 KiB ≪ VF body on both arms (SIMT VF=56714; VL-res VF=1964 still ≫ dual-VLD floor ~64–128 cyc).

### H2 — Fused reduce+quant+eltwise with X VL/RF-resident → SIMD ~2× SIMT
**Re-tested at G=32 with an explicit VL-resident SIMD motif.**

| arm | X residency | VF_cyc | el/cyc | notes |
|-----|-------------|-------:|-------:|-------|
| simt-asc live G32 | X **reloads** from `x_ub` (fp16); `sf_inv` fragment RF | **56714** | **0.1444** | stock SimtVF |
| **sv8d VL-resident G32** | **X KEPT in VL vreg** across abs→reduce→mul; **no `RV_VLDS`** | **1964** | **4.1711** | per-group `vcmax(group=1)` (NG=2 compact PTOAS-illegal) |

**Ratio (el/cyc):** SIMD÷SIMT = 4.1711 / 0.1444 ≈ **28.9×** (SIMD faster).  
**~2× floor:** **exceeded** once X is VL-resident — opposite of G16 size-1+scratch sv8d (where SIMT÷SIMD≈4.50×).

**Caveat (SEMANTICS_GATE):** VL-resident schedule is **not** compact `vcmax(group=NG=2)` / VCGMAX-on-NG=2 (PTOAS rejects `vreg<2xf32>`). It is **2× `vcmax(group=1)`** with lane masks + size-1 recip + `vbrc` expand. Not the Gmap VL=256 NG=8 SoT either. Valid as **VL-resident Case-3 motif** measure, **INCONCLUSIVE** as inner compact-VCGMAX G32 SoT.

### H3 — membar/flag dominates small shape
**VL-resident:** no intra-VF `RV_SMEM_BAR` in harvest (scratch expand removed). Outer SET/WAIT_FLAG only. Dominant VF body: `RV_VSEL` (1024), `RV_VDIV` (256), `RV_VCMAX` (256), `RV_VDUP` (256), `RV_VMUL` (128), `RV_VAND` (128).

### G16 → G32 steer (t737u)
G=32 chosen as **SIMD-worst vs SIMT-best** group size to explore the gap. Prior G16 tip retained under COMPARE (§5).

---

## 2. Numbers table (G32 tip)

| path | arm | compile | gold | wall_us (msprof vec) | VF_us | VF_cyc | el/cyc (E=8192) | notes |
|------|-----|---------|------|---------------------:|------:|-------:|---------------:|-------|
| simt-asc | live | OK | PASS maxabs=0 | 32.77 | 31.51 | **56714** | **0.1444** | cann_92b2; VF_SIMT; R64×C128 G32 |
| simt-asc | spill_dist | OK | PASS maxabs=0 | 25.04 | 23.79 | **42819** | **0.1913** | spill faster than live this tip |
| **sv8d VL-resident** | **live** | **OK** | **PASS maxabs=1.192e-07** | **1.92** | **1.09** | **1964** | **4.1711** | KEEP X; VCMAX×256; **VLDS=0**; f32 |
| sv8d size-1 (G16 motif) | — | — | — | — | — | — | — | not re-run at G32 this tip |
| simd-pto sv8v | — | FAIL (hist) | n/a | NA | NA | NA | NA | LowerParallel stride=0/0 |

**el/cyc formula:** E / VF_cyc with E=8192. Prefer named VF / VF_SIMT instr row.

**Headline ratio:** VL-resident SIMD ÷ SIMT-ASC live ≈ **28.9×**.

---

## 3. Paths

| kind | path |
|------|------|
| SO simt G32 | `out/simt_g32/so/sv8_r64_c128_g32_t32_{live,spill_dist}.so` |
| ASC/TIR | `out/simt_g32/sources/sv8_r64_c128_g32_t32_*_{source,lowered,tir}.txt` |
| opsim simt | `out/simt_g32/opsim_sv8_r64_c128_g32_t32_{live,spill_dist}/` |
| VL-res SUMMARY | `out/dsl_vlres_g32/nohup_vlres_t737u_g32c.log` |
| VL-res opsim | `out/dsl_vlres/opsim_sv8d_vlres_r64_c128_g32_t32_vl_resident/` |
| VL-res instr_exe | `.../simulator/core0.veccore0/core0.veccore0_instr_exe.csv` |
| kernel | `benches/sv8/sv8d_vl_resident_case3.py` |
| oneshot | `oneshot_sv8_simt_g32.sh`, `run_vlres_live_g32.sh` |

---

## 4. Hypothesis verdict (one line each)
1. **UB-BW as G32 drawback on this e2e:** **FALSE** as live binder — EX/schedule (SIMT) or VL-resident VF body (SIMD); outer UB far under roof.  
2. **X VL-resident → SIMD ~2× SIMT:** **SUPPORTED and exceeded** — SIMD÷SIMT ≈ **28.9×** at G32 with KEEP-X motif (dtype footnote applies).  
3. **membar dominates VL-resident:** **FALSE** — no SMEM_BAR; VSEL/VDIV/VCMAX dominate.  
4. **Compact NG=2 VCGMAX:** **blocked** by PTOAS (`vreg<2xf32>`); measured arm uses group=1 halves — gate INCONCLUSIVE as compact-VCGMAX SoT.

---

## 5. COMPARE — prior G16 tip (t726u/t727u)

| arm | G | VF_cyc | el/cyc | motif |
|-----|--:|-------:|-------:|-------|
| simt-asc live | 16 | 28603 | 0.2864 | stock; X from UB |
| sv8d size-1 live | 16 | 128790 | 0.06361 | size-1 VMAX + scratch expand + VLDS=E |
| simt-asc live | **32** | **56714** | **0.1444** | stock; X from UB |
| sv8d VL-res live | **32** | **1964** | **4.1711** | KEEP X; VCMAX g=1 halves |

G16 SIMT÷SIMD(size-1) ≈ 4.50× (SIMT faster). G32 SIMD(VL-res)÷SIMT ≈ 28.9× (SIMD faster). Gap direction **flips** with VL-resident motif — primary lesson of t737u.

---

## 6. SEMANTICS_GATE

- simt-asc G32 live: `SEMANTICS_GATE: PASS — gold G32 Case-3 e2e measured; SoftLib_axis=1xG; era=case3-quant-bcast; not claiming SoftLib hand-craft / multi-motif SoT`
- sv8d VL-resident G32: `SEMANTICS_GATE: PASS as VL-resident Case-3 motif (KEEP X, VLDS=0); INCONCLUSIVE as inner compact VCGMAX/NG=2 SoT — PTOAS blocks vreg<2xf32>; used vcmax(group=1)×NG; not Gmap VL256 NG=8`
- SoftLib SIMT: `SEMANTICS_GATE: INCONCLUSIVE — SoftLib Case-3 hand-craft not measured this pack (A5 blocker)`
