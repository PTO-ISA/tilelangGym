# SV8 analysis — Case-3 quant e2e bcast G=32 (inner 1×G) · t740u SoftLib

**Date:** 2026-10-29 Asia/Hong_Kong · ST-spine · pack **t740u** (SoftLib SIMT Case-3 e2e matrix) + prior **t737u** (simt-asc G32 + VL-resident SIMD)  
**Format:** [`docs/ST_REPORT_FORMAT.md`](../../docs/ST_REPORT_FORMAT.md) + `tilelanggym-st-report` · companions: `tilelanggym-st-semantics-gate` · `tilelanggym-softlib-simt-schedule`  
**Primary metric:** **el/cyc = E / VF_cyc** · **E = R×C = 8192** · **`era=case3-quant-bcast`** (Case-3 · reduce + recip + bcast mul).  
**Dtype:** SIMT-ASC **fp16** X/Out; SoftLib SIMT **f32**; VL-resident SIMD **f32** (SoftLib÷SIMD ratio is same-dtype f32).  
**SOC / CANN:** Ascend950PR_9599 camodel · SoftLib/simt **cann_92b2** · VL-res opsim **cann_91b3**.  
**Host:** pto-b10.  
**Truth source:** pack RESULT (authoritative). **Never invent measured numbers.**  
**Never merge** with SV5–7 **`era=fp32-absmax`** SoT (Harness §13). SoftLib CCE = off main / N/A (t646). **Do not merge this PR** until Lok says yes.

**ST map:** SV8 = Case-3 e2e · tip gold **G=32** · `reduce_axis=inner` SoftLib_axis=`1xG`. Shape **R64×C128 G32** (E=8192). SoftLib layout inherited from absmax G32 **wc dual_drain 8.22** (MATCHES_1xG) — **not** G16 wr 10.98 (DIFFERENT_GOLD).

### Headline — `reduce_axis=inner` · `era=case3-quant-bcast` · G=32

| # | Path | Best on this tip | Era |
|---|------|------------------|-----|
| 1 | **SIMT SoftLib hand-craft** | dual_drain loadall+absall T512 · **6.7702** (VF **1210**) | case3-quant-bcast |
| 2 | **SIMD PTO-DSL VL-resident** | KEEP X · `vcmax(group=1)×NG` · **4.1711** (VF **1964**) | case3-quant-bcast |
| 3 | **SIMT-ASC stock** | `__simt_vf__` live · **0.1444** (VF **56714**) | case3-quant-bcast |

> SoftLib CCE: **N/A / off main** (t646).  
> simd-pto **sv8v**: historical COMPILE FAIL — not re-opened this tip.  
> **Ratios (measured):** SoftLib÷SIMD = 6.7702 / 4.1711 ≈ **1.62×** (SoftLib faster). SIMD÷SoftLib ≈ **0.62×**. H2 ~2× **SIMD** vs SoftLib **falsified** (inverted). SoftLib÷SIMT-ASC ≈ **46.9×**.

---

## 1. Aims

| Id | Aim |
|----|-----|
| **A1** | Measure SV8 **G32** Case-3 e2e SIMT-ASC vs VL-resident SIMD on R64×C128 G32 E=8192 |
| **A2** | Re-test H2 ~2× with SoftLib SIMT at ~8–11 el/cyc-class layout (not SIMT-ASC 0.14) |
| **A3** | Quantify membar/SMEM_BAR vs VF on VL-resident SIMD (vs G16 size-1 scratch) |
| **A4** | Classify bound: outer UB-BW vs EX/schedule vs VL-resident / SoftLib VF body |
| **A5** | SoftLib SIMT hand-craft Case-3 G32: measure schedule matrix from absmax 8.22/10.x peers |
| **A6** | Document layout/schedule + X residency for SoftLib / VL-res / SIMT-ASC |
| **A7** | COMPARE vs prior G16 tip + vs absmax SoftLib G32 dual_drain 8.22 (layout inheritance) |

---

## 2. Test-case design

| Axis | Choice |
|------|--------|
| Shape | R=64 · C=128 · **G=32** · SoftLib T∈{128,256,512} · **E=8192** |
| Why G32 | t737u: SIMD-worst vs SIMT-best; t740u: SoftLib 1×G warp-coop matches G32 absmax SoT family |
| Datatype | SoftLib **f32** · VL-res **f32** · SIMT-ASC **fp16** · **`era=case3-quant-bcast`** |
| Gold | e2e `Out = X * (1/max(absmax_group, 1e-6))` · E=8192 · G=32 |
| Reduce axis | **`reduce_axis=inner`** contiguous 1×G (§22) |
| SoftLib axis | **`axis=1xG`** (§21) — **not** wr/Nx1 appendix |
| SoftLib SIMT | pack `/mnt/fluxdata/happybot/sv8_softlib_simt_c3_g32_t740u/` · motif matrix below |
| SIMD VL-res | `sv8d_vl_resident_case3.py` · VL=64 · KEEP X · per-group `vcmax(group=1)` |
| SIMT-ASC | `sv8_case3_bcast.py` · live \| spill_dist · G=32 |
| SoftLib CCE | **N/A** (t646) |

**Sources:** `RESULT.md` · kernels under `benches/sv8/` · SoftLib ASC under pack `sources/` · `config.toml`.

---

## 3. Layout and schedule (SIMT and SIMD)

### 3.1 SoftLib SIMT G32 Case-3 e2e — A5 / A6

- **Motif:** warp-coop **`axis=1xG`** (full-warp G=32) · KEEP-X in RF across abs→redux→recip→mul.
- **Layout inheritance:** absmax SoftLib G32 **wc dual_drain T512 el/cyc=8.22** (MATCHES_1xG) — LDS.32 loadall + absall + dual-drain `asc_reduce_max` + membar stage walls. Adapted to Case-3 by adding per-group eps/recip + `Out=X*scale` STS (store_elems=8192).
- **Not used as gold peer:** G16 wr LDS.32 10.98 = DIFFERENT_GOLD (warp-level max) — appendix only (§21 #24).
- **Best tag:** `wc_c3e2e_back_loadall_absall_dual_drain_g32_t512_e8192` — loadall → absall → dual_drain redux pairs → recip → mul with KEPT `a[]` → STS.
- **Tried matrix (e2e, all gold PASS):** lds32 stock T128/T512; loadall membar; loadall+absall membar; dual_drain; dual_drain fused_mul; T∈{256,512}.

### 3.2 SIMD VL-resident G32 (`sv8d_vl_resident_case3.py`) — A6

- **Motif:** load X VL=64 → i32-VAND abs → for each of NG=2 groups: mask → `vcmax(..., group=1)` → size-1 eps/recip → `vbrc`/`vsel` paint scale → **`vmul` with KEPT X** → store. **No second X load** (`RV_VLDS=0`).
- **Why not compact `vcmax(group=NG=2)`:** PTOAS rejects `vreg<2xf32>`.

### 3.3 SIMT-ASC G32 (`sv8_case3_bcast.py`) — A6

- **live:** Parallel absmax+recip → `sf_inv` fragment; consumer mul reloads **X from `x_ub`** (not RF KEEP-X).

### 3.4 SoftLib CCE

**N/A** this pack (t646).

---

## 4. SIMD PTO-DSL performance (VL-resident)

| tag | status | VF_cyc | el/cyc | instr / schedule | gate |
|-----|--------|-------:|-------:|------------------|------|
| **vl_res G32 live** | **PASS** | **1964** | **4.1711** | KEEP X; VCMAX g=1×NG; VLDS=0 | § gate |

**SEMANTICS_GATE:** PASS as VL-resident Case-3 motif (KEEP X, VLDS=0); INCONCLUSIVE as inner compact VCGMAX/NG=2 SoT — PTOAS blocks `vreg<2xf32>`; used `vcmax(group=1)×NG`; not Gmap VL256 NG=8.

**IPC / figures:** mid IPC W=20 = **1.45** · mean_body = **1.334** · [`plots/simd_vlres_g32_running_ipc.png`](plots/simd_vlres_g32_running_ipc.png) · [`plots/simd_vlres_g32_instr_mix.png`](plots/simd_vlres_g32_instr_mix.png).

---

## 5. SIMT (Ascend default) lowering performance

| tag | status | VF_cyc | el/cyc | notes | gate |
|-----|--------|-------:|-------:|-------|------|
| **live G32** | **PASS** | **56714** | **0.1444** | VF_SIMT; cann_92b2 | § gate |
| spill_dist G32 | **PASS** | **42819** | **0.1913** | spill faster than live this tip | same era |

**SEMANTICS_GATE:** PASS — gold G32 Case-3 e2e measured; SoftLib_axis=1xG; era=case3-quant-bcast; not claiming SoftLib hand-craft SoT from this arm.

**IPC / figures:** mid IPC W=20 = **0.0** (sparse issue vs long span) · mean_body = **0.0685** · [`plots/simt_asc_live_g32_running_ipc.png`](plots/simt_asc_live_g32_running_ipc.png) · [`plots/simt_asc_live_g32_instr_mix.png`](plots/simt_asc_live_g32_instr_mix.png).

---

## 6. SIMT hand-craft (SoftLib) analysis

| Aim | A5 SoftLib SIMT Case-3 e2e G32 |
| Measured | **yes** — 10 e2e + 2 scale-only tags; pack t740u |
| Best e2e | **`wc_c3e2e_back_loadall_absall_dual_drain_g32_t512_e8192`** · el/cyc=**6.7702** · VF=**1210** · gold PASS (maxabs≈1.79e-7) |
| Motif | warp-coop · `axis=1xG` · KEEP-X · dual_drain loadall+absall · T512 |
| Layout peer | absmax SoftLib G32 dual_drain **8.22** (era=fp32-absmax) — schedule inheritance only; **do not merge eras** |
| SoftLib÷SIMD | 6.7702 / 4.1711 ≈ **1.62×** |
| SIMD÷SoftLib | ≈ **0.62×** — H2 ~2× SIMD **does not hold** vs SoftLib |

### SoftLib e2e tried table (all ok=True, E=8192)

| tag | T | VF_cyc | el/cyc |
|-----|--:|-------:|-------:|
| **back_loadall_absall_dual_drain** | **512** | **1210** | **6.7702** |
| back_dual_drain_fused_mul | 512 | 1288 | 6.3602 |
| back_loadall_absall_dual_drain | 256 | 1405 | 5.8306 |
| back_loadall_absall_membar | 512 | 1458 | 5.6187 |
| back_loadall_membar | 512 | 1487 | 5.5091 |
| back_dual_drain_fused_mul | 256 | 1624 | 5.0443 |
| back_loadall_membar | 256 | 2107 | 3.8880 |
| back_loadall_absall_membar | 256 | 2133 | 3.8406 |
| lds32_redux (stock) | 512 | 2565 | 3.1938 |
| lds32_redux (stock) | 128 | 6418 | 1.2764 |

Scale-only appendix (n=E/G=256 — **not** e2e el/cyc): dual_drain T512 **5.975** · T256 **5.245**.

**Instr mix (best e2e, call_count from OPPROF):** SIMT_LDS=256 · STS=256 · REDUX=256 · FDIV_I=256 · FMUL=256 · FADD=256 · IADD_I=512 · FSETP_I=512 · SEL=512.

**SEMANTICS_GATE:** PASS — gold G32 Case-3 e2e (Out=X*scale, n=8192); SoftLib_axis=1xG; motif=warp-coop dual_drain; era=case3-quant-bcast; layout inherited from absmax G32 dual_drain 8.22 (MATCHES_1xG); not wr/Nx1; not claiming absmax-era SoT merge.

**IPC / figures:** mid IPC W=20 = **3.30** · mean_body = **2.689** · [`plots/softlib_simt_c3e2e_dual_drain_g32_running_ipc.png`](plots/softlib_simt_c3e2e_dual_drain_g32_running_ipc.png) · [`plots/softlib_simt_c3e2e_dual_drain_g32_instr_mix.png`](plots/softlib_simt_c3e2e_dual_drain_g32_instr_mix.png) · caption motif=`axis=1xG`.

---

## 7. Running IPC + figures

| Arm | mid IPC W=20 | mean_body | figures |
|-----|-------------:|----------:|---------|
| SoftLib dual_drain G32 | **3.30** | **2.689** | running_ipc + instr_mix (§6) |
| VL-resident G32 | **1.45** | **1.334** | running_ipc + instr_mix (§4) |
| SIMT-ASC live G32 | **0.0** | **0.0685** | running_ipc + instr_mix (§5) |
| Rooftop / el/cyc summary | — | — | [`plots/sv8_g32_elcyc_summary.png`](plots/sv8_g32_elcyc_summary.png) |

Prior G16 tip plots retained under `plots/*_g16_*.png` (COMPARE only).

---

## 8. Conclusion

| Arm | el/cyc | Verdict |
|-----|-------:|---------|
| **SoftLib SIMT G32 dual_drain** | **6.7702** | A5 closed for this Case-3 e2e motif; ~8.22-class layout adapted; gold PASS |
| **SIMD VL-resident G32** | **4.1711** | KEEP X works; not compact-VCGMAX SoT |
| **SIMT-ASC G32 live** | **0.1444** | Stock SimtVF; X from UB; not the H2 peer |

**Ratios (measured only, era=case3-quant-bcast):**
- SoftLib÷SIMD ≈ **1.62×** (SoftLib faster) → H2 **~2× SIMD vs SoftLib falsified**
- SIMD÷SIMT-ASC ≈ **28.9×** (still holds vs stock Ascend SIMT)
- SoftLib÷SIMT-ASC ≈ **46.9×**

**COMPARE (A7):** Absmax SoftLib G32 dual_drain **8.22** (fp32-absmax) → Case-3 e2e SoftLib **6.77** on same schedule family (recip+mul tax). G16 size-1 SIMD was slower than SIMT; G32 VL-res flips vs SIMT-ASC; SoftLib flips again vs VL-res.

**Bound class (A4):** outer UB-BW not binder. SoftLib bound = LDS/REDUX/FDIV/FMUL body (mid IPC≈3.3). VL-res = VSEL/VDIV/VCMAX body (mid IPC≈1.45). SIMT-ASC = long EX/schedule + X reload (mean IPC≈0.07).

**Do not merge** until Lok review.
