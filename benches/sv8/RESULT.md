# SV8 G32 Case-3 e2e RESULT (t740u SoftLib + t737u VL-res)

**Host:** pto-b10 / Ascend950PR_9599 camodel  
**When:** 2026-10-29T14:54:42+08:00 Asia/Hong_Kong (SoftLib matrix DONE)  
**Packs:** SoftLib `/mnt/fluxdata/happybot/sv8_softlib_simt_c3_g32_t740u/` · prior VL-res/simt `/mnt/fluxdata/happybot/sv8_g16_e2e_t726u/`  
**Labels:** `reduce_axis=inner` SoftLib_axis=`1xG` `era=case3-quant-bcast`  
**Shape:** R=64 C=128 **G=32** E=8192  

**Env:**
- **SoftLib SIMT:** bisheng dav-3510 `__simt_vf__` + **cann_92b2**
- **simt-asc:** TILELANG=`tilelang-pto-isa-st` + **cann_92b2**
- **sv8d VL-resident:** PTO-DSL handwritten + **cann_91b3** opsim

**Dtype footnote:** SoftLib **f32** · VL-res **f32** · SIMT-ASC **fp16**. SoftLib÷SIMD is same-dtype f32.

---

## 1. Analysis (hypothesis)

### H2 — SIMD ~2× SIMT with X VL/RF-resident
**Re-tested vs SoftLib SIMT** at absmax-class layout (dual_drain / loadall / T512), not vs SIMT-ASC 0.14.

| arm | X residency | VF_cyc | el/cyc | notes |
|-----|-------------|-------:|-------:|-------|
| SoftLib dual_drain T512 | **KEEP-X RF** (loadall→mul) | **1210** | **6.7702** | axis=1xG; gold PASS |
| sv8d VL-resident G32 | **KEEP X VL** | **1964** | **4.1711** | vcmax g=1×NG |
| simt-asc live G32 | X reloads from `x_ub` | **56714** | **0.1444** | stock |

**SoftLib÷SIMD** = 6.7702 / 4.1711 ≈ **1.62×** (SoftLib faster).  
**SIMD÷SoftLib** ≈ **0.62×**. H2 ~2× **SIMD** vs SoftLib **falsified** (inverted).  
**SIMD÷SIMT-ASC** ≈ **28.9×** still holds vs stock Ascend.

---

## 2. SoftLib SIMT matrix (t740u) — measured

Best e2e: **`wc_c3e2e_back_loadall_absall_dual_drain_g32_t512_e8192`** el/cyc=**6.7702** VF=**1210** ok=True.

Full e2e table (sorted by el/cyc desc): see ANALYSIS §6. Scale-only appendix dual_drain T512 el/cyc=5.975 (n=256 — different basis).

Layout inheritance: absmax SoftLib G32 dual_drain **8.22** (MATCHES_1xG). G16 wr 10.98 DIFFERENT_GOLD — not used.

---

## 3. SEMANTICS_GATE

- SoftLib SIMT best: `SEMANTICS_GATE: PASS — gold G32 Case-3 e2e (Out=X*scale, n=8192); SoftLib_axis=1xG; motif=warp-coop dual_drain; era=case3-quant-bcast; layout inherited from absmax G32 dual_drain 8.22 (MATCHES_1xG); not wr/Nx1`
- sv8d VL-resident G32: `SEMANTICS_GATE: PASS as VL-resident Case-3 motif (KEEP X, VLDS=0); INCONCLUSIVE as inner compact VCGMAX/NG=2 SoT — PTOAS blocks vreg<2xf32>; used vcmax(group=1)×NG; not Gmap VL256 NG=8`
- simt-asc G32 live: `SEMANTICS_GATE: PASS — gold G32 Case-3 e2e measured; SoftLib_axis=1xG; era=case3-quant-bcast; not claiming SoftLib hand-craft SoT`

---

## 4. IPC (W=20) — from core0.veccore0.instr_log.dump

| arm | mid_ipc_W20 | mean_ipc_body | issues | span |
|-----|------------:|--------------:|-------:|-----:|
| SoftLib dual_drain G32 | 3.30 | 2.6893 | 3168 | 1178 |
| VL-res G32 | 1.45 | 1.334 | 2568 | 1925 |
| SIMT-ASC live G32 | 0.0 | 0.0685 | 3885 | 56682 |

Figures: `benches/sv8/plots/softlib_simt_c3e2e_dual_drain_g32_{running_ipc,instr_mix}.png`, `simd_vlres_g32_*`, `simt_asc_live_g32_*`, `sv8_g32_elcyc_summary.png`.

---

## 5. COMPARE — prior G16 + absmax SoftLib peer

| arm | G | era | el/cyc | motif |
|-----|--:|-----|-------:|-------|
| SoftLib Case-3 e2e dual_drain | 32 | case3-quant-bcast | **6.7702** | axis=1xG KEEP-X |
| SoftLib absmax dual_drain (peer) | 32 | fp32-absmax | **8.22** | axis=1xG (layout peer only) |
| VL-res SIMD | 32 | case3-quant-bcast | 4.1711 | KEEP X |
| simt-asc live | 32 | case3-quant-bcast | 0.1444 | stock |
| simt-asc live | 16 | case3-quant-bcast | 0.2864 | COMPARE |
| sv8d size-1 | 16 | case3-quant-bcast | 0.06361 | COMPARE |

Do **not** merge SoftLib 6.77 with absmax 8.22 as one era ladder (§13).
