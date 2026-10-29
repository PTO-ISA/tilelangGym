# SoftLib SIMT axis=1×G — Harness §20/§21 + skill gates (t705u)

**Date:** 2026-10-27 Asia/Hong_Kong · **era=fp32-absmax** · Case-1 · E=8192 · **no recip**  
**Skills:** `tilelanggym-softlib-simt-schedule` · `tilelanggym-st-semantics-gate` · Harness **§20** schedule matrix · **§21** `axis=1×G` only for main SoftLib SIMT SoT  
**Packs:** `…_multwarp_t686u` · `…_1xg_t697u` · `…_1xg_t698u` · `…_g32_back_t689u` · t685u T128 stock close

**Main SoftLib SIMT SoT (axis=1×G only — NOT wr Nx1):**

| ST | G | stock T128 (wc) | best T | best motif | el/cyc | end |
|----|--:|----------------:|-------:|------------|-------:|-----|
| SV5 | 16 | 1.13 | **T512** | wc halfmask **loadall+absall drain** LDS.32 | **4.74** | **best-sweep DONE** |
| SV6 | 32 | 1.57 | **T512** | wc **loadall+absall dual-drain** LDS.32 | **8.22** | **best-sweep DONE** |
| SV7 | 64 | 2.39 | **T512** | wc-multi **per=2** LDS.32 | **5.54** | **best-sweep DONE** |
| SV7 | 128 | 1.67 (T64 stock) | **T256** | wc-multi **per=4** LDS.32 | **3.72** | **best-sweep DONE** (T512 never measured — residual) |

**Nx1 appendix only (NOT main SoT):** G16 wr LDS.32 membar+sparse2 T512 **10.98** (`axis=Nx1`).

## SEMANTICS_GATE (per G, SoftLib SIMT 1×G)

| G | S1 gold | S2 layout | S3 opcode/motif | S4 VF | verdict |
|--:|---------|-----------|-----------------|-------|---------|
| 16 | Case-1 E/G=512 outs · halfmask 16-of-32 | contig LDS.32 · `axis=1xG` | wc halfmask + REDUX | VF named (1727 @ SoT) | **PASS** — no Nx1 pollution |
| 32 | Case-1 E/G=256 | contig LDS.32 · full-warp | wc dual-drain | VF 996–997 | **PASS** |
| 64 | Case-1 E/G=128 | contig · per=2 fold in-warp | wc-multi | VF 1478 @ T512 | **PASS** |
| 128 | Case-1 E/G=64 | contig · per=4 | wc-multi | VF 2201 @ T256 | **PASS** |

## Per-G §20 matrix — tried vs residual

### G16 (SV5) — COMPLETE for 1×G SoT claim
| Axis | Tried | Outcome |
|------|-------|---------|
| Motifs | wc halfmask, wg within-group, wr (Nx1 only) | **SoT = wc halfmask**; wg ≤1.92; wr → appendix |
| LDS.32 / LDS.64 | LDS.32 SoT; LDS.64 on wg + historical wr | LDS.32 wins 1×G class |
| bar / nobar | fused vs membarST vs membar+loadall | fused≈membarST @T128; **back+membar/loadall** wins @T512 |
| misched | on (default SoT packs) | off not re-swept at T512 SoT — **residual: misched=0 @ SoT T** |
| unroll/pack | loadall, absall, dual_drain, sparse2 (wr only) | **loadall+absall drain T512 = 4.74** SoT; t698 loadall+membar 4.63 peer |
| T | 128 / 256 / 512 | T512 wins |
| Why stock T128 loses | VF 7264 → el/cyc 1.13; no multi-warp latency hide | occupancy / issue depth |
| End | **best-sweep DONE** | residual: misched=0 @T512; optional LDS.64 wc halfmask T512 |

### G32 (SV6) — COMPLETE
| Axis | Tried | Outcome |
|------|-------|---------|
| Motifs | wc, wg, (wr N/A mid-G) | **SoT = wc dual-drain T512 8.22** (t689u); serial loadall+absall 7.48 loses |
| LDS | LDS.32 SoT; LDS.64 wg | wg ~0.98 |
| bar | fused≈membarST @T128; dual-drain staged | dual-drain wins |
| misched | on (SoT) | off residual @ SoT T |
| T | 128 / 256 / 512 | T512 wins (1.57 → 5.72 bare → **8.22** dual) |
| Why stock T128 loses | 1.57 vs 8.22 — multi-warp + back-pressure drain | |
| End | **best-sweep DONE** | |

### G64 (SV7) — COMPLETE (with named residual)
| Axis | Tried | Outcome |
|------|-------|---------|
| Motifs | wc-multi per=2, wg | **SoT wc-multi T512 5.54**; wg ~0.98 |
| WC-only (per=1) | **structurally no** for G>32 (need ≥2 elems/lane inside gold G) | stop: ISA/layout — see t704u |
| LDS | LDS.32 SoT; LDS.64 wg | |
| T | 128 / 256 / 512 | **5.54 @ T512**; T128=2.39 |
| Why stock T128 loses | latency hide / occupancy (2.39 → 5.54) | |
| End | **best-sweep DONE** | residual: T>512 not run; misched=0 @T512 |

### G128 (SV7) — COMPLETE (with named residual)
| Axis | Tried | Outcome |
|------|-------|---------|
| Motifs | wc-multi per=4, wg | **SoT wc-multi T256 3.72**; wg ~0.95 |
| WC-only | **no** (per=4 required) | |
| T | 64 / 128 / 256; **T512 never measured** | T256 best among measured |
| Why stock T64/T128 loses | 1.67 / 3.36 → **3.72** @ T256 | |
| End | **best-sweep DONE** among measured T | **residual: T512 unmeasured** (honest stop — not UNMEASURED-only success; T256 is max among tried) |

## COMPLETE vs residual summary
| Claim | Status |
|-------|--------|
| SoftLib SIMT main SoT is **axis=1×G** {4.74, 8.22, 5.54, 3.72} | **COMPLETE** |
| No wr 10.98 in main SoT | **COMPLETE** (appendix) |
| §20 matrix measured enough to claim best T/motif per G | **COMPLETE** with residuals: misched=0 @SoT T; G128 T512; optional LDS.64 wc @G16 T512 |
| SEMANTICS_GATE PASS · no Nx1 pollution | **PASS** |
