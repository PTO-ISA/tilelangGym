# SV5 analysis — small-block reduce (SIMD VCG vs SoftLib SIMT 1×G)

**Date:** 2026-10-22 Asia/Hong_Kong · ST-spine · **1×G SoftLib SIMT + VMI Gmap close 2026-10-27 (t705u)** · **t706u:** SoftLib CCE off main table · **SOC:** Ascend950PR_9599 opsim · **CANN / PTOAS:** cann_92b2 / pins in `config.toml`  
**Format:** [`docs/ST_REPORT_FORMAT.md`](../../docs/ST_REPORT_FORMAT.md) + `tilelanggym-st-report` · companions: `tilelanggym-softlib-simt-schedule` · `tilelanggym-st-semantics-gate` · `tilelanggym-vmi-gmap-close`  
**Primary metric:** **el/cyc = E / VF_cyc** · **E = 8192** fp32 · **era=fp32-absmax** (Case-1 · no recip).  
**Never invent measured numbers.** SoftLib scalar-wait/BUSY = **BACKLOG**.

**Conclusion arms (headline — SoftLib CCE omitted when VMI ≈ SoftLib):**

| # | Path | Best on this tip | Era |
|---|------|------------------|-----|
| 1 | **SIMD PTO-DSL / VMI** | VL=128 G=8 VAND **33.44** | 2026-10-22 · fp32-absmax |
| 2 | **SIMT-ASC stock** | **1.13** wc halfmask LDS.32 T128 (VF 7264) · default SoftLib `__simt_vf__` lowering | fp32-absmax |
| 3 | **SIMT SoftLib hand-craft** | **axis=1×G** wc halfmask loadall+absall drain T512 **4.74** | 2026-10-27 · fp32-absmax |

> Footnote (not a required arm): SoftLib CCE dual64 VAND absmax **31.39** — VMI already best (−6% vs 33.44); off main table per t706u.

**Not main SoftLib SIMT SoT:** historical wr LDS.64 fused **4.51** / wr membar+sparse2 **8.33** / wr T512 **10.98** — those are **`axis=Nx1`** (appendix). See [`reports/SOFTLIB_SIMT_1xG_HARNESS_GATE_t705u.md`](reports/SOFTLIB_SIMT_1xG_HARNESS_GATE_t705u.md).

---

## 1. Aims

| Id | Aim |
|----|-----|
| **A1** | Case-1 absmax fp32 · gold G=16 · E=8192 · compare SIMD block-reduce vs SoftLib SIMT |
| **A2** | VMI / PTO-DSL per-G map closed or explicit stop (`tilelanggym-vmi-gmap-close`) |
| **A3** | SoftLib SIMT main SoT = **`axis=1×G` only** (Harness §21); §20 schedule matrix documented COMPLETE vs residual |
| **A4** | SoftLib CCE **off main table** when VMI ≥ SoftLib (t646/t706u); optional footnote only |
| **A5** | SoftLib SIMT measured (not silent TODO) |

---

## 2. Test-case design

| Axis | Choice |
|------|--------|
| Shape | E=8192 · TILE8192 · gold G=16 · gate R64×C128 |
| Datatype | pure fp32 UB · **era=fp32-absmax** · no Cast / recip |
| SIMD PTO-DSL | `sv5d_absmax_vl128_g8.py` · VL=128 · VCGMAX G=8 · i32 VAND clear-sign |
| SoftLib SIMT | `__simt_vf__` · **wc halfmask** · `axis=1xG` · LDS.32 · T∈{128,256,512} |
| SoftLib CCE (footnote) | dual64 `__simd_vf__` VAND absmax — not a conclusion arm |
| Pass | gold PASS · VF named · SEMANTICS_GATE |

**Case matrix (spine — SoftLib CCE not required):**

| # | schedule | path | el/cyc | VF | era |
|---|----------|------|-------:|---:|-----|
| 1 | VMI VL128/NG8 VAND | PTO-DSL | **33.44** | 245 | fp32-absmax |
| 2 | **SIMT-ASC stock** (default SoftLib `__simt_vf__` @ legacy T) | SoftLib stock | **1.13** | 7264 | fp32-absmax |
| 3 | SoftLib SIMT wc halfmask loadall+absall drain T512 | SoftLib `__simt_vf__` | **4.74** | 1727 | fp32-absmax |
| 3s | SoftLib SIMT wc halfmask stock T128 | SoftLib | 1.13 | 7264 | fp32-absmax |

---

## 3. Layout and schedule

### 3.1 SIMD PTO-DSL
VL=128 · VCGMAX G=8 · abs = i32 VAND · store reduced Y · EXU/SLIDE-bound.

### 3.2 SoftLib CCE (footnote peer only)
dual64 PHYS_VL64×2 · VAND+VCGMAX · **31.39** · same class as VMI 33.44 — **not shown in headline** (VMI best).

### 3.3 SIMT-ASC stock
**SIMT-ASC stock** = SoftLib `__simt_vf__` **default lowering** at legacy stock T (same as SoftLib SIMT stock column) — **not** a separate unmeasured Ascend-only arm.

el/cyc **1.13** · wc halfmask LDS.32 **T128** · VF 7264 · tag `wc_abs_lds32_redux_g16_t128_e8192`.

### 3.4 SIMT SoftLib — **axis=1×G** (Harness §20/§21)
- Motif: **warp-coop halfmask** (16-of-32) · contig LDS.32 · REDUX · **not** wr Nx1.
- Best: **loadall+absall dual/serial drain T512 = 4.74** (t697u pack).
- Stock T128 = 1.13 loses on multi-warp latency hide / occupancy.
- Full §20 tried/residual: [`reports/SOFTLIB_SIMT_1xG_HARNESS_GATE_t705u.md`](reports/SOFTLIB_SIMT_1xG_HARNESS_GATE_t705u.md).

---

## 4. SIMD PTO-DSL performance

| tag | VF | el/cyc | instr (class) | gate |
|-----|---:|-------:|---------------|------|
| VL128/NG8 VAND | 245 | **33.44** | VCGMAX + VAND + VLD/VST | PASS |

VMI Gmap G16: **best-sweep DONE** · SoftLib CCE footnote only (VMI best). See [`reports/VMI_GMAP_ABSMAX_CLOSE_t705u.md`](reports/VMI_GMAP_ABSMAX_CLOSE_t705u.md).

Rolling IPC / opcode-lane plots for SoftLib VAND path: [`SIMD_RUNNING_IPC.md`](SIMD_RUNNING_IPC.md) / `ipc/`. VMI dig IPC: cite dig pack when present; live G16 not re-harvested this turn.

---

## 5. SIMT-ASC stock

**SIMT-ASC stock** = SoftLib `__simt_vf__` default lowering @ legacy T (= SoftLib SIMT stock).

el/cyc **1.13** · wc halfmask LDS.32 T128 · VF 7264.

---

## 6. SIMT SoftLib hand-craft (A5)

### 6.1 SoT row (main)
| motif | axis | LDS | T | pack | el/cyc | end |
|-------|------|-----|--:|------|-------:|-----|
| wc halfmask loadall+absall drain | **1×G** | 32 | **512** | loadall+absall | **4.74** | **DONE** |

### 6.2 Why stock T128 loses
T128 halfmask VF 7264 → **1.13** el/cyc vs T512 **4.74** — higher T hides LD/REDUX latency; §20 T-sweep (128/256/512) measured.

### 6.3 Motifs tried (not SoT)
| tag class | el/cyc | note |
|-----------|-------:|------|
| wg LDS.64 loadall membar T128–512 | ≤1.92 | within-group · weak T-scale |
| wr LDS.32 membar+sparse2 T512 | **10.98** | **`axis=Nx1` appendix only** |
| historical wr LDS.64 fused T128 | 4.51 | Nx1 / polluted prior tip |

### 6.4 SEMANTICS_GATE
**PASS** — S1 Case-1 E/G=512 · S2 halfmask contig LDS.32 `axis=1xG` · S3 REDUX motif · S4 VF named · **no Nx1 pollution in main SoT**.

### 6.5 Harness checklist (t705u)
| Check | Result |
|-------|--------|
| §20 LDS.32/64 | COMPLETE (LDS.32 SoT; LDS.64 on wg) |
| §20 bar/nobar | COMPLETE (fused≈membarST @T128; back+membar @T512) |
| §20 misched | residual: misched=0 @ SoT T not re-swept |
| §20 unroll/pack/loadall/drain | COMPLETE |
| §21 axis=1×G only in main SoT | **COMPLETE** |
| softlib-simt-schedule Done-when | **COMPLETE** (+ named residuals) |
| st-semantics-gate | **PASS** |

---

## 7. Conclusion

| Arm | Best el/cyc | Ratio vs VMI |
|-----|------------:|-------------:|
| VMI PTO-DSL | **33.44** | 1.00× |
| SoftLib SIMT 1×G best | **4.74** | 0.14× |
| **SIMT-ASC stock** | **1.13** | 0.034× |

> SoftLib CCE dual64 **31.39** (0.94× VMI) — footnote only; not a required conclusion arm (t706u).

**COMPARE harvest:** SoftLib SIMT tip moved **4.51 wr → 4.74 wc 1×G** (t703u/t705u axis fix). VMI G16 unchanged 33.44. Gate docs + RESULT in `reports/`.

**PR:** https://github.com/PTO-ISA/tilelangGym/pull/8
