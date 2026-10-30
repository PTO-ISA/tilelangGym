# ANALYSIS — SV8 Case-3 (voice SB8) G32 SoftLib ≻ SIMD

**ST id:** `sv8` · voice **SB8** = SV8 Case-3 e2e (no literal `SB8` tree)  
**When:** Asia/Hong_Kong 2026-10-30 · redo under tip `tilelanggym-st-report` + rebuilt harness  
**Authorized:** Lok t542u → REGRESSION-2 handoff → PTO-ISA  
**Pack / tip:** `/workspace/sv8_g32_sb8_t772u/` · dumps `dumps/` · plots `plots/` · tgz `/workspace/sv8_g32_sb8_t772u_pr9_t773u.tgz`  
**Prior segment harvest (supporting, not replaced):** `ANALYSIS_SB8_SEGMENT_IPC_G32.md` · `sb8_g32_segment_harvest.json`  
**Arch KS lock (Lok t540u):** npu_skills **`3101`** = default truth; SoftLib **`3510`** = microbench verify layer  
**Harness cite:** `/workspace/research_topup_2026-10-25/HARNESS_REBUILD.md` + `SKILLS_REBUILD.md` — onesu §6.0* **N/A** for pure VF Case-3 (keep VF el/cyc + IPC primary)  
**PRC index:** `REGRESSION_salvage_2026-09-17/perf-review-knowledge/perf-review-checklist.md` (gate OK 2026-10-30)

**Do not touch:** REGRESSION-2 SV5–7 Nx1 outer (`glm53_sv567_*` / t788u_r2).

---

## 1. Aims

| id | Aim |
|----|-----|
| **A1** | Case-3 quant-bcast e2e (absmax → scale → mul) at **G32**, E=8192, `era=case3-quant-bcast`. |
| **A2** | Headline SoftLib SIMT dual_drain ≻ SIMD best; explain via **per-segment mix + PR9 running-IPC+opcode overlay** (not IPC-only). |
| **A3** | SIMD_best = max(VL-resident KEEP-X, …); fold recip+VMUL (t779) ≡ VL-res; UB-reload (t756) falsified; optional VDIV(x,absmax) contrast. |

Every measured claim below cites `A#`.

---

## 2. Test-case design

| Field | Value |
|-------|-------|
| Shape | R64×C128 · **G=32** · E=**8192** · TILE8192 |
| Era | `era=case3-quant-bcast` (Case-3 recip ≠ absmax-only) |
| Dtype | f32 |
| Host / model | pto-b10 · Ascend950PR_9599 camodel · verify-layer label SoftLib **3510**; primary arch tag **3101** (npu_skills) |
| Pass | functional `np.allclose` vs `Out = X * (1/max(absmax,eps))` · maxabs ~1e-7 class |
| Primary KPI | el/cyc = E / VF_cyc (name VF row: SoftLib `VF_SIMT` · SIMD `VF`) |
| IPC | Running W=20 `(t−W,t]`; body proxy `body_calls/VF_cyc`; segment proxy `calls/sum_cycles` (pipe-overlapped; cyc% need not sum 100%) |

**Conclusion arms (3):**

1. **SIMD PTO-DSL best** — VL-resident KEEP-X (A3)
2. **SIMT Ascend default** — stock simt-asc (~0.14 class)
3. **SIMT SoftLib hand-craft** — dual_drain T512 KEEP-X `axis=1xG` (A2)

Dropped from headline: G16 SoftLib/size-1 pair (superseded), spill_dist (t756 optional unfinished), FA/membar trail tags.

---

## 3. Layout and schedule (SIMT and SIMD)

| Arm | Layout | Reduce | Scale → mul | Sync walls |
|-----|--------|--------|-------------|------------|
| SoftLib SIMT dual_drain | T512 · loadall+absall · **KEEP-X RF** · `axis=1xG` | warp REDUX / SEL / FSETP | sparse **FDIV_I** → **FMUL** | **3×** `asm volatile("" ::: "memory")` compiler clobber (LD · ABS · REDUX/recip→mul) — **not** HW SMEM_BAR (**PRC-SCH-001**) |
| SIMD VL-res | VL=64 · KEEP X in VL · **VLDS=0** | `VCMAX(group=1)×NG` + VMAX/VAND | size-1 **VDIV** + VSEL/VDUP paint + **VMUL** | no asm clobber; `S_sync=0%` |
| SIMD UB-reload (falsified) | size-1 reduce + scratch expand + X reload | — | reload mul | **VLDS=8192** · **SMEM_BAR=256** (**PRC-MEM-001** / **PRC-MEM-002**) |
| SIMT Asc stock | stock Ascend SimtVF | stock stream | — | not SoftLib best |

SoftLib SoT motif: **`axis=1xG` only** (Harness §21) · tag `wc_c3e2e_back_loadall_absall_dual_drain_g32_t512_e8192`.

---

## 4. SIMD PTO-DSL performance (A3)

### 4.1 Measured table (`era=case3-quant-bcast` · G32 · E=8192)

| arm | el/cyc | VF | note |
|-----|-------:|---:|------|
| **SIMD VL-resident KEEP-X** | **4.1711** | **1964** | **SIMD_best** |
| SIMD recip+VMUL (t779) | **4.1711** | **1964** | ≡ VL-res SoT (not a new schedule) |
| SIMD VL VDIV(x,absmax) contrast (t779) | **4.9799** | **1645** | +19.4% vs recip; still SoftLib ahead |
| SIMD UB-reload (t756) | **0.066739** | **122747** | falsified — VLDS=8192 |

Sources: tip harvest · `RESULT_simd_recip_vmul_g32.md` · `/workspace/sv8_ubreload_g32_t756u/RESULT_t756u.md`.

**SIMD_best = max(4.1711, 0.066739, …) = 4.1711 (VL-resident).**  
**PRC-MEM-001:** KEEP-X / VL-resident beats UB-reload on this shape.  
**PRC-SCH-002:** scale forms A (recip+VMUL) ≡ VL-res; B (VL VDIV) faster inside SIMD but SoftLib÷B still **1.360×**.

### 4.2 SEMANTICS_GATE (SIMD VL-res)

```
SEMANTICS_GATE: PASS — G32 gold matches A1; VF body uses VCMAX/VMAX/VAND (inner reduce_axis); era=case3-quant-bcast; el/cyc from VF=1964
```

### 4.3 Body / mid IPC (SIMD_best)

| metric | value |
|--------|------:|
| body_calls | 2581 |
| body IPC≈ | 1.3142 |
| official mid IPC W=20 | **1.45** |
| PR9 mid-med (15–85%) | **1.35** |

PR9: [`plots/simd_vlres_g32_running_ipc_pr9.png`](plots/simd_vlres_g32_running_ipc_pr9.png) · `plot_style=opcode_lanes_ipc` · lanes VLDI/VAND/VMAX/VCMAX/VDIV/VDUP/VSEL/VMUL/VSTI (**PRC-IPC-001**).

---

## 5. SIMT (Ascend default) lowering performance

| arm | el/cyc | VF | note |
|-----|-------:|---:|------|
| SoftLib SIMT-ASC stock | **~0.14 class** | — | stock Ascend SimtVF; **not** SoftLib best (A2) |
| (G16 footnote only) simt-asc | 0.2864 | 28603 | superseded SoftLib headline target — do **not** use as G32 SoftLib≻SIMD pair |

Stock path is LD/ST-bound / low density vs dual_drain hand-craft; kept as conclusion arm 2 only.

---

## 6. SIMT hand-craft (SoftLib) analysis (A2)

| arm | schedule | VF row | VF_cyc | el/cyc | body IPC≈ | mid IPC W=20 | PR9 mid-med |
|-----|----------|--------|-------:|-------:|----------:|-------------:|------------:|
| **SoftLib SIMT dual_drain T512 KEEP-X** | loadall+absall · `axis=1xG` | `VF_SIMT` | **1210** | **6.7702** | 2.6182 | **3.30** | **3.40** |

**Ratio SoftLib÷SIMD_best el/cyc = 1.623×** (VF gap ≈ 754 cyc).

### SEMANTICS_GATE (SoftLib SIMT)

```
SEMANTICS_GATE: PASS — axis=1xG (not Nx1/wr merge); G32 gold; era=case3-quant-bcast; dual_drain motif; el/cyc from VF_SIMT=1210
```

### Driver (re-verified on dumps / harvest)

1. **Issue density** — SoftLib PR9 mid-med **~3.4** vs SIMD **~1.35** (**PRC-IPC-001**).
2. **Cheap S_scale** — SoftLib `FDIV_I:256` vs SIMD VSEL/VDIV/VDUP tax (**PRC-SCH-002**); Δcyc(SIMD−Soft) on S_scale = **+8746**.
3. **RF-resident KEEP-X** — SoftLib few FDIV; SIMD already VLDS=0 so win is **not** “avoid second VLDS” alone (**PRC-MEM-001**).
4. **Compiler clobber ≠ HW bar** — SoftLib 3× `asm ::: memory`; both arms `S_sync=0%` (**PRC-SCH-001**). Cite `asm_clobber_check/REPORT_ASM_MEMORY_CLOBBER.md`.

PR9: [`plots/softlib_simt_c3e2e_dual_drain_g32_running_ipc_pr9.png`](plots/softlib_simt_c3e2e_dual_drain_g32_running_ipc_pr9.png) · lanes LDS→REDUX/SEL/FSETP→FDIV→FMUL→STS.

---

## 7. Running IPC + PR9 overlay figures (FORCED · both arms · A2)

| arm | PR9 `*_running_ipc_pr9.png` | `plot_style` | `meta.lane_opcodes` (non-empty) | staged |
|-----|----------------------------|--------------|----------------------------------|--------|
| SoftLib dual_drain | [`plots/softlib_simt_c3e2e_dual_drain_g32_running_ipc_pr9.png`](plots/softlib_simt_c3e2e_dual_drain_g32_running_ipc_pr9.png) | **opcode_lanes_ipc** | S2R,LDS,FADD,SEL,FSETP_I,REDUX,FDIV_I,STS,END,MOVI,FMUL | [`…_pr9_staged.png`](plots/softlib_simt_c3e2e_dual_drain_g32_running_ipc_pr9_staged.png) |
| SIMD VL-res | [`plots/simd_vlres_g32_running_ipc_pr9.png`](plots/simd_vlres_g32_running_ipc_pr9.png) | **opcode_lanes_ipc** | VLDI,VMUL,VAND,VMAX,VCMAX,VDIV,VDUP,VSTI,VCI,VCMP_GE,VCMP_LT,VSEL | [`…_pr9_staged.png`](plots/simd_vlres_g32_running_ipc_pr9_staged.png) |

**PRC-IPC-001:** both arms PRESENT — reject any IPC-only `*_running_ipc.png` without `_pr9`.

Remake (if needed):
```bash
PYTHONPATH=/workspace/tilelangGym-pr8-tip \
  python /workspace/REGRESSION_salvage_2026-09-17/scripts/make_pr9_running_ipc_overlay.py \
    --dump /workspace/sv8_g32_sb8_t772u/dumps/softlib_dual_drain_g32.instr_log.dump \
    --out-png /workspace/sv8_g32_sb8_t772u/plots/softlib_simt_c3e2e_dual_drain_g32_running_ipc_pr9.png \
    --title "sv8 / SoftLib SIMT Case-3 e2e dual_drain G32 T512 (axis=1xG) — running IPC (W=20)" --E 8192
# same for dumps/vlres_g32.instr_log.dump → simd_vlres_g32_running_ipc_pr9.png
```

Rooftop % vs named resource: **N/A** this turn (VF el/cyc primary; no named A5 roof lock applied to Case-3 e2e beyond measured VF).

---

## 8. Per-segment instr mix + IPC + stage mid-IPC (SIMD and SoftLib · A2)

### 8.1 Per-segment (CSV call aggregates · both arms)

| segment | SIMD opcodes (top) | SIMD IPC | SIMD cyc% | SoftLib opcodes (top) | SoftLib IPC | SoftLib cyc% | winner | note |
|---------|--------------------|---------:|----------:|----------------------|------------:|-------------:|--------|------|
| `S_load` | MOV_ALIGNv2(MTE2)x1, RV_VLDI:128 | 0.0487 | 134.9% | SIMT_LDS:256, MOV_ALIGNv2(MTE2)x1 | 0.0232 | 914.1% | **SIMD** | Δcyc(SIMD−Soft)=-8412 |
| `S_reduce` | RV_VCMAX:256, RV_VMAX:256, RV_VAND:128, … | 0.1004 | 327.7% | SIMT_REDUX:256, IADD/FSETP/SEL/FADD | 0.1159 | 1459.8% | **SIMD** | Δcyc=-11227 |
| `S_scale` | RV_VSEL:1024, RV_VDIV:256, RV_VDUP:256 | 0.1155 | 679.9% | SIMT_FDIV_I:256 | 0.0556 | 380.8% | **SoftLib** | Δcyc=**+8746** · **PRC-SCH-002** |
| `S_eltwise` | RV_VMUL:128 | 0.1250 | 52.1% | SIMT_FMUL:256 | 0.1111 | 190.4% | **SIMD** | Δcyc=-1280 |
| `S_store` | MOV_ALIGNv2(MTE3)x1, RV_VSTI:128 | 0.0510 | 128.9% | SIMT_STS:256, MTE3x1 | 0.0388 | 547.4% | **SIMD** | Δcyc=-4092 |
| `S_sync` | (none) | N/A | 0.0% | (none) | N/A | 0.0% | **N/A** | HW bar 0; SoftLib still has compiler clobber **PRC-SCH-001** |

**Winner metric:** SoftLib if SIMD `sum_cyc` ≫ SoftLib on that segment (SoftLib cheaper cycle tax); reverse → SIMD. SoftLib mid IPC≈3.3 ⇒ high pipe-overlap — SoftLib `sum_cyc` can exceed VF while finishing wall VF sooner. Overall race = **VF / el/cyc**.

**Gap driver #1:** `S_scale` (SoftLib wins on cycle tax) + SoftLib denser issue across overlapped stages (PR9).

JSON: [`sb8_g32_segment_harvest.json`](sb8_g32_segment_harvest.json).

### 8.2 Stage mid-IPC (defining-opcode windows · W=20 mid 15–85%)

Window = first..last issue of defining opcodes (**overlap OK**). SoftLib dual_drain overlaps heavily; SIMD VL-res is a **fused** loop → every stage mid ≈ body mid (~1.35) — written explicitly, not invented walls.

#### SoftLib SIMT (official mid 3.30 / PR9 mid-med 3.40)

| segment | defining opcodes | VF-local ticks | mid_med IPC | mid_mean IPC | n_defining |
|---------|------------------|---------------:|------------:|-------------:|-----------:|
| `S_load` | LDS | 51–695 | **3.25** | 3.0724 | 256 |
| `S_reduce` | REDUX, FADD, SEL, FSETP_I | 190–1037 | **3.5** | 3.4901 | 1536 |
| `S_scale` | FDIV_I | 320–1065 | **3.5** | 3.4719 | 256 |
| `S_eltwise` | FMUL | 369–1137 | **3.45** | 3.3511 | 256 |
| `S_store` | STS | 421–1177 | **3.4** | 3.2209 | 256 |

#### SIMD VL-res (official mid 1.45 / PR9 mid-med 1.35) — fused ⇒ mid≈body

| segment | defining opcodes | VF-local ticks | mid_med IPC | mid_mean IPC | n_defining |
|---------|------------------|---------------:|------------:|-------------:|-----------:|
| `S_load` | VLDI | 13–1857 | **1.35** | 1.363 | 128 |
| `S_reduce` | VCMAX, VMAX, VAND | 16–1889 | **1.35** | 1.3639 | 640 |
| `S_scale` | VSEL, VDIV, VDUP | 18–1918 | **1.35** | 1.3627 | 1536 |
| `S_eltwise` | VMUL | 77–1922 | **1.35** | 1.365 | 128 |
| `S_store` | VSTI | 84–1929 | **1.35** | 1.3651 | 128 |

JSON: [`plots/sb8_g32_stage_mid_ipc.json`](plots/sb8_g32_stage_mid_ipc.json).

---

## 9. Conclusion (3 arms)

| # | Arm | el/cyc | VF | one-liner |
|---|-----|-------:|---:|-----------|
| 1 | **SIMD PTO-DSL best** (VL-res KEEP-X) | **4.1711** | 1964 | SIMD_best; recip+VMUL≡same; UB-reload 0.0667 falsified; VL-VDIV contrast 4.9799 still &lt; SoftLib (**A3**) |
| 2 | **SIMT Ascend default** | **~0.14** | — | stock; not SoftLib best |
| 3 | **SIMT SoftLib hand-craft** (dual_drain T512 `axis=1xG`) | **6.7702** | 1210 | SoftLib÷SIMD **1.623×**; density + cheap S_scale + compiler clobber walls (**A2**) |

**Headline (A2):** SoftLib SIMT ≻ SIMD on G32 Case-3 — explained by PR9 mid ~3.4 vs ~1.35 and S_scale Δcyc, citing **PRC-IPC-001 · PRC-SCH-001 · PRC-SCH-002 · PRC-MEM-001**. No G16 SoftLib headline drift.

### COMPARE harvest

| vs | status |
|----|--------|
| KPI vs REGRESSION-2 tip numbers (handoff table) | **MATCH** SoftLib 6.7702 / SIMD 4.1711 / t779 / t756 |
| Motif-seed Coverage COMPARE | SoftLib dual_drain loadall+absall T512 ✓ · VL-res KEEP-X ✓ · recip≡VL-res ✓ · UB-reload falsified ✓ · stock Asc ~0.14 ✓ · spill_dist stop (not needed for SIMD_best) |
| KPI N/A rows | none for primary pair |

### Checklist (st-report SoftLib≻SIMD bar)

- [x] Aims `A1…` cited
- [x] SEMANTICS_GATE both main arms PASS
- [x] SoftLib SIMT measured (not silent TODO)
- [x] `era=case3-quant-bcast` on rows
- [x] PR9 both arms `*_running_ipc_pr9.png` + `opcode_lanes_ipc` + non-empty lanes
- [x] Per-segment mix+IPC table both arms
- [x] Stage mid-IPC table both arms (SIMD fused → mid≈body explicit)
- [x] Ready PRC-ids cited
- [ ] `EXPLORATION_TRACE.md` — **MISSING** (see §10)

---

## 10. Trace pointer

**MISSING:** `/workspace/sv8_g32_sb8_t772u/EXPLORATION_TRACE.md`  
**Why:** this turn is an **ANALYSIS redo** from REGRESSION-2 measured tip (t740u/t772u/t773u/t756u/t779u) under tip st-report spine — **not** a fresh `tilelanggym-st-explore-from-prompt` / clean-regress loop. MB DONE for SoftLib schedule ladder was already claimed upstream; this deliverable does not re-open explore.  
**Next-cmd (optional):** if Lok wants TRACE for peanut PR refresh, run explore-from-prompt companion and write `EXPLORATION_TRACE.md` linking the existing dumps — do **not** invent a synthetic TRACE.

---

## 11. G16 footnote (superseded SoftLib target)

| arm | el/cyc | VF | note |
|-----|-------:|---:|------|
| G16 simt-asc | 0.2864 | 28603 | stock — **not** SoftLib |
| G16 size-1 SIMD | 0.06361 | 128790 | not VL-res SoT |
| **G32 SoftLib dual_drain (this ANALYSIS)** | **6.7702** | **1210** | **primary SoftLib≻SIMD pair** |
| **G32 VL-resident SIMD** | **4.1711** | **1964** | SIMD_best |

---

## 12. MISSING / next

| item | status | next-cmd |
|------|--------|----------|
| PR9 both arms | **PRESENT** | — |
| Per-segment + stage mid-IPC | **PRESENT** | — |
| EXPLORATION_TRACE.md | **MISSING** | optional explore TRACE if peanut PR asks |
| Peanut / tilelangGym PR #11 refresh | **not pushed** | needs Lok yes — see `STATUS.md` |
| SIMD vdiv_x PR9 overlay | optional (not SIMD_best) | `make_pr9… --dump simd_recip_vmul_t779u/vdiv_x_g32.instr_log.dump` |
| spill_dist (t756) | unfinished · not needed for SIMD_best | leave |
