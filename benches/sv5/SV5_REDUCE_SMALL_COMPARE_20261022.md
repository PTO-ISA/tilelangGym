# SV5 reduce-small-group — SIMD compute-bound + Simt framing (pure fp32)

**Date:** 2026-10-22 (Asia/Hong_Kong)  
**SOC / CANN:** Ascend950PR_9599 opsim · cann_92b2  
**Host:** pto-b10  
**Scope:** **pure fp32 input, no cast.** Cast / fp16→fp32 paths remain parked.

**Primary metric:** **el/cyc = E / VF_cyc** with **E = 8192**.  
Secondary (Simt): **B/cyc** on first→last `SIMT_LDS` wall; LDS-window el/cyc ≈ B/cyc ÷ 4.

**One-line story:** SoftLib SIMD dual64 is **compute / latency+issue bound** — **reduce-max ~35 el/cyc**, **absmax VAND ~31** (legacy VMUL+VMAX abs ~20). Stock Simt small-group map is **bank-bound ~1.65 el/cyc (&lt;2)**; remapping to **per-warp contig** recovers **~4–10 el/cyc** on the LDS window (still far below the SV1 A-only LDS.32 roof ~100 B/cyc ≈ **25 el/cyc load-only**).

---

## 0. Lead table (el/cyc first)

| schedule | path | map / mix | **el/cyc** | how read | PASS |
|----------|------|-----------|----------:|----------|:----:|
| **SIMD reduce-max** SoftLib dual64 | SoftLib `__simd_vf__` | VLD+VCGMAX+VST ×128 | **35.01** | VF E/cyc | ✅ |
| **SIMD absmax VAND** SoftLib dual64 | SoftLib `__simd_vf__` | VLD+VAND+VCGMAX+VST ×128 | **31.39** | VF E/cyc | ✅ |
| SIMD absmax VMUL+VMAX *(legacy)* | SoftLib `__simd_vf__` | VLD+VMUL+VMAX+VCGMAX+VST ×128 | 19.65 | VF E/cyc | ✅ |
| SIMD FA serial VL64×2 | PTO-DSL FA | contig | 8.98 | VF E/cyc | ✅ |
| **Simt stock stream** (bank-conflict) | SoftLib `__simt_vf__` ub_stream | **stride64** G=16 | **1.65** | VF E/cyc | ✅ |
| SoftLib contig-warp + redux LDS.64 | SoftLib SIMT | contig float2 + `asc_reduce_max` | **4.51** VF · **~10.2** LDS-win (40.8/4) | both | ✅ |
| SoftLib contig-warp LDS.32 interleaved | SoftLib SIMT | contig + fold-as-load | **3.99** VF · **~8.8** LDS-win (35.1/4) | both | ✅ |
| SoftLib contig-warp LDS.32 load-all | SoftLib SIMT | contig float + redux | **3.46** VF · **~6.7** LDS-win (26.6/4) | both | ✅ |

**UB / load roofs (for contrast — not what SIMD hits):**

| roof | value | use |
|------|------:|-----|
| UB peak | **128 B/cyc** | absolute LDS bandwidth |
| SV1 SoftLib A-only LDS.32 (T128) | **~101 B/cyc (~79% of 128)** → **~25 el/cyc load-only** | contig load-only reference |
| SIMD SLIDE (VCGMAX) resource | **~64 el/cyc** | 1 VCGMAX/cyc × 64 el dual64 row / 2 cyc |
| SIMD EXU absmax VAND | **~42.7 el/cyc** | `ceil(2/2)+2 = 3` cyc/row |
| SIMD EXU absmax VMUL+VMAX | **32 el/cyc** | `ceil(4/2)+2 = 4` cyc/row |

---

## 1. Aims

Same **E=8192 fp32** tile, three schedules:

1. SoftLib **SIMD** dual64 interleave + VCGMAX G=8 — **primary** (reduce-max / VAND absmax).
2. Simt **stock** small-group map (stride64 bank cliff).
3. SoftLib **SIMT contig + warp redux_max** (continuous load, LDS.32 / LDS.64).

All arms: **fp32 in UB, no `Cast` / no `SIMT_F2F`.**

---

## 2. SIMD — compute-bound / latency + issue (primary)

### 2.1 Instruction mix (Lok framing)

Lok’s sketch: **LD (interleave) + abs/reduce + VCGMAX + ST (~32B reduced only)**.  
**Primary abs is `VAND(~0x80000000)`, not VMAX** — VMAX was the old abs path (`VMUL(−1)+VMAX`).

**Per dual64 row (128 el = PHYS_VL 64 × 2 trips):**

```
2×VLD  +  2×(VAND | ∅)  +  2×VCGMAX(G=8)  +  2×VST(tiny)
```

**Per trip ≈** `1×VLD + 1×(VAND|none) + 1×VCGMAX + 1×VST`.

| arm | whole-kernel mix (counts) | ST size |
|-----|---------------------------|---------|
| **reduce-max** | VLD×128, VCGMAX×128, VST×128 | reduced only (~8–16 el / ~32–64 B) — **not** full-tile write |
| **absmax VAND** | VLD×128, **VAND×128**, VCGMAX×128, VST×128 | same tiny ST |
| absmax VMUL+VMAX *(legacy)* | VLD×128, VMUL×128, VMAX×128, VCGMAX×128, VST×128 | same |

VCGMAX G=8 → 8 outs per trip; device scalar pair-max stitches G8→G16 gold. ST is **not** the mem bottleneck.

### 2.2 Why not mem-bound

| resource | rate | cycles / dual64 row (128 el) | if sole → el/cyc |
|----------|------|-----------------------------:|-----------------:|
| VLD | dual-issue **2/cyc** (2×256 B) | **1** | 128 |
| **VCGMAX** | **1/cyc EXQ0 (SLIDE)** | **2** | **64** |
| VAND (ADD, dual ✓) | can share EXQ with non-SLIDE | adds to EXU (below) | — |
| VST | 1/cyc, but only ~8–16 el | **2** (issue) | **not BW-bound** |

Alternating 2R / 1R+1W (~192 B) framing matters for **full-tile STS** (e.g. SV1 ADD). Here the write is tiny → **tight roof is SLIDE / EXU issue + dep latency**, not UB peak.

A5 EXU (from `npu_skills` prediction / a5-vector-pipeline):

```
EXU_cyc/row = ceil(n_non_SLIDE / 2) + n_SLIDE
```

SLIDE (`VCGMAX`) is **EXQ0-only**. Non-SLIDE (VAND / VMUL / VMAX) dual across EXQ0/EXQ1 → ~half also steal Q0.

| arm | n_non | n_SLIDE | EXU cyc/row | **resource el/cyc** | measured whole | mid input el/cyc |
|-----|------:|--------:|------------:|--------------------:|---------------:|-----------------:|
| reduce-max | 0 | 2 | 2 | **64** | **35.01** | **51.2** (~80% of 64) |
| **absmax VAND** | **2** | 2 | **3** | **~42.7** | **31.39** | **41.6** (~97% of 42.7) |
| absmax VMUL+VMAX | 4 | 2 | 4 | **32** | 19.65 | 25.6 (~80% of 32) |

### 2.3 Measured SoftLib dual64 (E=8192)

| arm | VF | **el/cyc** | mid IPC (W=20) | mid input el/cyc | mean body IPC | mean el_L (VLD-win) |
|-----|---:|----------:|---------------:|-----------------:|--------------:|--------------------:|
| **reduce-max** | 234 | **35.01** | **2.40** | **51.2** | 1.96 | 55.6 |
| **absmax VAND** | 261 | **31.39** | **2.50** | **41.6** | 2.31 | 43.3 |
| absmax VMUL+VMAX | 417 | 19.65 | 1.95 | 25.6 | 1.70 | 24.8 |

- **Whole-kernel 35 / 31** include kickstart + tail; **mid already ~50–55 / ~42** on the resource roofs.
- reduce-max mid **51.2 / 64 ≈ 80%** of SLIDE roof — head/tail pull average to **35** (Lok: “35 is ok”).
- VAND mid **41.6 / 42.7 ≈ 97%** of EXU roof — latency win vs VMUL+VMAX (VAND lat ~5, single non-SLIDE vs two).
- ~½ of VANDs land on EXQ0 → EXQ0/row ≈ 2 VCGMAX + 1 VAND = **3** vs reduce-max’s **2** → **+50% Q0 issue**, the **64 → ~43** step (not “20% of VANDs”).

**Dep chains:**

- reduce-max: `VLD → VCGMAX → VST`
- VAND absmax: `VLD → VAND → VCGMAX → VST` (shorter than VMUL→VMAX→VCGMAX)
- legacy: `VLD → VMUL → VMAX → VCGMAX → VST` (VCGMAX issue drops ~1.0 → 0.4 mid)

### 2.4 SIMD artifacts

| what | path |
|------|------|
| reduce-max vs absmax (VMUL) run | `/workspace/sv5_simd_baseline/out/sv5_reducemax_vs_absmax_20261022/` |
| **VAND absmax** run | `/workspace/sv5_simd_baseline/out/sv5_vand_absmax_20261022/` |
| IPC plots + mid notes | `/workspace/sv567_analysis_draft/sv5_ipc_20261022/` |
| reduce-max rolling20 PNG | `…/sv5_simd_dual64_reducemax_rolling20_ipc.png` |
| VAND absmax rolling20 PNG | `…/sv5_simd_dual64_absmax_vand_rolling20_ipc.png` |
| Q0 compare PNG | `…/sv5_reducemax_vs_absmax_ipc_compare.png` |
| Rooftop (npu_skills) | `/workspace/sv567_analysis_draft/npu_skills_rooftop_20261022/` |

---

## 3. Simt — stride bank cliff &lt;2 el/cyc; switch to per-warp contig

### 3.1 Stock stream — bank-bound ~1.65 el/cyc

Small-group map `A0[i·(T·G) + tid·G + t]` with **G=16 fp32** → concurrent lanes at fixed `t` hit addresses **64 B apart** → 32 banks × 8 B → **4 unique banks / 8-way conflict**.

| tag | VF_cyc | **el/cyc** | B/cyc (LDS wall) | STK | LDK | F2F |
|-----|-------:|----------:|-----------------:|----:|----:|----:|
| `stock_stream_stride64_fp32_e8192_t128` | 4975 | **1.647** | **7.81** | **0** | **0** | **0** |

- **&lt;2 el/cyc** — matches A-only stride64 ubench (~6.8–8.0 B/cyc ≈ 1.7–2.0 el/cyc of load alone).
- Motif is **UB mem / bank-bound**, not ALU. Spill not involved (STK=LDK=0).
- **Need per-warp contiguous remap** to escape the cliff.

### 3.2 SV1 reference — UB load roof ~100 B/cyc ≈ 25 el/cyc

From SoftLib **A-only LDS.32** (SV1 ADD path, contig):

| ref | B/cyc | el/cyc (load-only) |
|-----|------:|-------------------:|
| T128 SoftLib A-only LDS.32 | **~101** (~79% of 128) | **~25** (101/4) |
| T256 MI-on A-only (ADD whole-VF SoT) | ~95 LDS-win | whole-VF **10.79** (serial 1R+1W roof **16**) |

That **~25 el/cyc** is the **UB load-only roof reference** for contig LDS.32 — **not** what Simt reduce hits (redux + abs/SEL stretch the LDS window).

Cite: `/workspace/sv567_analysis_draft/sv1_add_lds32_ipc_20261022/NOTES.md`

### 3.3 Contig-warp redux — measured now

Contig LDS → per-lane absmax → `asc_reduce_max` (`SIMT_REDUX`×4 warps) → STS broadcast. Pure fp32.

| arm | VF el/cyc | **B/cyc** | LDS-win el/cyc | % of 128 | % of width |
|-----|----------:|----------:|---------------:|---------:|-----------:|
| **LDS.64** load-all | **4.51** | **40.76** | **~10.2** | 31.8% | **63.7% of 64** |
| LDS.32 interleaved | **3.99** | **35.05** | **~8.8** | 27.4% | — |
| LDS.32 load-all | **3.46** | **26.62** | **~6.7** | 20.8% | **83% of 32** |

**vs stock 1.65:** contig-warp is **~2.5–3×** on whole-VF el/cyc and **~4–5×** on LDS-window bytes.  
**vs SV1 ~101 B / ~25 el load-only:** redux body still sits at **~35–41 B/cyc** — abs/SEL + REDUX dilate the LDS wall; **not** the same ceiling as A-only ADD.

LDS.32 diagnosis (short): **schedule yes (interleave +32% B/cyc), membar no** (BAR=0, SET/WAIT framing only). Prefer **LDS.64** for the Simt contig headline (~64% of width roof).

### 3.4 Simt artifacts

| what | path |
|------|------|
| fp32 stock + contig-warp remasure | `/workspace/sv567_analysis_draft/sv5_fp32_compare_20261022/` (box) · `/tmp/sv5_fp32_compare_20261022` (pto-b10) |
| IPC plots (stock + contig) | `/workspace/sv567_analysis_draft/sv5_ipc_20261022/` |
| SV1 A-only LDS.32 notes | `/workspace/sv567_analysis_draft/sv1_add_lds32_ipc_20261022/NOTES.md` |

---

## 4. Answers (Lok)

| question | answer |
|----------|--------|
| SIMD primary? | **Compute-bound / issue+latency.** Report **reduce-max 35.01** and **VAND absmax 31.39**; legacy VMUL abs 19.65 is historical. |
| Mix = LD + 1VMAX + 1VCGMAX + 1ST(32B)? | **Per trip ≈ 1 LD + 1 abs-op + 1 VCGMAX + 1 ST**; **abs-op is VAND now**, not VMAX. Per dual64 row: **2×** each. ST is reduced-only (~32–64 B), not full tile. |
| Why 35 not ~64? | Head/tail + shallow ILP; **mid already ~51 / 64 ≈ 80%**. Deeper unroll/SWP is the lever, not a lower mem roof. |
| VAND dual-issue? | **Yes** (ADD unit); VCGMAX still EXQ0-only. ~½ VAND on Q0 → EXU 3/row → ~43 roof; mid **97%**. |
| Stock Simt &lt;2? | **Yes — 1.65 el/cyc**, bank-bound; need **per-warp contig**. |
| SV1 ~100 B → 25 el? | **Yes** as **load-only contig LDS.32 roof**. Contig-warp redux now: **~41 B (~10 el)** LDS.64 window / **~4.5** VF el/cyc — not that roof. |

---

## 5. Provenance

| arm | OUT (pto-b10) | OUT (box) |
|-----|---------------|-----------|
| SoftLib dual64 reduce-max / VMUL absmax | `/tmp/sv5_reducemax_vs_absmax_20261022` | `sv5_simd_baseline/out/sv5_reducemax_vs_absmax_20261022/` |
| SoftLib dual64 **VAND absmax** | `/tmp/sv5_vand_absmax_20261022` | `sv5_simd_baseline/out/sv5_vand_absmax_20261022/` |
| SoftLib dual64 (legacy VMUL abs only) | `/tmp/sv5_ccs_softlib_fix_20261022` | `sv5_simd_baseline/out/ccs_softlib_fix_20261022/` |
| fp32 stock + contig-warp | `/tmp/sv5_fp32_compare_20261022` | `sv567_analysis_draft/sv5_fp32_compare_20261022/` |
| A-only contig / stride64 | `/tmp/lds_aonly_stride64_20261022` | under analysis draft / prior OUT |
| IPC dumps + PNGs | — | `sv567_analysis_draft/sv5_ipc_20261022/` |
| npu_skills rooftop | tip `e5e7ef2` on pto-b10 | `sv567_analysis_draft/npu_skills_rooftop_20261022/` |

**Parked:** fp16→fp32 cast, keep_in_rf STK/LDK, TileLang f16 `ub_stream` 1.488 — revisit under reduce-sum if cast matters.

Companion: rolling-20 IPC detail in `sv5_ipc_20261022/NOTES.md`, `REDUCEMAX_MIDPOINT_IPC.md`, `ABSMAX_VAND_MIDPOINT_IPC.md`, `ABSMAX_VAND_VS_Q0.md`.


## Membar stages / ST-in-roof (2026-10-22)

Full write-up: [`SV5_SIMD_MEMBAR_STAGES_20261022.md`](./SV5_SIMD_MEMBAR_STAGES_20261022.md).

**Roof:** SoftLib dual64 row `max(LD,EXU,ST)` with **ST=2** (ties SLIDE on reduce-max → still 64 el/cyc; VAND EXU=3 binds).

**Simt contig+REDUX membar sparse2 (warp=32, 2 writers/warp intent):** whole **6.35** el/cyc; LD-stage **29.6 B/cyc (23% of 128)** — **not ~50%**. Measured STS=4 (1/warp). ~50% is A-only LDS.64 vs 128 roof.

**SoftLib membar** separates ST (visible ~0.66 VST/cyc) but spill drops whole-VF to ~12.7; fused 35/31 remains primary.

