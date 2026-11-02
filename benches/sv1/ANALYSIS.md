# SV1 ANALYSIS — short-live stream eltwise (SimtVF vs handwritten SIMD twin) — clean regress t813u / PR9 regen t815u

**Date:** 2026-10-31 HKT (KPI) · **PR9 / report regen:** 2026-11-02 HKT (t815u) · **SOC:** Ascend950PR_9599 opsim · **CANN:** cann_92b2 · **TileLang:** `/mnt/fluxdata/happybot/projects/tilelang-pto-isa-st` (simt-asc) · **ptoas:** venv-ptoas-vmi-0.1.3 · **era:** `fp32-stream-eltwise` · **E=8192** (documented; A1) · KPI **el/cyc = E / named VF_cyc**. All KPI numbers measured fresh in pack `glm53_sv1567_sf_mr2_t813u` on 2026-10-31 (`RESULT.tsv` rows `arm=SimtVF-simt-asc` / `arm=SIMD-PTODSL-twin`). PR9 overlays re-emitted 2026-11-02 with tip `tilelangGym-pr8-t705u` `rolling_ipc` (`plot_style=opcode_lanes_ipc`, non-empty `lane_opcodes`).

**SEMANTICS_GATE: PASS — S1 in-kernel / suite numeric check (C=A+B fp32); S2 SimtVF VF_SIMT body vs SIMD twin vf_real_execute_time body (era=fp32-stream-eltwise, reduce_axis=na-eltwise); S3 stream mix only (LDS/FADD/STS vs VLD/VADD/VST — no reduce family); S4 named counters (VF_SIMT / vf_real_execute_time) — A1 / A10.**

**Separation (A10):** SV1 is the stream-eltwise case (load→add→store, UB-bound) — **not** the absmax Motif ladder; its rows are labeled `reduce_axis=na-eltwise` and are never merged into SV5–7 absmax SoT tables. The SV1 SoftLib A-only LDS SoT (Harness V20 ~101 B/cyc class) is a different case/era — not measured here, not merged.

**Format note (t815u):** tip `benches/sv1/plots/` historically shipped IPC-only figures (`*_ex_ipc_30cyc.png`, `*_rolling20_ipc.png`) without opcode-lane overlays. Those are **rejected** as PR9. This report uses only `*_running_ipc_pr9.png` + JSON sidecars with `plot_style=opcode_lanes_ipc` and non-empty `lane_opcodes`.

---

## 1. Aims (SV1)

- **A10:** measure SimtVF + handwritten SIMD twin; write this ANALYSIS on the st-report spine; no fold into absmax SoT.
- **A1:** E documented = **8192** (both arms; E=16384 SIMD TILE-ladder rows carry their own E); named VF counters: SimtVF = `VF_SIMT` row cycles (`core0.veccore0_instr_exe.csv`); SIMD twin = `vf_real_execute_time` sum over VF PUSHQ blocks.
- RF/stream axis: `ST_SV1_TEMPS=frag` (RF fragment temps) vs `shared` (UB-stream temps) — §5.
- Roof framing (design targets, cited not measured): Simt LD.64 A+B ≈ **8 el/cyc** classic lead; LDS.b32 serial roofs 16 (1-in) / 10.667 (A+B); SIMD UB-stream theory ≈ **44 el/cyc** (ref/sv1 §7–§8 framing; Harness V20 for the SoftLib A-only class).

## 2. Test-case design

| arm | kernel | tags | axes |
|-----|--------|------|------|
| SimtVF (TileLang T.SimtVF, `ST_VF_MODE=simt-asc`) | `kernels_sv1/sv1_stream_eltwise.py` | `sv1_e8192_t{T}_{frag\|shared}` | T∈{128,512,1024} × temps∈{frag,shared} |
| handwritten SIMD twin (PTO-DSL, VL=64 = pack **.64**) | `kernels_sv1/sv1dt_tiled_add.py` (+`_sv1dt_tiled_template.py`) | `sv1dt_e{E}_tile{TILE}_vl64` | E=8192 TILE8192 hub; E=16384 TILE∈{2048,4096,8192,16384} ladder |

Gold: `C = A + B` fp32, in-kernel / suite numeric check (S1). Kernel bodies: per-tile MTE GM→UB load A/B → VL64 vload/vadd/vstore → MTE store C (SIMD); `vf_region(threads)` Parallel(E) add with t1/t2/t3 temps (SimtVF).

## 3. Layout / schedule

SimtVF: single `T.Kernel(1)`, whole-E `vf_region` (TILE8192-class), frag temps (RF) vs shared temps (UB). SIMD twin: per-tile MTE stream loop with MTE3↔MTE2 inter-tile priming; VL=64 fixed (`.64` pack).

## 4. SIMD (handwritten twin) results — fresh

| tag | E / TILE | VF_cyc | el/cyc | % of theory ~44 |
|-----|----------|-------:|-------:|----------------:|
| sv1dt_e8192_tile8192_vl64 (hub, same E as SimtVF) | 8192 / 8192 | **279** | **29.3620** | ~67% |
| sv1dt_e16384_tile2048_vl64 | 16384 / 2048 | 816 | 20.0784 | ~46% |
| sv1dt_e16384_tile4096_vl64 | 16384 / 4096 | 640 | 25.6000 | ~58% |
| sv1dt_e16384_tile8192_vl64 | 16384 / 8192 | 558 | 29.3620 | ~67% |
| sv1dt_e16384_tile16384_vl64 | 16384 / 16384 | 519 | **31.5684** | ~72% |

TILE ladder: per-tile MTE fill/drain amortizes 2K→16K (20.1→31.6); per-tile el/cyc saturates at TILE8192 (29.36 both E). S3 hits (hub): VLD=256, VADD=128, VST=128 (+control) — pure stream mix, no reduce family (axis=na).

## 5. SIMT Ascend — SimtVF arm (fresh)

| tag | T | temps | VF_SIMT cyc | el/cyc | % of ~8 |
|-----|---|-------|------------:|-------:|---------:|
| sv1_e8192_t128_frag | 128 | frag (RF) | 4680 | 1.7504 | ~22% |
| **sv1_e8192_t512_frag** ★ | **512** | **frag (RF)** | **3405** | **2.4059** | ~30% |
| sv1_e8192_t1024_frag | 1024 | frag (RF) | 3510 | 2.3339 | ~29% |
| sv1_e8192_t512_shared | 512 | shared (UB temps) | 6840 | 1.1977 | ~15% |

- **T ladder:** T512 peak fresh; T1024 turns down (3510 vs 3405 — address-math overhead grows with warps at this tip).
- **RF vs UB-stream:** frag (RF) **2.00×** shared-temps (2.406 vs 1.198). The `shared` variant stages t1/t2/t3 in UB (temps tower): STS=256 vs 64 (4× store traffic) — the direct **add-in-shared no-temps** peer (ref §6 class) is **MISSING** + next-cmd: `ST_SV1_TEMPS=noshared_temps` variant or hand kernel adding `c_ub[i]=a_ub[i]+b_ub[i]` directly.
- Lowering class (from counts): LDS=128 warp-issues move 2×E×4B=64KB → **16B/lane (float4/LDS.128-class)** on this tip, with heavy address/pred math (SHFI=656, LOP3=512, MOV=256). Era note: ref/sv1 v1 primary stock was **float2/LDS.64-class** at a different TileLang tip — different era; **no cross-era merge** (numbers kept separate; ref is reference only).

## 6. SoftLib SIMT — N/A for SV1 this run

SV1 Aims (A10) cover SimtVF + handwritten SIMD twin. The SoftLib handwritten LDS.32/64 A-only rooftop family (Harness V20 SoT class) is a **separate case** — not measured, not merged (see header note). SoftLib CCE optional when VMI/SIMD twin already SoT for this case — not required. SoftLib SIMT SoT `axis=1xG` applies only to absmax Motif STs (SV5–7), not this eltwise case.

## 7. Running IPC + PR9 overlay figures (FORCED, PRC-IPC-001) — both arms

| arm | png | plot_style | mid-med IPC | lane_opcodes |
|-----|-----|------------|------------:|--------------|
| SimtVF E8192 T512 frag | `plots/sv1_simtvf_running_ipc_pr9.png` | **opcode_lanes_ipc** | 0.50 | S2R, LDS, FADD, STS, END, MOV, LOP3 |
| SIMD twin E8192 TILE8192 VL64 | `plots/sv1_simdt_running_ipc_pr9.png` | **opcode_lanes_ipc** | 2.20 | VLD, VADD, VST |

JSON sidecars co-located: `plots/sv1_*_running_ipc_pr9.json` (top-level + `meta.lane_opcodes` non-empty). **REJECT** IPC-only `*_running_ipc.png` / `*_ex_ipc_30cyc.png` / `*_rolling20_ipc.png` as PR9 compliance.

Plot VF (3352 / 241) < official VF (3405 / 279) — first-issue silence gap; headline KPI from official counters.

## 8. Per-segment opcode mix + IPC + stage mid-IPC (both arms)

**SimtVF T512 frag** (fused body — windows span whole body):

| stage | window | n | mid-med IPC |
|-------|--------:|---:|------------:|
| S_load (LDS) | 0..3186 | 160 | 0.55 |
| S_eltwise (FADD) | 155..3330 | 256 | 0.50 |
| S_store (STS) | 190..3347 | 64 | 0.50 |

**SIMD twin E8192 TILE8192:**

| stage | window | n | mid-med IPC |
|-------|--------:|---:|------------:|
| S_load (VLD) | 0..185 | 256 | 2.40 |
| S_eltwise (VADD) | 7..191 | 128 | 2.30 |
| S_store (VST) | 12..235 | 128 | 2.20 |

SimtVF is **latency-bound** (mid IPC 0.5: per-thread serial 16-element chains; LDS issue→use latency not hidden by 16 warps), not issue-bound — consistent with the roof framing (§1) and ref §7.2 "need ≥256 threads / split stages" conclusion. Fused loop → every stage mid ≈ body mid (explicit).

Source: `plots/sv1_stage_mid_ipc.json` keys `sv1_simtvf` / `sv1_simdt`.

### 8.1 Balanced opcode call_count tables (from `core0.veccore0_instr_exe.csv`)

**SimtVF E8192 T512 frag — body_calls = 2212:**

| opcode | cc | | opcode | cc |
|--------|--:|---|--------|--:|
| SIMT_SHFI | 656 | | SIMT_S2R | 32 |
| SIMT_LOP3 | 512 | | SIMT_LEA | 16 |
| SIMT_FADD | 256 | | SIMT_END | 16 |
| SIMT_MOV | 256 | | scalar math residual | 21 |
| SIMT_IADD_I | 192 | | control group | **63** |
| SIMT_LDS | 128 | | **Σ = 2212 = body_calls** | ✓ |
| SIMT_STS | 64 | | | |

**SIMD twin E8192 TILE8192 — body_calls = 576:**

| opcode | cc | | opcode | cc |
|--------|--:|---|--------|--:|
| RV_VLD | 256 | | scalar math residual | 13 |
| RV_VST | 128 | | RV_PSET | 2 |
| RV_VADD | 128 | | RV_VLOOPv2 | 1 |
| RV_SEND | 3 | | control group | **45** |
| | | | **Σ = 576 = body_calls** | ✓ |

Source: `plots/sv1_opcode_mix.json` keys `SV1_SimtVF_vs_SIMDtwin::A()` / `::B()`.

## 9. ISA mix SimtVF vs SIMD twin (per N×4B → el/cyc) — FORCED

Unit line: `N=32 · E=8192 · units=256 · IPC=body_calls/VF_cyc (body)`.

| family | SimtVF ops/N | SIMD twin ops/N | Δ | note |
|--------|-------------:|----------------:|---:|-------|
| load (LDS / VLD) | 0.5000 | 1.0000 | **+100%** | float4 LDS.128 vs VL64 (only 2 warp-equivs/issue) |
| eltwise (FADD / VADD) | 1.0000 | 0.5000 | −50% | |
| store (STS / VST) | 0.2500 | 0.5000 | +100% | |
| other data (SHFI/LOP3/MOV/LEA/IADD_I… + scalar) | 6.6445 | 0.5742 | −91% | SimtVF address/pred math dominates |
| control | 0.2461 | 0.1758 | — | |
| **total ops/N** | **8.6406** | **2.2500** | **−74%** | balanced ✓ |

Bridge:
- pred_el/cyc (SimtVF) = 0.6496 × 32 / 8.6406 = **2.4059**; measured **2.4059** (pred/meas = 1.0000)
- pred_el/cyc (SIMD) = 2.0645 × 32 / 2.2500 = **29.3620**; measured **29.3620** (pred/meas = 1.0000)
- pred_ratio = (0.6496/2.0645) × (2.2500/8.6406) = **0.0819**; measured ratio = 2.4059/29.3620 = **0.0819** → **verdict OK**

**Attribution:** unlike the absmax pairs, the SIMD twin wins on **IPC (3.2×)** more than ops/N (3.8×) — the SimtVF arm is latency-bound (mid IPC 0.5, serial per-thread chains) while the VL64 vector twin streams at IPC ~2.06; the measured 12.2× el/cyc gap decomposes exactly into issue-economics × IPC.

Source: `plots/sv1_isa_mix_perN.json` pair `SV1_SimtVF_vs_SIMDtwin`.

## 10. Conclusion

- **SimtVF SoT (fresh, era=fp32-stream-eltwise, E=8192): 2.4059 el/cyc** — `sv1_e8192_t512_frag` (T512 peak; T1024 2.334; shared-temps 1.198 = RF/stream axis measured, tower variant).
- **Handwritten SIMD twin SoT: 29.3620 el/cyc at E=8192 TILE8192 VL64** (31.5684 at E=16384 TILE16384; per-tile saturation at TILE8192). ~67% of theory rooftop ~44.
- **Both-arm compare (same E=8192): SIMD twin ≻ SimtVF ≈ 12.2×** — PR9 both arms + balanced mixes + ISA-mix bridge shipped (§7–§9); attribution: latency-bound SimtVF vs streamed VL64 vector loop.
- Honest gaps: direct add-in-shared no-temps peer **MISSING** (next-cmd in §5); T64/T2048 rungs not measured; sv1d_e256 fixed probe skipped (subsumed); ref/sv1 v1 numbers kept reference-only (different era/tip lowering).

## 11. TRACE pointer

Pack: `/mnt/fluxdata/happybot/glm53_sv1567_sf_mr2_t813u/EXPLORATION_TRACE.md` · artifacts: `RESULT.tsv` (SV1 rows), `out/sv1_harvest.json`, `out/opcode_mix.json`, `out/isa_mix_perN.json`, `out/stage_mid_ipc.json`. This regen mirrors SV1 slices under `plots/sv1_*.json`.

**PRC cites:** PRC-IPC-001 (both-arm PR9). PRC-MEM-001/002 are Case-3/SV8-scoped — N/A here.
