# ANALYSIS — SB8 per-segment instr mix + IPC (G32 SoftLib ≻ SIMD)

**Identity:** Voice **SB8** = **SV8 Case-3 e2e** harness case. No literal `SB8`/`sb8` tree.
**Primary pair (SoftLib SIMT ≻ SIMD):** **G32** R64×C128 E=8192 · `era=case3-quant-bcast` · `reduce_axis=inner` · SoftLib `axis=1xG`.
**NOT the G16 pair:** G16 simt-asc 0.286 vs size-1 SIMD 0.064 was the wrong target for this voice ask — see footnote + `/workspace/sv8_g16_e2e_t726u/ANALYSIS_SB8_SEGMENT_IPC.md` (superseded for SoftLib headline).
**When:** Asia/Hong_Kong 2026-10-29 · method: `tilelanggym-st-report` § Per-segment + **PR9 opcode-lanes+IPC** (t773u).
**Packs:** SoftLib `/mnt/fluxdata/happybot/sv8_softlib_simt_c3_g32_t740u/` · VL-res under `/mnt/fluxdata/happybot/sv8_g16_e2e_t726u/out/dsl_vlres/` · box `/workspace/sv8_g32_sb8_t772u/` (+ dumps `/workspace/sv8_g32_sb8_t772u/dumps/`, plots `/workspace/sv8_g32_sb8_t772u/plots/`).
**IPC definition:** segment proxy = `calls / sum_cycles(instr_exe.csv cycles column)` (pipe-overlapped; **cyc% need not sum to 100%**). Body IPC ≈ `body_calls / VF_cyc`. Running IPC W=20: **PR9 overlay PRESENT** (`plot_style=opcode_lanes_ipc`).

## Headline KPI

| arm | schedule / layout | VF row | VF_cyc | el/cyc (E/VF) | body_calls | body IPC≈ | mid IPC W=20 (official) | PR9 mid-med |
|-----|-------------------|--------|-------:|--------------:|-----------:|----------:|------------------------:|------------:|
| **SoftLib SIMT** | dual_drain loadall+absall T512 · KEEP-X RF · `axis=1xG` | `VF_SIMT` | **1210** | **6.7702** | 3168 | 2.6182 | **3.30** | **3.40** |
| **SIMD VL-res** | KEEP X in VL · `vcmax(group=1)×NG` · **VLDS=0** | `VF` | **1964** | **4.1711** | 2581 | 1.3142 | **1.45** | **1.35** |

**Ratio el/cyc SoftLib÷SIMD = 1.623×** (SoftLib faster). VF gap ≈ 754 cyc (SIMD longer).
Official mid IPC from t740u absolute-cycle mid; PR9 mid-med = median over mid 15–85% of first-issue VF window (plot VF SoftLib≈1178 / SIMD≈1935 — slightly shorter than official VF; plot-el/cyc SoftLib≈6.95 / SIMD≈4.23 — **KPI stays official VF**).

Sources: SoftLib tag `wc_c3e2e_back_loadall_absall_dual_drain_g32_t512_e8192` · VL-res `sv8d_vlres_r64_c128_g32_t32_vl_resident` · dumps `dumps/softlib_dual_drain_g32.instr_log.dump` · `dumps/vlres_g32.instr_log.dump`.


## ISA-mix pointer (t546u)

Full **ISA mix SIMT vs SIMD (per 32×4B → el/cyc)** bridge lives in primary [`ANALYSIS.md`](ANALYSIS.md) §9 (JSON [`plots/sb8_g32_isa_mix_per32.json`](plots/sb8_g32_isa_mix_per32.json)). This segment note keeps per-segment mix+IPC + PR9; do not duplicate the ops/N table here.

## PR9 running-IPC + instr-mix overlay (both arms)

| arm | PR9 overlay (lanes+IPC) | staged (+ stage bounds) | one-line |
|-----|-------------------------|-------------------------|----------|
| SoftLib | [`plots/softlib_simt_c3e2e_dual_drain_g32_running_ipc_pr9.png`](plots/softlib_simt_c3e2e_dual_drain_g32_running_ipc_pr9.png) | [`…_pr9_staged.png`](plots/softlib_simt_c3e2e_dual_drain_g32_running_ipc_pr9_staged.png) | Top: LDS→REDUX/SEL/FSETP→FDIV→FMUL→STS issue marks; bottom: IPC ramp to ~4.3, mid-med **3.40** |
| SIMD VL-res | [`plots/simd_vlres_g32_running_ipc_pr9.png`](plots/simd_vlres_g32_running_ipc_pr9.png) | [`…_pr9_staged.png`](plots/simd_vlres_g32_running_ipc_pr9_staged.png) | Top: VLDI/VAND/VMAX/VCMAX/VDIV/VDUP/VSEL/VMUL/VSTI fused across VF; bottom: flat ~1.1–1.5, mid-med **1.35** |

**Note:** Older `benches/sv8/plots/*_running_ipc.png` without `_pr9` are **IPC-only** (t740u) — **not** PR9 overlay. Prefer `*_pr9*.png` for stage / mix claims.

Absolute box paths (parent attach):
- `file:///workspace/sv8_g32_sb8_t772u/plots/softlib_simt_c3e2e_dual_drain_g32_running_ipc_pr9.png`
- `file:///workspace/sv8_g32_sb8_t772u/plots/softlib_simt_c3e2e_dual_drain_g32_running_ipc_pr9_staged.png`
- `file:///workspace/sv8_g32_sb8_t772u/plots/simd_vlres_g32_running_ipc_pr9.png`
- `file:///workspace/sv8_g32_sb8_t772u/plots/simd_vlres_g32_running_ipc_pr9_staged.png`

## Stage mid-IPC (defining-opcode windows · W=20 mid 15–85%)

Window = first..last issue of defining opcodes (**overlap OK**). SoftLib dual_drain overlaps heavily; SIMD VL-res is a **fused** loop so every stage mid ≈ body mid (~1.35).

### SoftLib SIMT (official mid 3.30 / PR9 mid-med 3.40)

| segment | defining opcodes | VF-local ticks | mid_med IPC | mid_mean IPC | n_defining |
|---------|------------------|---------------:|------------:|-------------:|-----------:|
| `S_load` | LDS | 51–695 | **3.25** | 3.0724 | 256 |
| `S_reduce` | REDUX, FADD, SEL, FSETP_I | 190–1037 | **3.5** | 3.4901 | 1536 |
| `S_scale` | FDIV_I | 320–1065 | **3.5** | 3.4719 | 256 |
| `S_eltwise` | FMUL | 369–1137 | **3.45** | 3.3511 | 256 |
| `S_store` | STS | 421–1177 | **3.4** | 3.2209 | 256 |

### SIMD VL-res (official mid 1.45 / PR9 mid-med 1.35)

| segment | defining opcodes | VF-local ticks | mid_med IPC | mid_mean IPC | n_defining |
|---------|------------------|---------------:|------------:|-------------:|-----------:|
| `S_load` | VLDI | 13–1857 | **1.35** | 1.363 | 128 |
| `S_reduce` | VCMAX, VMAX, VAND | 16–1889 | **1.35** | 1.3639 | 640 |
| `S_scale` | VSEL, VDIV, VDUP | 18–1918 | **1.35** | 1.3627 | 1536 |
| `S_eltwise` | VMUL | 77–1922 | **1.35** | 1.365 | 128 |
| `S_store` | VSTI | 84–1929 | **1.35** | 1.3651 | 128 |

JSON: [`plots/sb8_g32_stage_mid_ipc.json`](plots/sb8_g32_stage_mid_ipc.json).

## Per-segment instr mix + IPC (CSV call aggregates · both arms)

| segment | SIMD opcodes (top) | SIMD IPC | SIMD cyc% | SoftLib opcodes (top) | SoftLib IPC | SoftLib cyc% | winner | note |
|---------|--------------------|---------:|----------:|----------------------|------------:|-------------:|--------|------|
| `S_load` | MOV_ALIGNv2(MTE2)x1, RV_VLDI:128 | 0.0487 | 134.9% | SIMT_LDS:256, MOV_ALIGNv2(MTE2)x1 | 0.0232 | 914.1% | **SIMD** | Δcyc(SIMD−Soft)=-8412 |
| `S_reduce` | RV_VCMAX:256, RV_VMAX:256, RV_VAND:128, RV_VCMP_LT:2, RV_VCMP_GE:2, RV_VCI:1 | 0.1004 | 327.7% | SIMT_REDUX:256, SIMT_IADD_I:512, SIMT_FSETP_I:512, SIMT_SEL:512, SIMT_FADD:256 | 0.1159 | 1459.8% | **SIMD** | Δcyc(SIMD−Soft)=-11227 |
| `S_scale` | RV_VSEL:1024, RV_VDIV:256, RV_VDUP:256, RV_VDUPS:7 | 0.1155 | 679.9% | SIMT_FDIV_I:256 | 0.0556 | 380.8% | **SoftLib** | Δcyc(SIMD−Soft)=8746 |
| `S_eltwise` | RV_VMUL:128 | 0.1250 | 52.1% | SIMT_FMUL:256 | 0.1111 | 190.4% | **SIMD** | Δcyc(SIMD−Soft)=-1280 |
| `S_store` | MOV_ALIGNv2(MTE3)x1, RV_VSTI:128 | 0.0510 | 128.9% | SIMT_STS:256, MOV_ALIGNv2(MTE3)x1 | 0.0388 | 547.4% | **SIMD** | Δcyc(SIMD−Soft)=-4092 |
| `S_sync` | (none) | N/A | 0.0% | (none) | N/A | 0.0% | **N/A** | Δcyc(SIMD−Soft)=0 |

**Winner column metric:** SoftLib if SIMD `sum_cyc` ≫ SoftLib on that segment (SoftLib cheaper cycle tax); SIMD if reverse. **Caveat:** SoftLib mid IPC≈3.3 ⇒ high pipe-overlap — SoftLib `sum_cyc` can exceed VF while still finishing wall VF sooner. Overall race = **VF / el/cyc**.

### Gap drivers (rank by Δcyc = SIMD−SoftLib sum_cyc; SoftLib≻SIMD ⇒ positive helps SoftLib)

1. `S_scale` — SIMD≈13354 vs SoftLib≈4608 (Δ=+8746) — clearest SoftLib segment win (VSEL paint vs `FDIV_I`)
2. `S_sync` — 0 vs 0
3. `S_eltwise` / `S_store` / `S_load` / `S_reduce` — SoftLib higher overlapped sum_cyc but denser issue (see PR9 lanes) still wins wall VF

## Overall conclusion

On SB8 (=SV8 Case-3 **G32**), SoftLib SIMT dual_drain T512 reaches **6.7702 el/cyc** (VF **1210**) vs SIMD VL-resident **4.1711** (VF **1964**) — SoftLib÷SIMD ≈ **1.62×**. PR9 overlays make the difference visual: SoftLib lanes show staged LDS→REDUX/SEL→FDIV→FMUL→STS with body mid-med **~3.4**; SIMD lanes show a **fused** VL-res loop (VSEL/VCMAX/VMAX/…) with flat mid-med **~1.35**. SoftLib wins on **schedule density + cheap S_scale**, not on “avoid second VLDS” (SIMD already VLDS=0).

## How to remake the plots (harness)

Forced checklist: `REGRESSION_salvage_2026-09-17/L6_ANALYSIS_INVENTORY.md` § **PR9 running-IPC + instr-mix overlay (t773u)**.
Recipe script: `REGRESSION_salvage_2026-09-17/scripts/make_pr9_running_ipc_overlay.py`.
```bash
PYTHONPATH=/workspace/tilelangGym-pr8-tip \
  python /workspace/REGRESSION_salvage_2026-09-17/scripts/make_pr9_running_ipc_overlay.py \
    --dump /workspace/sv8_g32_sb8_t772u/dumps/softlib_dual_drain_g32.instr_log.dump \
    --out-png /workspace/sv8_g32_sb8_t772u/plots/softlib_simt_c3e2e_dual_drain_g32_running_ipc_pr9.png \
    --title "sv8 / SoftLib SIMT Case-3 e2e dual_drain G32 T512 (axis=1xG) — running IPC (W=20)" --E 8192
# same for vlres_g32.instr_log.dump → simd_vlres_g32_running_ipc_pr9.png
# require JSON plot_style=opcode_lanes_ipc
```

## G16 footnote (wrong SoftLib target — superseded)

| arm | el/cyc | VF | note |
|-----|-------:|---:|------|
| G16 simt-asc | 0.2864 | 28603 | stock Ascend SIMT — **not** SoftLib |
| G16 size-1 SIMD | 0.06361 | 128790 | not VL-res SoT |
| **G32 SoftLib dual_drain (this file)** | **6.7702** | **1210** | **primary SoftLib≻SIMD pair** |
| **G32 VL-resident SIMD** | **4.1711** | **1964** | SIMD_best |

## MISSING / next-cmd

- PR9 opcode-lanes+IPC for both G32 arms: **PRESENT** (this turn).
- Older IPC-only PNGs remain for archive; do not cite as PR9.
- Body4 firstLDS→lastSTS window: SoftLib LDS→STS span is the PR9 body; not further split beyond defining-opcode stages.
- Literal SB8 tree: none found.

JSON harvest: `/workspace/sv8_g32_sb8_t772u/sb8_g32_segment_harvest.json` · stage mid `/workspace/sv8_g32_sb8_t772u/plots/sb8_g32_stage_mid_ipc.json`.
