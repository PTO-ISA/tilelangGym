# SV1 analysis — short-live stream eltwise (Simt + SIMD)

**Date:** 2026-10-17 HKT · **SOC:** Ascend950PR_9599 opsim · **CANN / PTOAS:** cann_92b2 / PTOAS nightly (`config.toml` pins) · **TileLang tip:** `e700041f` (float4 / max_vector_bits 64→128)

This is the **only** SV1 analysis doc. Older NOTES / VERDICT / rooftop-32 writeups were removed.

---

## 1. Aims

- Show a **short-live** stream eltwise where inputs do **not** need to stay in register: load → add → store.
- Demonstrate how **SIMD LD/ST high bandwidth** performs well on this **UB-bound** case.

Kernel: one TileLang file `sv1_stream_eltwise.py` (`vf_region` + `ST_VF_MODE`); handwritten SIMD twin `sv1d_stream_eltwise.py` / tiled `sv1dt_*` via `scripts/gen_sv1dt_kernel.py` (VL=64). Aligned Simt tags `sv1tt_e{E}_tile{TILE}_t{T}` (`SUMMARY_sv1_aligned_e16k_tile8k.tsv`); aligned SIMD tags `sv1dt_e{E}_tile{TILE}_vl64` (`SUMMARY_sv1d_aligned_e16k_tile8k.tsv`).

---

## 2. Test-case design

| Axis | Choices |
|------|---------|
| Problem size | **E = 16K** f32 elements |
| Tile sizes | **2K / 4K / 8K / 16K** @ fixed **E=16K** (both Simt and SIMD) |
| SIMT threads | **128 / 256 / 512** @ fixed **E=16K, TILE=8192** (not untiled) |
| VMI / handwritten SIMD DSL | **VL=64**, **same E=16K + TILE 2K/4K/8K/16K** (per-tile MTE stream) |

**Footprint note (8K tile):** one f32 array @ 8K = **32 KB**; A+B = **64 KB**; A+B+C = **96 KB** — can fit UB / approach whole MRF (128 KB). After the tile ladder we **focus on 8K** as the compare hub vs SIMD.

This set is the full sensitivity for the UB-bound case: **tile size**, **thread dim**, **Simt vs SIMD**, and whether current `T.Parallel` / DSL already meet **xx%** of the Simt / SIMD rooftops (sections below).

**Pin note:** aligned Stock Simt + tiled SIMD numbers below are from **2026-10-17** opsim under **cann_92b2 + tip e700041f** (Simt) / **cann_92b2** (SIMD DSL). Prior whole-E SIMD size ladder (`SUMMARY_sv1d_size_ladder.tsv`) and older cann_91b3 Simt tables are **appendix-only** — do not mix into §3 / §4.1–4.2 primary tables.


## 3. SIMD performance

Handwritten DSL **VL=64**, **tiled** short-live stream @ fixed **E=16384** (`data/SUMMARY_sv1d_aligned_e16k_tile8k.tsv`):

| TILE | VF_cyc | IPC | el/cyc | us |
|-----:|-------:|----:|-------:|---:|
| 2048 | 824 | 1.2913 | **19.8835** | 7.69 |
| 4096 | 644 | 1.6211 | **25.4410** | 5.87 |
| **8192** | **560** | **1.8464** | **29.2571** | **4.99** |
| 16384 | 520 | 1.9788 | **31.5077** | 4.55 |

Kernel: `sv1dt_*` / `scripts/gen_sv1dt_kernel.py` — per tile MTE load A/B → VL=64 vload/vadd/vstore → MTE store C. Tag `sv1dt_e16384_tile{TILE}_vl64`.

**Theory est (Lok): ~44 el/cyc.** Plot SoT on `plots/sv1d_e16k_tile8k_ex_ipc_30cyc.png` uses derived roof **42.67** (IPC→el conversion); cite Lok’s **~44** as the design target and **42.67** as the plot’s derived roof.

From that same plot footer (authoritative mid-body IPC @ **E=16K TILE=8K**):

| Metric | Value | vs roof |
|--------|------:|---------|
| Mid-body median IPC | **2.20** → **~34.1 el/cyc** | **≈80%** of 42.67 · **≈78%** of 44 |
| Peak excl fill/drain IPC | 2.77 | — |
| Peak mid IPC | 2.63 | — |
| Early fill IPC | 3.37 | — |
| Overall VF (TILE8K) | **29.26 el/cyc** | **≈66%** of 44 · **≈69%** of 42.67 |

![SIMD E16K TILE8K IPC](plots/sv1d_e16k_tile8k_ex_ipc_30cyc.png)

**Head-to-head vs aligned Simt (same E=16K, matched TILE):**

| TILE | SIMD el/cyc | Simt T512 el/cyc (§4.1) | ratio |
|-----:|------------:|------------------------:|------:|
| 2048 | 19.88 | 2.44 | ~8× |
| 4096 | 25.44 | 2.45 | ~10× |
| **8192** | **29.26** | **2.40** | **~12×** |
| 16384 | 31.51 | 2.44 | ~13× |

**Takeaway:** at matched **TILE=8K E=16K**, SIMD DSL lands **29.26 el/cyc** overall VF (~66% of Lok’s ~44). Vs Simt TILE8K@T512 (**2.40**), SIMD is ~**12×**. Larger TILE amortizes per-tile MTE fill/drain (2K→16K: 19.9→31.5 el/cyc). Prior whole-E E-ladder is appendix-only — not the §3 compare SoT.



## 4. SIMT performance

Stock pack under tip e700041f is wider-vector (LDS:FADD:STS ≈ **256:512:128** in the aligned harvest — float4-class). Lead framing still uses Lok’s **LD.64 A+B ≈ 8 el/cyc** roof for % columns (same A+B dual-load story).

**Lead framing (Lok):** LD.64 A+B theory ≈ **8 el/cyc**. Aligned stock Simt lands **~2.40–2.45 el/cyc** → about **~30%** of that ~8 roof under **cann_92b2 + e700041f**. (Prior cann_91b3 untiled T512 **3.71** / ILP16 **4.08** are older pins — see §4.3.) Alternate 1R+1W@16 is a footnote in §4.4.

Data: `data/SUMMARY_sv1_aligned_e16k_tile8k.tsv` (+ `.json`, `NOTES_sv1_aligned_e16k_tile8k.md`).

### 4.1 Tile-size performance (E=16384, fixed T=512)

| Case | VF_cyc | el/cyc | % of LD.64 A+B (~8) |
|------|-------:|-------:|--------------------:|
| TILE2048 @ T512 | 6721 | **2.44** | 30% |
| TILE4096 @ T512 | 6692 | **2.45** | 31% |
| **TILE8192 @ T512** | 6825 | **2.40** | **30%** |
| TILE16384 @ T512 | 6721 | **2.44** | 30% |

![Tile vs VF el](plots/sv1_tile_vs_vf_el_instr.png)

**Point:** at fixed T=512, tile size is **flat (~2.40–2.45 el/cyc)**. **8K** remains the focus tile for Simt↔SIMD compare (footprint / hub), not because it uniquely wins the ladder.

### 4.2 Thread-dim performance (E=16384, TILE=8192)

| Case | VF_cyc | el/cyc | IPC_body | % of LD.64 A+B (~8) |
|------|-------:|-------:|---------:|--------------------:|
| TILE8192 @ T128 | 8221 | **1.99** | 0.506 | 25% |
| TILE8192 @ T256 | 7041 | **2.33** | 0.600 | 29% |
| **TILE8192 @ T512** | 6825 | **2.40** | **0.638** | **30%** |

![Thread vs IPC/el](plots/sv1_thread_vs_ipc_el.png)

![Simt Body4 IPC TILE8K](plots/sv1_simt_e16k_tile8k_body4_ex_ipc_30cyc.png)

**Point:** more threads raise stock Simt el/cyc on the **8K tile**; best aligned stock is **TILE8192 @ T512 = 2.40**.

**Body4 plot:** `plots/sv1_simt_e16k_tile8k_body4_ex_ipc_30cyc.png` is the **aligned TILE8K@T512** Body4 capture under cann_92b2. Aligned **IPC_body = 0.638** also in `SUMMARY_sv1_aligned_e16k_tile8k.tsv`. (Legacy `sv1_simt_e16k_body4_ex_ipc_30cyc.png` = prior untiled — do not use as SoT.)

### 4.3 ILP16 @ T128 (schedule sensitivity — prior pin)

| Case | el/cyc | % of LD.64 A+B (~8) | Status |
|------|-------:|--------------------:|--------|
| ILP16 @ T128 | **4.08** | **51%** | **Prior untiled / older pin — not re-run under align** |

![ILP16 timeline](plots/sv1_ilp16_T128_timeline_ipc_el.png)

**Point:** historical hand ILP16 at T128 beat **older** stock untiled T512 (**4.08** vs **3.71**). That claim is **not** restated against aligned TILE8K@T512 **2.40** until ILP16 is re-measured under cann_92b2 + e700041f.

### 4.4 Footnote — alternate roof only (do not mix into §4.1–4.3)

Prior pack docs also used **1R+1W @ 128 B each → 16 el/cyc**. Against that roof: aligned TILE8K@T512 ≈ **15%**. Lead framing for Simt stock stays **~8**; keep **16** only here.

### 4.5 Side note: LD.32

- Theory: LD.32 can do **128 B/cyc**, but **too many instructions / hard to schedule**.
- For LD.32 A+B (dual load on one R port), **theory best ~8 el/cyc**.
- SoftLib / best-thread cases land around **~4–5 el/cyc** VF average under realistic schedules:
  - SoftLib **A-only** LDS.b32 contig T128: VF **4.54** el/cyc @ E=8192 (label **A-only** — not A+B).
  - Aligned stock Simt (A+B): **~2.40–2.45** el/cyc under new pins.
  - SoftLib A-only T512 / T1024 VF **8.5 / 9.9** el/cyc are **A-only** — do **not** mix with A+B theory without labeling.
- SoftLib A-only LDS.b32 contig T128: LDS-window **101 B/cyc (79% of 128)**, LDS-window elems **25.3**; whole-VF elems **4.54** — see `/workspace/st_out_cce_vadds_const/reports/`.
- SoftLib A+B LDS.b32 contig T128: **~52 B/cyc** (~41% of 128) from LDS width BW report.

### 4.6 SoftLib LDS.b32 / LDS.b64 BW — PURE port isolation (2026-10-17)

**SoT:** `/workspace/st_out_cce_lds_bw_confirm/reports/LD32_LD64_BW_CONFIRM.md` (pto-b10, Ascend950PR_9599, E=8192 T=128).

Prior SoftLib T128 ~**101 B/cyc** (section 4.5 / §5 baseline) was **compiler IADD↔LDS interleave** (`I1L1…`) — **not** LDS-port isolation. Confirmed PURE numbers below require:

1. **Self-scheduled (自己排好):** hand/manual schedule with explicit **IADD-then-LDS** separation (`sep_ptr` — all address IADD/LEA first, hard phase barrier, then LDS burst alone). Disasm must show LDS region with **zero** IADD/LEA (`separation_ok=True`, motif `L64` / `L32`).
2. **MI schedule disabled** (Lok: 「MR schedule disabled」 → SoftLib/SV1 flag): `-mllvm -enable-misched=false -mllvm -enable-post-misched=false`. Default MI keeps MIXED interleave even with `sep_ptr`.

| Width | Tag | Disasm | Peak B/cyc | Sustained first→last B/cyc | vs 128 roof |
|-------|-----|--------|----------:|---------------------------:|------------:|
| **LDS.b32 (LD32)** | `bw_b32_sep_ptr_e8192_t128_mi_off` | **PURE** `L64` | **128.00** | **~115** (114.98) | **100%** peak · **~90%** sustained |
| **LDS.b64 (LD64)** | `bw_b64_sep_ptr_e8192_t128_mi_off` | **PURE** `L32` | (burst†) | **~64.5** (64.50) | **~50%** sustained |

† LD64 short-window rolling peak can exceed 128 (1× LDS.b64/cyc × 256 B); **claim metric is sustained first→last = 64.50**.

**One-liner:** under PURE / MI-off / self-scheduled IADD→LDS: **LD32 peak 128 B/cyc, sustained ~115; LD64 ~64.5 B/cyc (~half)**. Isolation costs whole-VF (LD32 PURE whole-VF **7.19** el/cyc vs T1024 **9.89**) — do not mix PURE port-BW with MIXED whole-VF chase (§5).

---

## 5. CCE T128 manual-schedule side note + verdict

SoftLib CCE A-only LDS.b32+const, E=8192 T=128 (reports under `/workspace/st_out_cce_vadds_const_sched/reports/` + `/workspace/st_out_cce_vadds_const/reports/`):

| Probe | VF | whole-VF el/cyc | LDS-win B/cyc | vs roof |
|-------|---:|----------------:|--------------:|---------|
| Baseline load-all (`-O2` MI on) | 1806 | **4.54** | **101.14** (79% of 128) | LDS-win **25.3** el/cyc |
| **Manual best** (`loadall_barrier` / `iadd_first`) | **882** | **9.29** | **101.45** | **~94%** of T1024 **9.89**; **~37%** of LDS-win roof **25.3** |
| MI-off (`-mllvm -enable-misched=false`) | 2381 | **3.44** | 48.91 | **worse** than `-O2` |
| `-O0` | 3380 | 2.42 | 44.34 | worse |
| T512 / T1024 (same SoftLib style) | 964 / **828** | **8.5 / 9.89** | LDS-win B/cyc **falls** as warps↑ | T1024 is the whole-VF roof reference |

| Probe | Result |
|-------|--------|
| All-four IADD+LDS+FADD+STS same cycle | **0** cycles at T128/512/1024 |
| Latency alone forbid ~4-issue? | **No** — schedule/port limited |
| Can T128 schedule alone beat T1024 9.89? | **No** — best **9.29**; win is **more warps (32 vs 4)**, not MI order |

### Verdict (this SV1 case)

| Path | Measured | Roof framing | % of roof |
|------|----------|--------------|----------:|
| **SIMD DSL VL=64 (TILE8K@E16K)** | **29.26 el/cyc** overall VF; mid-body **~34.1** | Lok **~44** (plot **42.67**) | **~66%** overall · mid **~80%** of 42.67 |
| **Simt aligned stock (TILE8K@T512)** | **2.40** el/cyc | LD.64 A+B **~8** | **~30%** |
| same Simt vs alt roof | 2.40 | 1R+1W@128B **16** | **~15%** |
| CCE hand (A-only LDS window) | **101 B/cyc** | 128 B | **79%** window BW |
| CCE manual T128 whole-VF | **9.29** el/cyc | T1024 **9.89** / LDS-win **25.3** | **~94%** of T1024 · **~37%** of LDS-win roof |

**Bottom line:** SIMD DSL tiled @ **E=16K TILE=8K** delivers **29.26 el/cyc** overall VF (**~66% of ~44**). Aligned Simt stock (**TILE=8192 @ T=512**) is **2.40 el/cyc** — about **30%** of the **~8** LD.64 A+B theory (**~15%** on the 16 el/cyc 1R+1W roof). Matched-TILE head-to-head: SIMD **29.26** vs Simt **2.40** (~**12×**). CCE hand work shows LDS BW *can* approach the 128 B roof on A-only windows (~79%); manual T128 schedule reaches **9.29** el/cyc (**~94%** of T1024) but **MI-off is worse** — need more warps for the whole-VF roof, not just a better MI order.

---

## 6. Shared vs fragment — direct GM→shared only (no S-MEM staging tower)

**S-MEM tower removed.** The old shared arm used `t1/t2/t3 = alloc_shared` staging (shared→shared copies) and looked **−37%/−50%** vs fragment — that pattern is **gone**. No shared→fragment `T.copy` either.

**Kept path (SDS / direct GM→shared):**

1. `T.copy(A/B → a_ub/b_ub)` — GM→shared
2. Add **in shared:** `c_ub[i] = a_ub[i] + b_ub[i]` (no t1/t2/t3)
3. `T.copy(c_ub → C)`

Fragment arm stays as RF-temps baseline (`alloc_fragment` t1/t2/t3). Knob: `ST_SV1_TEMPS=frag|shared` → tags `frag` / `gmshared`. Stock **float2**, opsim 2026-10-16 HKT, cann_92b2, `ST_VF_MODE=simt-asc`.

| Case | frag VF / el/cyc | gmshared VF / el/cyc | Δ el/cyc | STS |
|------|-----------------:|---------------------:|---------:|----:|
| E8192 T128 | 3160 / **2.592** | 3160 / **2.592** | **0%** (tie) | 128 both |
| E8192 T512 | 2205 / **3.715** | 2205 / **3.715** | **0%** (tie) | 128 both |

Instr mix identical (`SIMT_LDS/STS/FADD/IADD`). See `data/SHARED_VS_FRAGMENT_AB_GM_SHARED_FLOAT2.md`.

**Verdict:** under direct GM→shared / add-in-shared, **shared matches fragment** — no shared penalty. Prefer either for this short-live stream case.

---

## 7. Meta / pins

| Pin | Value |
|-----|-------|
| SOC / opsim | Ascend950PR_9599 |
| CANN | cann_92b2 (`config.toml`: `cann_92b2/9.2.0-beta.2`) |
| TileLang tip (aligned matrix) | **e700041f** (`tilelang-pto-isa-st`) |
| PTOAS | nightly (`config.toml`: `0.67@1b36b6f…`) |
| Date | **2026-10-17 HKT** |

### Artifacts

- `data/` — **`SUMMARY_sv1_aligned_e16k_tile8k.tsv`** (+ `.json`, `NOTES_sv1_aligned_e16k_tile8k.md`; primary Simt §4.1–4.2), **`SUMMARY_sv1d_aligned_e16k_tile8k.tsv`** (+ `.json`, `NOTES_…`; primary SIMD §3 TILE@E16K), appendix `SUMMARY_sv1d_size_ladder.tsv`, `SHARED_VS_FRAGMENT_AB_GM_SHARED_FLOAT2.md` (+ `.tsv`/`.json`), legacy `SUMMARY_sv1_simt_tile16k.tsv` / `SUMMARY_sv1_simt_T_dim_rooftop.tsv` (older cann_91b3 / untiled — do not mix into §4.1–4.2), `OVERHEAD_ilp16.json`, bubbles / MIX / GPU compare
- `plots/` — **`sv1d_e16k_tile8k_ex_ipc_30cyc.png`** (SIMD §3 SoT @ E16K TILE8K), `sv1_tile_vs_vf_el_instr.png` (**aligned TILE@T512**), `sv1_thread_vs_ipc_el.png` (**aligned TILE8K×T**), **`sv1_simt_e16k_tile8k_body4_ex_ipc_30cyc.png`** (aligned Simt Body4), legacy `sv1d_e16k_ex_ipc_30cyc.png` / `sv1_simt_e16k_body4_ex_ipc_30cyc.png` (whole-E / untiled — appendix), ILP timeline / waterfall
- SoftLib CCE — `/workspace/st_out_cce_vadds_const/reports/VADDS_CONST_B32_OPSIM.md`, `VADDS_CONST_T512_1024_OPSIM.md`; schedule dig `/workspace/st_out_cce_vadds_const_sched/reports/VERDICT.md`; **PURE LD32/LD64 BW** `/workspace/st_out_cce_lds_bw_confirm/reports/LD32_LD64_BW_CONFIRM.md`
- SoftLib LDS BW — `/workspace/st_out_cce_lds_bw_confirm/reports/LD32_LD64_BW_CONFIRM.md` (+ `COMBINED_LDS_PORT_AND_MI_OFF.md`)
- Aligned opsim OUT (pto-b10) — Simt `/tmp/sv1_aligned_e16k_tile8k_20261017/` · SIMD tiled `/tmp/sv1d_aligned_e16k_tile8k_20261017/`

### Weekly gate

`scripts/regression/oneshot_sv1_e8192_t128_t512.sh` — **E=8192**, **T=128** + **T=512**, `ST_VF_MODE=simt-asc`, stock schedule only.
