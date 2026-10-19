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
| SIMT threads | **128 / 256 / 512 / 1024 / 2048** @ fixed **E=16K, TILE=8192** (not untiled) |
| VMI / handwritten SIMD DSL | **VL=64**, **same E=16K + TILE 2K/4K/8K/16K** (per-tile MTE stream) |

**Footprint note (8K tile):** one f32 array @ 8K = **32 KB**; A+B = **64 KB**; A+B+C = **96 KB** — can fit UB / approach whole MRF (128 KB). After the tile ladder we **focus on 8K** as the compare hub vs SIMD.

This set is the full sensitivity for the UB-bound case: **tile size**, **thread dim**, **Simt vs SIMD**, and whether current `T.Parallel` / DSL already meet **xx%** of the Simt / SIMD rooftops (sections below).

**Pin note:** aligned Stock Simt + tiled SIMD numbers below are from **2026-10-17** opsim under **cann_92b2 + tip e700041f** (Simt) / **cann_92b2** (SIMD DSL). Prior whole-E SIMD size ladder (`SUMMARY_sv1d_size_ladder.tsv`) and older cann_91b3 Simt tables are **appendix-only** — do not mix into §4 / §5.1–5.2 primary tables.


## 3. Best schedule

**Target** for this short-live stream (load → add → store; UB-bound) is path-specific:

| Path | Target best schedule |
|------|----------------------|
| **SIMD / VMI DSL** (`sv1d` / `sv1dt`, VL=64) | Keep the kernel as a **simple load → add → store** loop on the data. No fancy instruction schedule is required — VL=64 UB vector ld/st + VADD is the design target. |
| **SIMT** (`T.Parallel` / SimtVF) | Best schedule does **not** hinge on fragment vs shared temps for this case (direct GM→shared ties fragment — see later shared-vs-fragment section). What matters is a **relatively large thread dim (T=256 / T=512)** for **latency hiding** on UB ld/st, because the kernel is **UB load-bound**. |

Later SIMD / SIMT sections measure how close stock + hand paths land relative to these targets (SIMD near theory rooftop with the simple VL loop; Simt stock peaking around T512 on the TILE=8K hub).

## 4. SIMD performance

Handwritten DSL **VL=64**, **tiled** short-live stream @ fixed **E=16384** (`data/SUMMARY_sv1d_aligned_e16k_tile8k.tsv`):

| TILE | VF_cyc | IPC | el/cyc | us |
|-----:|-------:|----:|-------:|---:|
| 2048 | 824 | 1.2913 | **19.8835** | 7.69 |
| 4096 | 644 | 1.6211 | **25.4410** | 5.87 |
| **8192** | **560** | **1.8464** | **29.2571** | **4.99** |
| 16384 | 520 | 1.9788 | **31.5077** | 4.55 |

Kernel: `sv1dt_*` / `scripts/gen_sv1dt_kernel.py` — per tile MTE load A/B → VL=64 vload/vadd/vstore → MTE store C. Tag `sv1dt_e16384_tile{TILE}_vl64`.

**Theory rooftop: ~44 el/cyc.** Plot SoT on `plots/sv1d_e16k_tile8k_ex_ipc_30cyc.png` is a **single VF** 30-cyc window (first body VF; dual-tile ping/pong gap omitted). Derived roof **42.67** (IPC→el conversion); cite **theory rooftop ~44** as the design target and **42.67** as the plot’s derived roof.

From that same plot footer (authoritative mid-body IPC @ **E=16K TILE=8K**):

| Metric | Value | vs roof |
|--------|------:|---------|
| Mid-body median IPC | **2.17** → **~34.1 el/cyc** | **≈80%** of 42.67 · **≈78%** of theory rooftop (~44) |
| Peak excl fill/drain IPC | 2.77 | — |
| Peak mid IPC | 2.63 | — |
| Early fill IPC | 3.27 | — |
| Overall VF (TILE8K) | **29.26 el/cyc** | **≈66%** of theory rooftop (~44) · **≈69%** of 42.67 |

![SIMD E16K TILE8K IPC](plots/sv1d_e16k_tile8k_ex_ipc_30cyc.png)

**Takeaway (SIMD alone):** at **TILE=8K E=16K**, SIMD DSL lands **29.26 el/cyc** overall VF (~66% of theory rooftop ~44); mid-body ~**34.1** el/cyc (~78% of ~44). Larger TILE amortizes per-tile MTE fill/drain (2K→16K: 19.9→31.5 el/cyc). Prior whole-E E-ladder is appendix-only — not the §4 compare SoT. Head-to-head vs Simt is after §5.



## 5. SIMT performance

Stock pack under tip e700041f is wider-vector (LDS:FADD:STS ≈ **256:512:128** in the aligned harvest — float4-class). Lead framing still uses Lok’s **LD.64 A+B ≈ 8 el/cyc** roof for % columns (same A+B dual-load story).

**Lead framing (Lok):** LD.64 A+B theory ≈ **8 el/cyc**. Aligned stock Simt lands **~2.40–2.45 el/cyc** → about **~30%** of that ~8 roof under **cann_92b2 + e700041f**. Thread ladder **peaks @ T512** then drops @ T2048 (§5.2). Alternate 1R+1W@16 is a footnote in §5.3.

Data: `data/SUMMARY_sv1_aligned_e16k_tile8k.tsv` (+ `.json`, `NOTES_sv1_aligned_e16k_tile8k.md`).

### 5.1 Tile-size performance (E=16384, fixed T=512)

| Case | VF_cyc | el/cyc | % of LD.64 A+B (~8) |
|------|-------:|-------:|--------------------:|
| TILE2048 @ T512 | 6721 | **2.44** | 30% |
| TILE4096 @ T512 | 6692 | **2.45** | 31% |
| **TILE8192 @ T512** | 6825 | **2.40** | **30%** |
| TILE16384 @ T512 | 6721 | **2.44** | 30% |

![Tile vs VF el](plots/sv1_tile_vs_vf_el_instr.png)

**Point:** at fixed T=512, tile size is **flat (~2.40–2.45 el/cyc)**. **8K** remains the focus tile for Simt↔SIMD compare (footprint / hub), not because it uniquely wins the ladder.

### 5.2 Thread-dim performance (E=16384, TILE=8192)

| Case | VF_cyc | el/cyc | IPC_body | % of LD.64 A+B (~8) |
|------|-------:|-------:|---------:|--------------------:|
| TILE8192 @ T128 | 8221 | **1.99** | 0.506 | 25% |
| TILE8192 @ T256 | 7041 | **2.33** | 0.600 | 29% |
| **TILE8192 @ T512** | **6825** | **2.40** | **0.638** | **30%** |
| TILE8192 @ T1024 | 6857 | **2.39** | 0.616 | 30% |
| TILE8192 @ T2048 | 11270 | **1.45** | 0.540 | 18% |

![Thread vs IPC/el](plots/sv1_thread_vs_ipc_el.png)

![Simt Body4 IPC TILE8K](plots/sv1_simt_e16k_tile8k_body4_ex_ipc_30cyc.png)

**Point:** el/cyc is **not monotonic**. It rises 128→256→**512 (peak 2.40)**, stays flat at **T1024 (2.39)**, then **drops hard at T2048 (1.45)**. Turning point ≈ **T512** on this TILE=8K stock schedule. Body4 plot remains the **TILE8K@T512** capture (IPC_body **0.638**).


**Forward (not run):** later try **T128/T256 fully unroll** and let **ccec** schedule, then compare vs hand/manual — do not treat as SoT until measured.


### 5.3 Footnote — alternate roof only (do not mix into §5.1–5.2)

Prior pack docs also used **1R+1W @ 128 B each → 16 el/cyc**. Against that roof: aligned TILE8K@T512 ≈ **15%**. Lead framing for Simt stock stays **~8**; keep **16** only here.

### 5.4 Side note: LD.32

- Theory: LD.32 can do **128 B/cyc**, but **too many instructions / hard to schedule**.
- For LD.32 A+B (dual load on one R port), **theory best ~8 el/cyc**.
- SoftLib / best-thread cases land around **~4–5 el/cyc** VF average under realistic schedules:
  - SoftLib **A-only** LDS.b32 contig T128: VF **4.54** el/cyc @ E=8192 (label **A-only** — not A+B).
  - Aligned stock Simt (A+B): **~2.40–2.45** el/cyc under new pins.
  - SoftLib A-only T512 / T1024 VF **8.5 / 9.9** el/cyc are **A-only** — do **not** mix with A+B theory without labeling.
- SoftLib A-only LDS.b32 contig T128: LDS-window **101 B/cyc (79% of 128)**, LDS-window elems **25.3**; whole-VF elems **4.54** — see `/workspace/st_out_cce_vadds_const/reports/`.
- SoftLib A+B LDS.b32 contig T128: **~52 B/cyc** (~41% of 128) from LDS width BW report.

### 5.5 SoftLib LDS.b32 / LDS.b64 BW — PURE port isolation (2026-10-17)

**SoT:** `/workspace/st_out_cce_lds_bw_confirm/reports/LD32_LD64_BW_CONFIRM.md` (pto-b10, Ascend950PR_9599, E=8192 T=128).

Prior SoftLib T128 ~**101 B/cyc** (section 5.5 / §6 baseline) was **compiler IADD↔LDS interleave** (`I1L1…`) — **not** LDS-port isolation. Confirmed PURE numbers below require:

1. **Self-scheduled (自己排好):** hand/manual schedule with explicit **IADD-then-LDS** separation (`sep_ptr` — all address IADD/LEA first, hard phase barrier, then LDS burst alone). Disasm must show LDS region with **zero** IADD/LEA (`separation_ok=True`, motif `L64` / `L32`).
2. **MI schedule disabled** (Lok: 「MR schedule disabled」 → SoftLib/SV1 flag): `-mllvm -enable-misched=false -mllvm -enable-post-misched=false`. Default MI keeps MIXED interleave even with `sep_ptr`.

| Width | Tag | Disasm | Peak B/cyc | Sustained first→last B/cyc | vs 128 roof |
|-------|-----|--------|----------:|---------------------------:|------------:|
| **LDS.b32 (LD32)** | `bw_b32_sep_ptr_e8192_t128_mi_off` | **PURE** `L64` | **128.00** | **~115** (114.98) | **100%** peak · **~90%** sustained |
| **LDS.b64 (LD64)** | `bw_b64_sep_ptr_e8192_t128_mi_off` | **PURE** `L32` | (burst†) | **~64.5** (64.50) | **~50%** sustained |

† LD64 short-window rolling peak can exceed 128 (1× LDS.b64/cyc × 256 B); **claim metric is sustained first→last = 64.50**.

**One-liner:** under PURE / MI-off / self-scheduled IADD→LDS: **LD32 peak 128 B/cyc, sustained ~115; LD64 ~64.5 B/cyc (~half)**. Isolation costs whole-VF (LD32 PURE whole-VF **7.19** el/cyc vs T1024 **9.89**) — do not mix PURE port-BW with MIXED whole-VF chase (§6).

### 5.6 Head-to-head vs SIMD (E=16K, matched TILE ≤8K)

| TILE | SIMD el/cyc | Simt T512 el/cyc (§5.1) | ratio |
|-----:|------------:|------------------------:|------:|
| 2048 | 19.88 | 2.44 | ~8× |
| 4096 | 25.44 | 2.45 | ~10× |
| **8192** | **29.26** | **2.40** | **~12×** |

**Takeaway:** at the TILE=8K hub, SIMD is ~**12×** Simt T512 (**29.26** vs **2.40**). Simt tile ladder is already **flat** (§5.1) — growing TILE does **not** close the gap. The gap is still a **scheduling bubble with long-latency LDS** on the stock Simt path (UB ld-bound; latency hiding helps with T but does not remove the bubble), not a “need bigger TILE” story.


---

## 6. SoftLib hand schedule — why Simt is hard (1-in VADDS+const)

Stock Simt on this UB ld-bound case leaves a **scheduling bubble with long-latency LDS** (§5.6 H2H takeaway: TILE8K hub SIMD **29.26** vs Simt T512 **2.40**, ~**12×**). SoftLib hand schedules below are the **proof of that claim** and a **hand ceiling** — not a sudden side note. We simplify the kernel so the schedule difficulty is visible in isolation, then show what a working hand motif can recover (and what fails).

### 6.1 Claim + simplify

**Claim:** the stock Simt gap is schedule / LDS-latency, not “need a bigger TILE.”

**Simplify for the probe:** **VADDS with a const** = **one input**, **four instructions** per element stream:

| Stage | Op |
|-------|-----|
| addr | `IADD` |
| load | `LDS` |
| compute | `FADD` + const |
| store | `STS` |

A-only rooftop framing **~16** el/cyc. Dual-input A+B (same motif, §6.4) uses roof **~8** (LD.64 A+B). Reports live under `/workspace/st_out_cce_vadds_const*/reports/` and `/workspace/st_out_cce_lds_bw_confirm/reports/` (pto-b10 mirrors under `/tmp/st_out_cce_*_20261017`).

### 6.2 Stage schedule — working motif vs SWP

**Working hand motif (measured):** **phase-separated IADD-then-LDS** plus **`loadall_barrier` / `iadd_first`**. Addr+load go first as a packed phase; FADD/STS tail is hidden across warps (T256/T512/T1024), not stuffed into LDS gaps. Full phase pack (all IADD → all LDS → …) is the same idea.

**SWP / Body4 packing was tried and lost.** A 4-walk software pipeline that packs IADD / FADD / STS into LD gaps is **not** the winning schedule on this machine for the A-only probe: SWP @ T128 landed ~**3.4** el/cyc (LDS median gap ≈**27** → needs ~**14** outstanding) — **falsified** vs the phase-sep + barrier motif. See SoftLib `SW_PIPELINE_AONLY_4x128.md` (under `/workspace/st_out_cce_vadds_const_sched/reports/`).

Lok’s earlier “SWP or full-unroll matched stage latency → ~9.x/16” intuition for the **ceiling** was right on the ballpark for the **winning** motif; the implementation that actually hits ~9–10 el/cyc is **phase-sep + barrier / warp hide**, not Body4 SWP.

### 6.3 Measured 1-in A-only (SoftLib LDS.b32+const)

SoftLib CCE A-only LDS.b32+const, E=8192 (reports under `/workspace/st_out_cce_vadds_const_sched/reports/` + `/workspace/st_out_cce_vadds_const/reports/`; T256/T512: `T256_8WARP_RESCHED.md`, `T512_16WARP_RESCHED.md`):

| Probe | VF | whole-VF el/cyc | LDS-win B/cyc | vs roof |
|-------|---:|----------------:|--------------:|---------|
| Baseline load-all (`-O2` MI on) T128 | 1806 | **4.54** | **101.14** (79% of 128) | LDS-win **25.3** el/cyc |
| **Manual best T128** (`loadall_barrier` / `iadd_first`) | **882** | **9.29** | **101.45** | **~58%** of A-only **16**; **~94%** of T1024 **9.89** |
| **Manual best T256** (`loadall_barrier` / `iadd_first`) | **759** | **10.79** | **94.71** | **~67.5%** of **16** — SoftLib A-only ceiling |
| Manual best T512 (`loadall_barrier` / `iadd_first`) | 796 | **10.29** | 55.07 | **~64%** of **16** |
| T512 / T1024 baseline SoftLib style | 964 / **828** | **8.5 / 9.89** | LDS-win B/cyc **falls** as warps↑ | T1024 is a useful whole-VF roof reference |
| MI-off (`-mllvm -enable-misched=false`) T128 | 2381 | **3.44** | 48.91 | **worse** than `-O2` |
| `-O0` T128 | 3380 | 2.42 | 44.34 | worse |

| Probe | Result |
|-------|--------|
| All-four IADD+LDS+FADD+STS same cycle | **0** cycles at T128/512/1024 |
| Latency alone forbid ~4-issue? | **No** — schedule/port limited |
| Can T128 schedule alone beat T1024 9.89? | **No** — best **9.29**; win is **more warps (T256 10.79)**, not MI order alone |
| Body4 / 4-walk SWP @ T128 | **~3.4** el/cyc — **falsified** (§6.2) |

### 6.4 Dual-input A+B hand schedule (LDS.b64)

Same **phase-sep + `loadall_barrier` / `iadd_first`** motif, now **two sources** (A+B), **LDS.b64** (float2). Roof **~8** el/cyc (Lok A+B / LD.64 framing).

SoT: `/workspace/st_out_cce_lds_bw_confirm/reports/LDS64_AB_TWO_SOURCE.md` (pto-b10 `/tmp/st_out_cce_lds64_ab_20261017`).

| T | best kind | whole-VF el/cyc | % of roof 8 |
|--:|-----------|----------------:|------------:|
| 128 | `loadall_barrier` / `iadd_first` | **3.62** | **45%** |
| **256** | `loadall_barrier` / `iadd_first` | **4.82** | **60%** |
| 512 | `baseline_loadall` | **4.67** | **58%** |

- **Best:** T256 `ab_b64_loadall_barrier_e8192_t256` / `ab_b64_iadd_first_e8192_t256` → **4.82** el/cyc (**60%** of 8).
- LDS-win ~**49–55** B/cyc across the ladder.
- Beats prior ILP16 A+B **4.08** el/cyc.
- Lok expected ~**5–6** vs ~**8** — measured **4.82** is that ballpark (slightly under the midpoint).

A+B needs **roughly 2×** the load traffic of 1-in, so absolute el/cyc sits under the A-only ceiling; utilization vs its own roof (**60%** of 8) is in the same story as A-only (**67.5%** of 16) once the hand motif is applied.

### 6.5 Single-input LDS.b32 probe — yes, this *is* the SoftLib ceiling

Lok asked whether the same 1-in VADDS+const case with **LDS.b32** (roughly **2× IADD + 2× LDS** vs packing as LDS.b64) can actually be best.

**Answer: yes.** The SoftLib A-only hand ceiling we already quote **is** LDS.b32 + const with the phase-sep / barrier motif (§6.3). Whole-VF bests:

| T | el/cyc | % of roof 16 | report |
|--:|-------:|-------------:|--------|
| 128 | **9.29** | **58%** | `VERDICT.md` |
| **256** | **10.79** | **67.5%** | `T256_8WARP_RESCHED.md` |
| 512 | **10.29** | **64%** | `T512_16WARP_RESCHED.md` |
| 1024 | **9.89** | **62%** | SoftLib T1024 reference |

So “LDS.32 twice the IADD/LDS — is whole-VF actually best?” → **measured SoftLib ceiling = LDS.b32 A-only @ T256 `loadall_barrier` = 10.79 (~67.5% of 16)**. Wider T helps hide the FADD/STS tail up to T256, then turns down (T512 10.29, T1024 9.89).

### 6.6 Verdict (this SV1 case)

| Path | Absolute el/cyc | Roof | Utilization |
|------|----------------:|------|------------:|
| **1. Current SIMT lowering** (stock TileLang, aligned best hub) | **2.40** (TILE8192 @ T512; thread peak — T1024 2.39 / T2048 1.45) | LD.64 A+B **~8** | **~30%** |
| **2. SoftLib / hand reschedule best** | A-only LDS.b32+const **10.79** @ T256 `loadall_barrier` (T512 10.29 · T1024 9.89 · T128 hand 9.29); A+B LDS.b64 **4.82** @ T256 (T512 4.67 · T128 3.62) | A-only **~16** / A+B **~8** | **~67.5%** of 16 · **~60%** of 8 |
| **3. SIMD best** (handwritten DSL VL=64 @ E=16K) | **31.51** (TILE16384; TILE8K hub **29.26**) | theory rooftop **~44** | **~72%** of 44 (TILE8K **~66%**) |

**Bottom line:** stock Simt TILE8K peaks at **T512 = 2.40 el/cyc (~30% of ~8)** and turns down by T2048 — gap is a **scheduling bubble + long-latency LDS** (§5.6), not TILE size. SoftLib proves the claim: simplify to 1-in VADDS+const, apply **phase-sep IADD-then-LDS + barrier** (not Body4 SWP), and A-only reaches **10.79 (~67.5% of 16)**; same motif on A+B LDS.b64 reaches **4.82 (~60% of 8)**. SIMD tiled best remains **31.51** / hub **29.26** (~**12×** stock Simt at TILE8K).


## 7. Shared vs fragment — direct GM→shared only (no S-MEM staging tower)

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

## 8. Meta / pins

| Pin | Value |
|-----|-------|
| SOC / opsim | Ascend950PR_9599 |
| CANN | cann_92b2 (`config.toml`: `cann_92b2/9.2.0-beta.2`) |
| TileLang tip (aligned matrix) | **e700041f** (`tilelang-pto-isa-st`) |
| PTOAS | nightly (`config.toml`: `0.67@1b36b6f…`) |
| Date | **2026-10-17 HKT** |

### Artifacts

- `data/` — **`SUMMARY_sv1_aligned_e16k_tile8k.tsv`** (+ `.json`, `NOTES_sv1_aligned_e16k_tile8k.md`; primary Simt §5.1–5.2), **`SUMMARY_sv1d_aligned_e16k_tile8k.tsv`** (+ `.json`, `NOTES_…`; primary SIMD §4 TILE@E16K), appendix `SUMMARY_sv1d_size_ladder.tsv`, `SHARED_VS_FRAGMENT_AB_GM_SHARED_FLOAT2.md` (+ `.tsv`/`.json`), legacy `SUMMARY_sv1_simt_tile16k.tsv` / `SUMMARY_sv1_simt_T_dim_rooftop.tsv` (older cann_91b3 / untiled — do not mix into §5.1–5.2), `OVERHEAD_ilp16.json`, bubbles / MIX / GPU compare
- `plots/` — **`sv1d_e16k_tile8k_ex_ipc_30cyc.png`** (SIMD §4 SoT @ E16K TILE8K, **single VF**), `sv1_tile_vs_vf_el_instr.png` (**aligned TILE@T512**), `sv1_thread_vs_ipc_el.png` (**aligned TILE8K×T**), **`sv1_simt_e16k_tile8k_body4_ex_ipc_30cyc.png`** (aligned Simt Body4), legacy `sv1d_e16k_ex_ipc_30cyc.png` / `sv1_simt_e16k_body4_ex_ipc_30cyc.png` (whole-E / untiled — appendix), ILP timeline / waterfall
- SoftLib CCE — `/workspace/st_out_cce_vadds_const/reports/VADDS_CONST_B32_OPSIM.md`, `VADDS_CONST_T512_1024_OPSIM.md`; schedule dig `/workspace/st_out_cce_vadds_const_sched/reports/VERDICT.md`; **PURE LD32/LD64 BW** `/workspace/st_out_cce_lds_bw_confirm/reports/LD32_LD64_BW_CONFIRM.md`
- SoftLib LDS BW — `/workspace/st_out_cce_lds_bw_confirm/reports/LD32_LD64_BW_CONFIRM.md` (+ `COMBINED_LDS_PORT_AND_MI_OFF.md`)
- Aligned opsim OUT (pto-b10) — Simt `/tmp/sv1_aligned_e16k_tile8k_20261017/` · SIMD tiled `/tmp/sv1d_aligned_e16k_tile8k_20261017/`

### Weekly gate

`scripts/regression/oneshot_sv1_e8192_t128_t512.sh` — **E=8192**, **T=128** + **T=512**, `ST_VF_MODE=simt-asc`, stock schedule only.
