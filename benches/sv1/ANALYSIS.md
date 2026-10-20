# SV1 analysis — short-live stream eltwise (Simt + SIMD)

**Date:** 2026-10-20 HKT · SoftLib fence×source / rooftop rewrite **2026-10-20** · float2 §5 fold **2026-10-20** · **SOC:** Ascend950PR_9599 opsim · **CANN / PTOAS:** cann_92b2 / PTOAS nightly (`config.toml` pins) · **Primary stock Simt:** untiled float2 E=8192 (`max_vector_bits=64`, LDS.64-class) · **Aligned e700041f E16K TILE matrix:** appendix-only

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
| **Primary stock Simt** | **E = 8192** f32, **untiled**, stock float2 (`max_vector_bits=64`), frag ≡ gmshared (direct GM→shared) |
| SIMT threads (primary) | **64 / 128 / 256 / 512 / 1024 / 2048** @ E=8192 untiled |
| SoftLib handwritten | E=8192 stage-sep / fence×source (LDS.32 vs LDS.64) — §6 |
| SIMD / VMI DSL (fair H2H hub) | **VL=64**, **E=16K + TILE 2K/4K/8K/16K** (per-tile MTE stream) — §4 |
| Aligned Simt (appendix) | E=16K TILE 2K–16K × T @ tip e700041f float4-class — historic fair-matrix; **not** §5 primary |

**Footprint note (E=8192 / SoftLib loadall):** one f32 array = **32 KB**; A+B = **64 KB**; A+B+C = **96 KB** — **75%** of MRF (128 KB). SIMD §4 still uses **TILE=8K @ E=16K** as the compare hub (same 96 KB A+B+C tile).

**Pin note:** primary stock Simt numbers (§5 / §8) are **2026-10-20** opsim under **cann_92b2**, untiled float2 E8192. Aligned Stock Simt E16K TILE @ tip **e700041f** (2026-10-17) is **appendix-only** — do not mix into §5.1–5.2 primary tables. SIMD §4 stays the tiled E16K VL=64 SoT.

## 3. Best schedule

**Target** for this short-live stream (load → add → store; UB-bound) is path-specific:

| Path | Target best schedule |
|------|----------------------|
| **SIMD / VMI DSL** (`sv1d` / `sv1dt`, VL=64) | Keep the kernel as a **simple load → add → store** loop on the data. No fancy instruction schedule is required — VL=64 UB vector ld/st + VADD is the design target. |
| **SIMT** (`T.Parallel` / SimtVF) | Best schedule does **not** hinge on fragment vs shared temps for this case (direct GM→shared ties fragment — see §8). What matters is a **large thread dim** for **latency hiding** on UB ld/st (kernel is **UB load-bound**). Primary stock float2 peaks at **T=1024** (see §5.2). |

Later SIMD / SIMT sections measure how close stock + hand paths land relative to these targets (SIMD near theory rooftop with the simple VL loop; Simt stock float2 peaking at **T1024 = 3.846** el/cyc on E8192 untiled).

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

**Primary SoT — stock untiled float2** (E=8192, `max_vector_bits=64`, LDS:FADD:STS = **256:256:128**, frag ≡ gmshared). Opsim **2026-10-20**, cann_92b2, `ST_VF_MODE=simt-asc`. Data: `data/SHARED_VS_FRAGMENT_AB_GM_SHARED_FLOAT2.md` (+ `.tsv` / `.json`).

**Lead framing (Lok):** LD.64 A+B theory ≈ **8 el/cyc**. Primary stock lands **T512 = 3.715** (~**46%** of ~8) and peaks at **T1024 = 3.846** (~**48%** of ~8). T2048 = 3.787 (above T512, below T1024).

Aligned e700041f E16K TILE ladder (~2.40 @ TILE8K T512, float4-class LDS:FADD:STS ≈ 256:512:128) is **appendix-only** — different pack / config; do not cite as current stock.

### 5.1 T512 hub (E=8192 untiled float2)

No multi-TILE float2 ladder was re-measured. Primary hub is the **untiled T512** point only:

| Case | VF_cyc | el/cyc | % of LD.64 A+B (~8) | wall µs |
|------|-------:|-------:|--------------------:|--------:|
| **untiled T512** | **2205** | **3.715** | **46%** | **1.23** |

**Point:** stock @ **T512 = 3.715** el/cyc under E8192 untiled float2 (frag ≡ gmshared). Prefer this hub row for "default stock @ T512" citations. Aligned TILE@T512 flat ladder (~2.40–2.45) is appendix — not primary.

Tile plot `plots/sv1_tile_vs_vf_el_instr.png` — **obsolete for §5.1** (aligned E16K TILE@T512 only; no float2 multi-TILE). Kept on disk under appendix artifacts; not embedded here.

### 5.2 Thread-dim performance (E=8192 untiled float2)

| Case | VF_cyc | el/cyc | % of LD.64 A+B (~8) | wall µs |
|------|-------:|-------:|--------------------:|--------:|
| untiled T64 | 4497 | 1.822 | 23% | 2.50 |
| untiled T128 | 3160 | 2.592 | 32% | 1.76 |
| untiled T256 | 2389 | 3.429 | 43% | 1.33 |
| untiled T512 | 2205 | 3.715 | 46% | 1.23 |
| **untiled T1024** | **2130** | **3.846** | **48%** | **1.18** |
| untiled T2048 | 2163 | 3.787 | 47% | 1.20 |

![Thread vs el/cyc](plots/sv1_thread_vs_ipc_el.png)

**Point:** el/cyc **rises through T1024**; peak is **T1024 = 3.846**. T2048 = **3.787** (still above T512 3.715, below T1024). Turning point ≈ **T1024** on this untiled float2 schedule.

**1R+1W footnote:** prior pack docs also used **1R+1W @ 128 B each → 16 el/cyc**. Against that roof: T1024 ≈ **24%**, T512 ≈ **23%**. Lead framing for Simt stock stays **~8**; keep **16** only as this footnote.

**Forward (not run):** later try **T128/T256 fully unroll** and let **ccec** schedule, then compare vs hand/manual — do not treat as SoT until measured. Multi-TILE float2 @ fixed T512 not yet measured.

## 6. SoftLib handwritten Simt rooftop analysis

**Date:** SoftLib rooftop rewrite **2026-10-20**. Stock Simt primary is **E8192 untiled float2**: hub **T512 = 3.715** (~46% of LD.64 A+B ~8) / peak **T1024 = 3.846** (~48%) from §5. SoftLib below is the handwritten Simt rooftop story: width (LDS.32 vs LDS.64), schedule (1-in first), then dual-input %rooftop + full fence×source table.

### 6.1 LDS.32 vs LDS.64

PURE / MI-off / self-scheduled port isolation (IADD→LDS separated; disasm LDS region with zero IADD/LEA). SoT: `/workspace/st_out_cce_lds_bw_confirm/reports/LD32_LD64_BW_CONFIRM.md`.

| Width | Tag | Peak B/cyc | Sustained first→last B/cyc | vs 128 |
|-------|-----|----------:|---------------------------:|-------:|
| **LDS.b32** | `bw_b32_sep_ptr_e8192_t128_mi_off` | **128** | **~115** (114.98) | **~90%** sustained |
| **LDS.b64** | `bw_b64_sep_ptr_e8192_t128_mi_off` | (burst) | **~64.5** (64.50) | **~half** |

**Implication:** Simt el/cyc rooftop **depends on width**. For stream eltwise A+B:

| Framing | Roof el/cyc | Notes |
|---------|------------:|-------|
| Classic lead (LD.64 A+B) | **~8** | Stock / §5.1–5.2 % columns |
| SoftLib LDS.b32 **1-in** serial | **16** | LDS 32 + STS 32 el/cyc ports |
| SoftLib LDS.b32 **A+B** serialized harmonic | **10.67** | `1/(1/16+1/32)` — dual-load LDS 16 + STS 32 |

Default stock TileLang uses wider / LDS.64-class lowering. SoftLib hand targets **LDS.b32** for the high ceiling.

### 6.2 Schedule — single-source (1-in VADDS+const) first

One input keeps the case simple: IADD → LDS → FADD(+const) → STS. SoftLib A-only LDS.b32+const, E=8192. Reports under `/workspace/st_out_cce_vadds_const*/reports/` and T256 fence harvest `T256_FENCE_X_SOURCE_SUMMARY.md`.

**1. MI-off stage-sep ≈ 8.x baseline.** All LDS then all FADD then all STS (`loadall_barrier` / `iadd_first`; simple same-latency stages + warp hide; MI off):

| role | MI | VF | el/cyc | tag / note |
|------|----|---:|-------:|---|
| **Controllable baseline** | OFF | **942** | **8.70** | stage-sep under hand control |
| SoftLib ceiling | ON | **759** | **10.79** | same source; MI compresses (−183 VF / +2.10 el) |

**2. Split LD and ST stages to reach ~10.x.** Compute can fuse either way. MI-on stage-sep + **one** compiler `asm volatile("" ::: "memory")` fence (LDS-bar *or* STS-bar) → **10.79** @ T256. Second fence unused (`fence1 ≡ store-only ≡ barrier` on VF). **Nofence** collapses to **6.87** — schedule/overlap, not mem latency (LDS/STS issue→retire latency fence-invariant: LDS 28.516/12.878, STS 21.496/20.737).

**3. Performance rooftop (split / fenced case).** Serialized walls: **%ser ≈ 72.6%** of roof **16** (ser 11.6199). Stage util for 1-in MI-on: LDS-win ~**74%** / post ~**76%**. Hand MI-off mix attempts do **not** beat 8.70:

| probe | el/cyc | vs MI-off 8.70 |
|-------|-------:|----------------|
| best F↔S mix | **8.83** (+0.13) | marginal |
| idx[] IL / bad IL | **~6.06** | loses |
| F/S bleed reimpl | **~5.6** | loses |

The ~**+2** el/cyc is **MI’s I↔L**, not C source-order mix.

**4. Threads and SWP.** Need **≥256 threads** for LDS/STS outstanding — T128 (4 warps) cannot hide LDS latency ≈27. SWP Body4 @ T128 falsified ~**3.4** el/cyc (interleaves FADD/STS into LDS window → LDS-win collapses).

**Compact T ladder — 1-in barrier best (MI-on `loadall_barrier` / `iadd_first`):**

| T | warps | VF | el/cyc | % of roof 16 |
|--:|------:|---:|-------:|-------------:|
| 128 | 4 | 882 | **9.29** | 58% |
| **256** | **8** | **759** | **10.79** | **67.5%** |
| 512 | 16 | 796 | **10.29** | 64% |

Best whole-VF: **T256 = 10.79**. Wider T helps hide FADD/STS up to T256, then turns down.

**§6.2 conclusion.** To approach the SoftLib ceiling we must **split the schedule into LDS and STS stages** (compute fused either way), using **LDS.32 + STS.32**. Rooftops are the **serial model**:

| Case | Serial roof | Derivation |
|------|------------:|------------|
| **1-in / 1-out** | **16** el/cyc | 128 B/c R + 128 B/c W |
| **Dual-input A+B** | **10.667** el/cyc | `1/(1/16+1/32)` |

MI-off stage-sep alone stays ~**8.x**; split LD/ST + MI-on reaches ~**10.x** (~**72.6%** of serial **16**).

### 6.3 Dual-input — same %rooftop story + full fence×source table

Bring A+B back: performance rooftop **% similar** to 1-in. MI-on stage-sep + **one** fence reaches **≈72–73%** of the serialized rooftop (1-in **72.6%** of 16; A+B **72.2%** of 10.67). Full report: SoftLib opsim harvest `T256_FENCE_X_SOURCE_SUMMARY.md`.

**Roofs:** %s1/%s2 vs LDS **32** / STS **32** el/cyc. **%ser** = `E/(s1_wall+s2_wall)` ÷ roof (**16** / **10.67**). **nofence:** s1/s2/%ser = **N/A** — raw `[first_I,last_S]` body matches fenced but **VF ≫ body**; use **(E/VF)/roof** only for %whole.

| source | fence | VF | whole (E/VF) | %whole | s1 el/cyc | %s1 | s2 el/cyc | %s2 | ser el/cyc | %ser | LDS mean/var | STS mean/var |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| 1-in | nofence | 1192 | 6.8725 | **43.0%**† | N/A | N/A | N/A | N/A | N/A | N/A | 28.516/12.878 | 21.496/20.737 |
| 1-in | LDS-bar | 759 | 10.7931 | — | 27.4899 | 85.9% | 20.1278 | 62.9% | 11.6199 | **72.6%** | 28.516/12.878 | 21.496/20.737 |
| 1-in | STS-bar | 759 | 10.7931 | — | 27.4899 | 85.9% | 20.1278 | 62.9% | 11.6199 | **72.6%** | 28.516/12.878 | 21.496/20.737 |
| A+B | nofence | 1853 | 4.4209 | **41.4%**† | N/A | N/A | N/A | N/A | N/A | N/A | 31.443/32.470 | 22.504/36.165 |
| A+B | LDS-bar | 1119 | 7.3208 | — | 22.9468 | 71.7% | 11.6034 | 36.3% | 7.7065 | **72.2%** | 31.926/26.969 | 21.543/24.249 |
| A+B | STS-bar | 1119 | 7.3208 | — | 22.9468 | 71.7% | 11.6034 | 36.3% | 7.7065 | **72.2%** | 31.926/26.969 | 21.543/24.249 |

† nofence **%whole** = `(E/VF)/roof`. Fenced **%ser** = serialized walls. Latency issue→retire fence-invariant on 1-in; A+B fence1≡store-only. VF collapse is schedule/overlap, not mem latency.

**Classic VF framing vs serial rooftop (not fixed-8):** SoftLib **LDS.b32** A+B %rooftop uses serial **10.667**. Do **not** fix SoftLib b32 % to **8** — that **~8** is the **LD.64 A+B** classic lead from §5.

| Case | el/cyc | Serial roof | % of serial |
|------|-------:|------------:|------------:|
| SoftLib A+B LDS.b32 T512 `loadall_barrier` | **7.42** | **10.667** | **~69.5%** |
| SoftLib A+B LDS.b32 T256 fence1 | **7.32** | **10.667** | **~68.6%** |
| SoftLib A+B LDS.b64 T256 `loadall_barrier` (width contrast) | **4.82** | LD.64 A+B **~8** | **~60%** of LD.64 roof |

Fence-table **%ser** for A+B remains **72.2%** of **10.67**. SoT A+B b32: `/workspace/st_out_cce_lds_bw_confirm/reports/LDS32_AB_TWO_SOURCE.md`. A+B b64: `LDS64_AB_TWO_SOURCE.md`.

---

## 7. Head-to-head highlight

**Answer first.**

- **SIMD** simple VL=64 schedule @ TILE8K: **29.26** el/cyc (~**66%** of theory rooftop ~44; mid-body ~34 / ~**78%**). A simple load→add→store does **28+**.
- **Simt rooftop** depends on **LDS.32 vs LDS.64**. Need **≥256 threads** for LDS/STS outstanding, and **split LD/ST stages** (LDS.32 + STS.32) for instruction schedule to approach the SoftLib ceiling. Serial roofs: 1-in/1-out **16** (128 B/c R + 128 B/c W); A+B **10.667**.
- **Default lowering** (stock untiled float2 LDS.64-class A+B, E8192): hub **T512 = 3.715** (~**46%** of LD.64 A+B ~8) → **~7.9×** vs SIMD 29.26; peak **T1024 = 3.846** (~**48%**) → **~7.6×** vs SIMD 29.26.
- **Handcraft best so far** (SoftLib A+B LDS.b32 T512 `loadall_barrier`): **7.42** → **~69.5%** of serial **10.667** (fence **%ser ~72%** of 10.67) → remaining gap **~3.9×** vs SIMD 29.26, which **matches the input UB BW difference** (256R+256R vs 128R).

**MRF = 128 KB.** SoftLib `loadall` keeps the live tile in RF (f32 E=8192: one array **32 KB**; A+B **64 KB**; A+B+C **96 KB** = **75%** of MRF). Stock Simt primary and SoftLib share the **E=8192 A+B+C = 96 KB** footprint; SIMD §4 TILE8K hub uses the same **96 KB** tile size at E16K.

**Fair H2H @ schedule/mapping (A+B arms vs SIMD TILE8K hub):**

| Mapping / schedule | el/cyc | Roof | Rooftop % | vs SIMD 29.26 | MRF live (KB / %) |
|--------------------|-------:|------|----------:|--------------:|------------------:|
| Stock Simt LDS.64 — E8192 untiled float2 **T512** (§5.1) | **3.715** | LD.64 A+B **~8** | **~46%** | **~7.9×** | A+B+C **96** / **75%** |
| Stock Simt LDS.64 — E8192 untiled float2 **T1024 peak** (§5.2) | **3.846** | LD.64 A+B **~8** | **~48%** | **~7.6×** | A+B+C **96** / **75%** |
| SoftLib A+B LDS.b64 — T256 `loadall_barrier` | **4.82** | LD.64 A+B **~8** | **~60%** | **~6.1×** | A+B+C **96** / **75%** |
| SoftLib A+B LDS.b32 — T512 `loadall_barrier` | **7.42** | serial **10.667** | **~69.5%** | **~3.9×** | A+B+C **96** / **75%** |
| SIMD VL=64 — TILE8K hub (§4) | **29.26** | theory **~44** | **~66%** | **1×** | A+B+C **96** / **75%** |

SoftLib phase-sep A+B closes most of the Simt schedule gap vs stock; remaining SoftLib↔SIMD gap tracks UB BW (256R+256R vs 128R), not "need a bigger TILE." All fair A+B arms sit at **~75% MRF** for the 8K/E=8192 A+B+C tile — SoftLib's win is schedule/width, not extra register budget.

## 8. Shared vs fragment — direct GM→shared only (no S-MEM staging tower)

**Scope:** §8 is the same layout A/B that feeds **§5 primary** — **E=8192 untiled float2**, frag ≡ gmshared. §5.1–5.2 use this ladder as the stock Simt SoT.

**S-MEM tower removed.** The old shared arm used `t1/t2/t3 = alloc_shared` staging (shared→shared copies) and looked **−37%/−50%** vs fragment (`data/SHARED_VS_FRAGMENT_AB.md`: T512 frag **2.406** / shared-tower **1.198**) — that pattern is **gone**. No shared→fragment `T.copy` either. **Do not cite tower −37%/−50% as the current finding.**

**Kept path (SDS / direct GM→shared):**
1. `T.copy(A/B → a_ub/b_ub)` — GM→shared
2. Add **in shared:** `c_ub[i] = a_ub[i] + b_ub[i]` (no t1/t2/t3)
3. `T.copy(c_ub → C)`

Fragment arm stays as RF-temps baseline. Knob: `ST_SV1_TEMPS=frag|shared` → tags `frag` / `gmshared`. Stock **float2**, opsim **2026-10-20** (T64–T2048), cann_92b2, `ST_VF_MODE=simt-asc`.

| Case | frag VF / el/cyc | gmshared VF / el/cyc | Δ el/cyc | STS |
|------|-----------------:|---------------------:|---------:|----:|
| E8192 T64 | 4497 / **1.822** | — | — | 128 |
| E8192 T128 | 3160 / **2.592** | 3160 / **2.592** | **0%** (tie) | 128 both |
| E8192 T256 | 2389 / **3.429** | 2389 / **3.429** | **0%** (tie) | 128 both |
| E8192 T512 | 2205 / **3.715** | 2205 / **3.715** | **0%** (tie) | 128 both |
| E8192 **T1024** | **2130 / 3.846** | **2130 / 3.846** | **0%** (tie) | 128 both |
| E8192 T2048 | 2163 / **3.787** | 2163 / **3.787** | **0%** (tie) | 128 both |

Instr mix identical (`SIMT_LDS/STS/FADD/IADD` = 256/128/256/384). Untiled float2 **peaks @ T1024 ≈ 3.846** (~**48%** of LD.64 A+B roof ~8). Data: `data/SHARED_VS_FRAGMENT_AB_GM_SHARED_FLOAT2.md`.

**Verdict:** under direct GM→shared / add-in-shared, **shared matches fragment** — no shared penalty. Prefer either for this short-live stream case. Tower −50% is staging-only history; do not cite as SDS.

## 9. Meta / pins

| Pin | Value |
|-----|-------|
| SOC / opsim | Ascend950PR_9599 |
| CANN | cann_92b2 (`config.toml`: `cann_92b2/9.2.0-beta.2`) |
| Primary stock Simt | **E8192 untiled float2** (frag ≡ gmshared), opsim **2026-10-20** |
| TileLang tip (aligned appendix matrix) | **e700041f** (`tilelang-pto-isa-st`) — appendix-only for Simt |
| PTOAS | nightly (`config.toml`: `0.67@1b36b6f…`) |
| Date | **2026-10-20 HKT** |

### Artifacts

- `data/` — **`SHARED_VS_FRAGMENT_AB_GM_SHARED_FLOAT2.md`** (+ `.tsv`/`.json`; **primary Simt §5.1–5.2 / §8**), **`SUMMARY_sv1d_aligned_e16k_tile8k.tsv`** (+ `.json`, `NOTES_…`; primary SIMD §4 TILE@E16K), SoftLib fence/BW reports under `/workspace/st_out_cce_*`, appendix `SUMMARY_sv1_aligned_e16k_tile8k.tsv` (+ `NOTES_sv1_aligned_e16k_tile8k.md`; **aligned E16K TILE — not §5 primary**), appendix `SUMMARY_sv1d_size_ladder.tsv`, legacy tower `SHARED_VS_FRAGMENT_AB.md` (staging-only), legacy `SUMMARY_sv1_simt_tile16k.tsv` / `SUMMARY_sv1_simt_T_dim_rooftop.tsv`
- `plots/` — **`sv1_thread_vs_ipc_el.png`** (**E8192 float2 T-dim** primary §5.2), **`sv1d_e16k_tile8k_ex_ipc_30cyc.png`** (SIMD §4 SoT), appendix/obsolete for primary Simt: `sv1_tile_vs_vf_el_instr.png` (aligned TILE@T512), `sv1_simt_e16k_tile8k_body4_ex_ipc_30cyc.png` (aligned Body4), legacy whole-E plots / ILP timeline
- SoftLib CCE — `/workspace/st_out_cce_vadds_const/reports/VADDS_CONST_B32_OPSIM.md`, `VADDS_CONST_T512_1024_OPSIM.md`; schedule dig `/workspace/st_out_cce_vadds_const_sched/reports/VERDICT.md`; **PURE LD32/LD64 BW** `/workspace/st_out_cce_lds_bw_confirm/reports/LD32_LD64_BW_CONFIRM.md`
- SoftLib LDS BW — `/workspace/st_out_cce_lds_bw_confirm/reports/LD32_LD64_BW_CONFIRM.md` (+ `COMBINED_LDS_PORT_AND_MI_OFF.md`)
- Opsim OUT (pto-b10) — float2 T-dim `/tmp/sv1_float2_tdim_E8192_20261020` · appendix aligned Simt `/tmp/sv1_aligned_e16k_tile8k_20261017/` · SIMD tiled `/tmp/sv1d_aligned_e16k_tile8k_20261017/`

### Weekly gate

`scripts/regression/oneshot_sv1_e8192_t128_t512.sh` — **E=8192**, **T=128** + **T=512**, `ST_VF_MODE=simt-asc`, stock schedule only. (Does not yet cover **T1024** peak — extend when gating peak SoT.)

### Appendix — aligned e700041f E16K TILE (not primary)

Historic fair-matrix under tip **e700041f** (float4-class, 2026-10-17): TILE@T512 flat **~2.40–2.45** el/cyc; TILE8K×T peaks **T512 = 2.40**, flat T1024 **2.39**, drop T2048 **1.45**. Plots: `sv1_tile_vs_vf_el_instr.png`, `sv1_simt_e16k_tile8k_body4_ex_ipc_30cyc.png`. See `data/NOTES_sv1_aligned_e16k_tile8k.md` / `SUMMARY_sv1_aligned_e16k_tile8k.tsv`. **Do not** use these as current stock Simt SoT — primary is §5 float2.
