# SV8 G16 Case-3 e2e RESULT (t727u steer + sv8d harvest)

**Host:** pto-b10 / Ascend950PR_9599 camodel  
**When:** 2026-10-29T11:35:00+08:00 live PASS; RESULT refresh Asia/Hong_Kong same day  
**Pack:** `/mnt/fluxdata/happybot/sv8_g16_e2e_t726u/`  
**Labels:** `reduce_axis=inner` SoftLib_axis=`1xG` `era=case3-quant-bcast` §22 → absmax expects VC/VCGMAX  
**Shape:** R=64 C=128 G=16 T=32  E=R*C=8192  
**Env (live PASS):** TILELANG=`tilelang-pto-isa-st` + **cann_92b2** (`cann-9.2.0-beta.2`); prior cann_91b3 blocked on `asc_pack_to_high` / `asc_load_l2_cache_mode`.  
**sv8d:** PTO-DSL handwritten SIMD (`sv8d_case3_bcast.py`), opsim OPPROF_20261029104314_PTJNJUBYPNEYDPUI, exitcode=0.

---

## 1. Analysis (hypothesis first)

### H1 — Prior SV1/SV5: G16 major drawback = UB BW when streaming A through UB
**Partially inherited, but not the live e2e bottleneck on this small shape.**

Outer GM↔UB: one `MOV_SRC_TO_DST_ALIGNv2` MTE2 (X 16384 B) + one MTE3 (Out 16384 B).  
Dual-VLD roof ≈ 512 B/cyc → theoretical floor ≈ 64 cyc for 32 KiB.  
Measured VF_SIMT live = **28603 cyc** ≫ 64 → **not outer-UB-BW bound**.  
Measured VF_SIMD (sv8d) live = **128790 cyc** ≫ 64 likewise.

### H2 — Fused reduce+quant-scale+eltwise keeps X in RF → SIMD may ~2× SIMT
**FALSIFIED for both arms as measured.**

**RF residency (simt-asc live, fresh ASC):**
- `float sf_inv[16]` — **stays in RF** across reduce→recip→mul (local array).
- **X does NOT stay in RF/fragment.** Both absmax and mul reload from `__ubuf__ half* buf_dyn_shmem` (shared `x_ub`):

```c
// reduce:
float vv = ((float)((__ubuf__ half*)buf_dyn_shmem)[(i*512 + threadIdx.x*16 + t)]);
// mul (same UB address space, +8192 = out_ub):
out = (half)((float)((__ubuf__ half*)buf_dyn_shmem)[...]) * s);
```

Lowered TIR matches: `T.Cast(..., x_ub[i, j])` in both Parallel loops; only `sf_inv` is `local.fragment`.

**Instr (live, call_count-weighted):** `SIMT_LDS=512`, `SIMT_STS=256`, `SIMT_FMUL=256`, `SIMT_F2F=768`, `SIMT_FMNMX=240`.  
LD vs EX: RVECLD calls≈529 vs RVECEX≈2837 → **EX/schedule-dominated inside VF**, with a full X re-load from UB for the mul.

**RF residency (sv8d PTO-DSL live):**
- **size-1 reduce** path: absmax is pairwise `RV_VMAX` (calls=16896) — not a VL-wide keep of the X tile.
- **Second VL X load for mul:** `RV_VLDS` calls=**8192** (=E) → full activation reload from UB for the scale mul; X tile is **not** held in V-reg across reduce→mul.
- **Scale expand:** scratch `VST → mem_bar(VST_VLD) → VLD` per group (`RV_VSTI`/`RV_VLDI` 640 each; `RV_SMEM_BAR` 1160 with details `type:VST_VLD` + `type:VLD_VST`).
- Scale recip uses `RV_VDIV` (512); expand uses `RV_VSEL` (1024).

**Implication for H2:** fusion keeps **scale** live (SIMT: `sf_inv` local; SIMD: after expand into V-reg for mul), but **activations** still stream UB→RF/V-reg per use on both arms. The hypothesized “X stays in register → SIMD ~2× SIMT” is **not** what either lowering does today.

### H3 — Small shape: membar/flag/cross-pipe sync may dominate
**Outer sync is real but small vs VF body on SIMT; on SIMD, membar is material (~15% VF) but not the primary tax.**

| event | simt live | sv8d live |
|-------|----------:|----------:|
| MTE2 copy X (cyc) | 1023 | (2 MOV_SRC_TO_DST_ALIGNv2, sum_cyc≈2914) |
| VF (cyc / us) | 28603 / 15.89 | **128790 / 71.55** |
| SET_FLAG+WAIT_FLAG count | 2+2 | 2+2 |
| Intra-VF `RV_SMEM_BAR` | **0** | **1160** (sum_cyc=18743, **14.6%** of VF) |

**sv8d opcode cycle shares vs VF=128790** (pipe-overlapped; shares do **not** sum to 100%):

| opcode | calls | sum_cycles | share_VF |
|--------|------:|-----------:|---------:|
| RV_VMAX | 16896 | 101376 | 78.7% |
| RV_VLDS | 8192 | 74807 | 58.1% |
| RV_VMUL | 8320 | 66560 | 51.7% |
| RV_SLDI | 6592 | 59454 | 46.2% |
| RV_SSTI | 6695 | 46282 | 35.9% |
| RV_SMEM_BAR | 1160 | 18743 | **14.6%** |
| RV_VLDI | 640 | 13952 | 10.8% |
| RV_VDIV | 512 | 8704 | 6.8% |
| RV_VSEL | 1024 | 6144 | 4.8% |
| RV_VSTI | 640 | 5760 | 4.5% |

- **membar tax:** ~14.6% of SIMD VF — real small-shape cost from scratch scale expand, **not** enough alone to explain 4.5× vs SIMT.
- **Dominant SIMD cost:** size-1 reduce (`VMAX` + scalar/scratch expand) + full X `VLDS` reload + scalar LD/ST traffic (`SLDI`/`SSTI`).
- SIMT: no intra-VF MEMBAR; outer flags wait on VF duration (pipeline), not extra sync work. Spill ASC `asc_syncthreads` between expand/consumer not re-harvested this pass.

### SIMD vs SIMT ratio / ~2×?
| arm | status | VF_cyc | el/cyc | notes |
|-----|--------|-------:|-------:|-------|
| simt-asc live | **PASS** gold | 28603 | **0.2864** | |
| simt-asc spill_dist | **PASS** gold | 29445 | 0.2782 | +842 cyc vs live |
| simd/sv8v live|spill | **FAIL** compile | NA | NA | `AscendSimdVFLowerParallel`: 2D Parallel stride=0/0 |
| **sv8d DSL live** | **PASS** gold maxabs=1.192e-07 | **128790** | **0.06361** | wall_us=72.39; size-1 reduce + scratch expand |

**Ratio (el/cyc):** SIMT÷SIMD = 0.2864 / 0.06361 ≈ **4.50×** (SIMT faster).  
**SIMD÷SIMT ≈ 0.222** — **~2× not hit; opposite direction.**

Attribution: (1) X not RF-resident on either arm; (2) sv8d size-1 reduce + scratch `VST→mem_bar→VLD` expand is far costlier than SimtVF warp schedule on G16; (3) membar ~15% VF is secondary to VMAX/VLDS/VMUL volume.

### UB BW estimate
- Outer: 32 KiB / 512 B/cyc roof → ≥64 cyc (satisfied easily on both arms).  
- Inner SIMD: VLDS 8192 calls, sum_cyc 74807 — LD volume is large **because of size-1 / reload motif**, not because outer UB roof binds.  
- **Verdict:** small-shape e2e is **EX / schedule bound** (SIMT) or **size-1-reduce + UB-reload + membar expand bound** (sv8d), **not** dual-VLD outer-UB-bound.

---

## 2. Numbers table

| path | arm | compile | gold | wall_us (msprof vec) | VF_us | VF_cyc | el/cyc (E=8192) | notes |
|------|-----|---------|------|---------------------:|------:|-------:|---------------:|-------|
| simt-asc | live | OK | PASS maxabs=0 | 17.16 | 15.89 | 28603 | 0.2864 | cann_92b2; LDS:512 STS:256 FMUL:256 |
| simt-asc | spill_dist | OK | PASS maxabs=0 | 17.62 | 16.36 | 29445 | 0.2782 | LDS:768 STS:512 LDK/STK:107 |
| simd-pto sv8v | live | FAIL | n/a | NA | NA | NA | NA | stride=0/0 LowerParallel |
| simd-pto sv8v | spill_dist | FAIL | n/a | NA | NA | NA | NA | same |
| **sv8d DSL** | **live** | **OK** | **PASS maxabs=1.192e-07** | **72.39** | **71.55** | **128790** | **0.06361** | SMEM_BAR=1160 (14.6% VF); VLDS=8192; VMAX=16896 |
| sv8d DSL | spill_dist | — | — | — | — | — | — | **honest skip** — spill sim still running / not harvested this turn |

Historical green (2026-09-23, cann_91b3+deps-native) for compare: live VF=28402 / 17.04 µs wall; spill 30545 / 18.23 µs — same RF pattern (X from UB).

**el/cyc formula:** E / VF_cyc with E=8192. Prefer VF instr row over block_delta (132818) / total_tick (134268).

---

## 3. Paths

| kind | path |
|------|------|
| SO live/spill | `out/simt/so/sv8_r64_c128_g16_t32_{live,spill_dist}.so` |
| ASC/TIR | `out/simt/sources/sv8_r64_*_{source,lowered,tir}.txt` |
| opsim simt | `out/simt/opsim_sv8_r64_*` |
| **sv8d SUMMARY** | `out/dsl/SUMMARY_sv8d_t727u.tsv` |
| **sv8d opsim live** | `out/dsl/opsim_sv8d_r64_c128_g16_t32_live/OPPROF_20261029104314_PTJNJUBYPNEYDPUI/` |
| **sv8d instr_exe** | `.../simulator/core0.veccore0/core0.veccore0_instr_exe.csv` |
| hist dumps | `hist_sv8_dumps/` (from `st_simtvf_sv5_sv6_sv8_reports.tgz`) |
| oneshot logs | `nohup_run8_simt_92b2.log`, `out/simt/oneshot_sv8_simt.log` |

---

## 4. Hypothesis verdict (one line each)
1. **UB-BW as G16 drawback:** true for streaming motifs / larger shapes; **this small e2e is EX / size-1-reduce bound**, outer UB far under roof.  
2. **X stays in RF after fuse:** **FALSE** on SimtVF (reload `x_ub`) and **FALSE** on sv8d (size-1 reduce + VLDS=E second load + scratch expand). Only scale/sf_inv RF-resident.  
3. **membar/sync dominates small shape:** **FALSE** as sole bottleneck; SIMD membar ≈**14.6%** VF (real but secondary to VMAX/VLDS); SIMT no intra-VF MEMBAR.  
4. **SIMD ~2× SIMT:** **FALSIFIED** on this sv8d motif — SIMT÷SIMD ≈ **4.50×** (SIMD slower); need a VL-resident X + non-scratch expand SIMD path before re-testing ~2×.
