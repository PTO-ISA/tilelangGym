# VMI outer Nx1 pure-VMAX absmax — t720u

**Who:** Lok t720u · executor  
**When:** 2026-10-29 HKT (Asia/Hong_Kong)  
**Env:** pto-b10 · cann_92b2 · ptoas-vmi 0.1.3 · Ascend950PR_9599 · `era=fp32-absmax`  
**reduce_axis:** **outer** (Harness §22) — elementwise VMAX fold only  
**Live pack:** `/mnt/fluxdata/happybot/vmi_absmax_outer_vmax_t720u/`  
**TSV:** box `/workspace/RESULT_outer_vmax_t720u.tsv` · b10 `$PACK/RESULT_outer_vmax_t720u.tsv`

---

## 1. Gold / KPI

| Field | Definition |
|-------|------------|
| **Gold** | View `A[E]` as `(G, n_cols=E/G)` row-major; `C[c]=max_r \|A[r,c]\|` |
| **E** | 8192 fp32 → fp32 absmax (**no recip**) |
| **G** | 16×1 / 32×1 / 64×1 / 128×1 |
| **KPI** | `el/cyc = E / named_VF_cyc` (`vf_real_execute_time`) |
| **Opcode** | **RV_VMAX** fold only — **forbidden** VCGMAX/VCMAX as reduce |

**Not harvested:** glm53 clean `o*_vmax_*` (Case-3 `vmaxs+vdiv` recip) — epilogue ≠ `era=fp32-absmax` (delta checklist).

---

## 2. SoT table (best per G)

| G (Nx1) | VMI outer VMAX el/cyc | tag | VF_cyc | abs-path | VL | gold ok | SEMANTICS_GATE |
|--:|--:|--|--:|--|--:|--|--|
| 16 | **46.28** | `outer_g16_vmax_vand_fold_vl256` | 177 | i32-VAND | 256 | PASS (mad=0) | **PASS** — S3 VCGMAX=0 VCMAX=0 VDIV=0; dump RV_VMAX=120 RV_VAND=128 |
| 32 | **45.77** | `outer_g32_vmax_vand_fold_vl256` | 179 | i32-VAND | 256 | PASS (mad=0) | **PASS** — dump RV_VMAX=124 RV_VAND=128; VCG/VCMAX/VDIV=0 |
| 64 | **45.51** | `outer_g64_vmax_vand_fold_vl128` | 180 | i32-VAND | 128 | PASS (mad=0) | **PASS** — dump RV_VMAX=126 RV_VAND=128; VCG/VCMAX/VDIV=0 |
| 128 | **45.26** | `outer_g128_vmax_vand_dual4_vl64` | 181 | i32-VAND | 64 | PASS (mad=0) | **PASS** — dump RV_VMAX=127 RV_VAND=128; VCG/VCMAX/VDIV=0 |

**Verdict: DONE** — required outer explore covered; SoT = max el/cyc among PASS rows; pure VMAX confirmed in OPPROF dump.

---

## 3. 4-arm cite (outer Nx1)

VMI = this turn. SoftLib SIMT = t708 (cite only). SoftLib CCE not remeasured (optional).

| G (Nx1) | VMI outer VMAX (t720) | SoftLib CCE (t708 cite) | SoftLib SIMT stock (t708) | SoftLib SIMT best (t708) |
|--:|--:|--:|--:|--:|
| 16 | **46.28** | 31.39 | 9.68 | 11.69 |
| 32 | **45.77** | 21.28 | 4.74 | 9.58 |
| 64 | **45.51** | 30.68 | 3.46 | 6.08 |
| 128 | **45.26** | 32.38 | 2.15 | 2.19 |

vs polluted t708 VMI column (inner Gmap reuse 33.44/22.14/30.68/32.25): **outer VMAX supersedes** those cells under §22.

---

## 4. Tried inventory (19 measured, all gold PASS)

| G | tag | mode | abs | VL | VF | el/cyc | note |
|--:|--|--|--|--:|--:|--:|--|
| 16 | `…_vand_fold_vl256` | fold | VAND | 256 | 177 | **46.28** | **SoT** |
| 16 | `…_vand_dual4_vl256` | dual4 | VAND | 256 | 178 | 46.02 | ~tied |
| 16 | `…_vand_split2_vl256` | split2 | VAND | 256 | 177 | 46.28 | tied SoT |
| 16 | `…_vabs_fold_vl256` | fold | VABS_FP | 256 | 200 | 40.96 | VAND wins |
| 16 | `…_vand_fold_vl128` | fold | VAND | 128 | 177 | 46.28 | retile peer = SoT |
| 32 | `…_vand_fold_vl256` | fold | VAND | 256 | 179 | **45.77** | **SoT** |
| 32 | `…_vand_dual4_vl256` | dual4 | VAND | 256 | 179 | 45.77 | tied |
| 32 | `…_vand_split2_vl256` | split2 | VAND | 256 | 179 | 45.77 | tied |
| 32 | `…_vabs_fold_vl256` | fold | VABS_FP | 256 | 202 | 40.55 | VAND wins |
| 32 | `…_vand_fold_vl128` | fold | VAND | 128 | 179 | 45.77 | retile peer |
| 64 | `…_vand_fold_vl128` | fold | VAND | 128 | 180 | **45.51** | **SoT** |
| 64 | `…_vand_dual4_vl128` | dual4 | VAND | 128 | 180 | 45.51 | tied |
| 64 | `…_vand_split2_vl128` | split2 | VAND | 128 | 181 | 45.26 | −0.5% |
| 64 | `…_vabs_fold_vl128` | fold | VABS_FP | 128 | 241 | 33.99 | VAND wins |
| 64 | `…_vand_fold_vl64` | fold | VAND | 64 | 264 | 31.03 | VL retile worse |
| 128 | `…_vand_fold_vl64` | fold | VAND | 64 | 306 | 26.77 | serial weak |
| 128 | `…_vand_dual4_vl64` | dual4 | VAND | 64 | 181 | **45.26** | **SoT** (ILP) |
| 128 | `…_vand_split2_vl64` | split2 | VAND | 64 | 265 | 30.91 | mid |
| 128 | `…_vabs_fold_vl64` | fold | VABS_FP | 64 | 309 | 26.51 | VAND fold peer |

**Stop:** G16/32/64 fold≈dual4≈split2 within noise — no further ILP. G128 dual4 required win kept. VABS peer measured everywhere; i32-VAND SoT.

---

## 5. OPPROF confirm (no VCGMAX/VCMAX)

SoT dumps (`core0.veccore0.instr_log.dump`):

```
outer_g16_vmax_vand_fold_vl256 : RV_VMAX=120 RV_VAND=128 RV_VCGMAX=0 RV_VCMAX=0 RV_VDIV=0  vf_real=177
outer_g32_vmax_vand_fold_vl256 : RV_VMAX=124 RV_VAND=128 RV_VCGMAX=0 RV_VCMAX=0 RV_VDIV=0  vf_real=179
outer_g64_vmax_vand_fold_vl128 : RV_VMAX=126 RV_VAND=128 RV_VCGMAX=0 RV_VCMAX=0 RV_VDIV=0  vf_real=180
outer_g128_vmax_vand_dual4_vl64: RV_VMAX=127 RV_VAND=128 RV_VCGMAX=0 RV_VCMAX=0 RV_VDIV=0  vf_real=181
```

---

## 6. Paths

| Where | Path |
|-------|------|
| pto-b10 pack | `/mnt/fluxdata/happybot/vmi_absmax_outer_vmax_t720u/` |
| box report | `/workspace/VMI_ABSMAX_OUTER_VMAX_t720u.md` |
| box TSV | `/workspace/RESULT_outer_vmax_t720u.tsv` |
| glm53 clean (untouched) | tmux `glm53-sv567-clean-t717u` — Case-3 outer peer only |
