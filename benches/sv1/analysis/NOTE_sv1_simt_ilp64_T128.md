# SimtVF T=128 ILP reschedule (.64 only) vs untiled T=2048

**Date:** 2026-10-15 HKT · **SOC:** Ascend950PR_9599 opsim · pack **LDS/STS type:5 (.64)** — **no .128**  
**Roof (Lok):** **1R+1W @ 128B each → 16 el/cyc** (not 32).

## Goal

Follow GPU ILP pattern (many independent loads → FADD → stores) but keep Ascend SimtVF **float2 / `.64` only**. Use CCE-visible ASC rewrite + `#pragma unroll` at **T=128**, check if it beats untiled **T=2048**.

## Verdict: **YES — ILP16 @ T128 beats T2048**

| Case | VF_cyc | **el/cyc** | **% of 16** | Body4 IPC | mid med | mid peak | FADD wait% | vs T2048 |
|------|-------:|-----------:|------------:|----------:|--------:|---------:|-----------:|----------|
| T128 baseline | 5700 | **2.874** | 18.0% | 0.374 | 0.400 | 0.500 | 50.0 | −25% |
| T2048 untiled | 4279 | **3.829** | 23.9% | 0.486 | 0.467 | 1.500 | 50.0 | baseline |
| ILP4 @ T128 | 4643 | **3.529** | 22.1% | 0.483 | 0.500 | 1.133 | 50.0 | −7.8% |
| ILP8 @ T128 | 4271 | **3.836** | 24.0% | 0.517 | 0.433 | 1.467 | **25.0** | ≈tie |
| **ILP16 @ T128** | **4015** | **4.081** | **25.5%** | **0.544** | 0.467 | **1.867** | **13.3** | **+6.6%** |

- Pack stays **`.64` only** (`LDS type:5`, `STS type:5`, **type:6 = 0**).
- FADD `waitBitMask≠0` drops **50% → 13.3%** at ILP16.
- Still only **~¼ of roof 16**; mid-median Body4 IPC ~0.47 (≪ mix-cap 4).

## Overhead breakdown — why only 4.08? (keep-trace)

**Keep:** `/workspace/sv1_sv_analysis/data/ilp_keeps/sv1t_e16384_t128_ilp16/`  
**Script:** `scripts/analyze_ilp16_overhead.py` · **JSON:** `data/OVERHEAD_ilp16.json`

### Cycle pie (VF = 4015)

| Piece | Cycles | % VF | Instr |
|-------|-------:|-----:|-------|
| **Prologue** (VF → 1st Body4) | **30.6** | **0.8%** | SIMT_S2R×2, SHFI, LEA |
| **Body** (1st Body4 → last complete) | **3712.7** | **92.5%** | LDS+STS+IADD_I+FADD |
| **Epilogue** (last Body4 → VF end) | **271.7** | **6.8%** | SET_FLAG, MOV_SRC_TO_DST |
| **Host kernel launch** | — | — | **Not in keep / not in 4.08** |

### Rates vs roof 16

| Denominator | el/cyc | % of 16 |
|-------------|-------:|--------:|
| VF SUMMARY | **4.081** | **25.5%** |
| Body complete span | 4.413 | 27.6% |
| **Mid avg (ΣFADD×32 / mid-span)** | **4.507** | **28.2%** |
| FADD first→last complete span | 5.378 | 33.6% |

### Waterfall (roof → measured)

```
Roof 16.0  ──(mid-body stall, Δ≈11.5)──►  Mid ~4.51  ──(pro+epi ~0.4)──►  VF 4.08
                 ▲ 65% mid cycles idle                         ▲ launch NOT here
                 Body4 mid-med IPC 0.47≪4
```

| Question | Answer |
|----------|--------|
| Kernel launch? | **No** — VF keep excludes host launch; 4.08 is already VF-only. |
| Prologue/epilogue? | **Minor** — 7.5% of VF (~0.4 el/cyc drag). Epilogue 272c is the larger of the two. |
| Head/tail schedule? | **Minor** — head ~2.93, mid ~4.51, tail ~5.46 el/cyc (FADD credit); mid already ≈ overall. |
| Mid peak sustained? | **No** — Body4 mid-peak 1.87 / FADD burst peak ~40 el/cyc in 30-cyc windows; **only ~3%** of mid windows ≥85% of Body4 peak; FADD-rate **median = 0** in 30–120-cyc windows (ILP batches). **Sustained mid ≈ 4.5 el/cyc (~28% of 16).** |
| Key overhead? | **Steady mid-body under-issue** (idle 65%, IADD tax, residual RAW) — not launch/head-tail. |

Plot: `plots/sv1_ilp16_overhead_waterfall.png`.

### Timeline (why 4.x)

Plot: `plots/sv1_ilp16_T128_timeline_ipc_el.png` (+ `.pdf`).

- Top: 30-cyc moving **Body4 IPC** (mid med **0.467**, peak **1.867**).
- Bottom: **el/cyc = FADD×32 / 30-cyc win** (+ Body4×share×32 proxy).
- Hlines: **roof 16**, **mid sustained ΣFADD×32/mid-span ≈ 4.51**, **VF E/VF_cyc ≈ 4.08**.
- Win FADD median often **0** (ILP batches); sustained mid uses count/span.
- Series/meta: `data/sv1_ilp16_T128_timeline_ipc_el_{series.csv,meta.json}`.


## Method

1. TileLang `sv1t_e16384_t128` fused loop → ASC rewrite to ILP{4,8,16} float2 regs.
2. `bisheng -O2 --npu-arch=dav-3510` → opsim keep staging.
3. Body4 = LDS+STS+IADD(+_I)+FADD; mid = 15% trim; 30-cyc sliding IPC.

## Instr mix (ILP16 rows)

LDS=512, STS=256, FADD=512, IADD_I=740 · all LDS/STS **type:5 (.64)**. FADD share of Body4 rows ≈ **25.3%**.

ILP does **not** cut IADD much; win is **issue order / dep depth** (FADD wait% ↓).

## Reading

1. Explicit software ILP at **fixed T=128** beats piling warps to T=2048 on el/cyc.
2. **4.08 ≈ 25% of roof 16** is almost entirely **mid-body rate (~4.5)**; VF wrapper tax is small.
3. Next levers: cut IADD (imm offsets), more ILP / better schedule, eventually **`.128`** (separate).



## GPU float4/.128 reschedule el/cyc (same E=16K, T=128)

See `NOTE_sv1_gpu_vs_ascend_ilp16_elcyc.md`. Headline (ncu `sm__cycles_active.max`):

| Case | Pack | el/cyc | vs Ascend ILP16 |
|------|------|-------:|----------------:|
| **Ascend ILP16 @ T128** | .64 | **4.081** (25.5% of roof **16**) | — |
| GPU TileLang TILE2048@T128 | float4 CUDA | **1.035** | 0.25× |
| GPU hand ILP4 **1CTA** .128 | **.128** | **0.877** | 0.21× |
| GPU hand unroll multi-CTA .128 | **.128** | **2.151** | 0.53× |

Fairest same-shape (1×T=128): Ascend ILP16 **~4.7×** GPU hand .128 1CTA and **~3.9×** TL GPU — different roofs/ISAs (camodel VF vs GB10); Ascend still only ¼ of its UB roof 16.

## Artifacts

```text
/workspace/sv1_sv_analysis/data/
  NOTE_sv1_simt_ilp64_T128.md
  OVERHEAD_ilp16.json
  ilp_keeps/sv1t_e16384_t128_ilp16/{trace.json,instr_exe.csv}
  SUMMARY_sv1_ilp64.{tsv,json}
/workspace/sv1_sv_analysis/plots/sv1_ilp16_overhead_waterfall.png
/tmp/sv1_ilp64/ on pto-b10 (sources, so, flat, reports)
```
