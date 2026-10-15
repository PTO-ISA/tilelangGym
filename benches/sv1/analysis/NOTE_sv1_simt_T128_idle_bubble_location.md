# T128 idle max=244 — is it misleading?

**Tag:** `sv1t_e16384_t128` (untiled E=16K, TILE=E=16384, pack .64)  
**Keep:** `/workspace/sv1_sv_analysis/data/sv1t_e16384_t128/`  
**Date:** 2026-10-15 HKT

## Verdict

**Yes — 244 is misleading as “typical idle.”** It is **one pathological late-body DRAIN hole**, not VF prologue and not mid-body steady state.

| Metric | Value |
|--------|------:|
| idle bubbles n | 1429 |
| mean / p50 / p90 / p99 | **2.61** / **1** / **7** / **8** |
| **global max** | **244** |
| **mid ±5% max** (fully inside) | **8** |
| **mid ±10% max** | **8** |

So the distribution Lok expects (~few cycles) is right; the 244 is a single outlier.

## Where does the 244 sit?

- Body = 1st LDS → last STS issue: cyc **[40, 5412]** span=5373
- **Max idle:** cyc **[4934, 5177]** len=**244**
- Start at **body 91.1%** → **late drain**, not head/prologue
- VF prologue before 1st LDS is only **40** cyc; head (first 5%) max idle = **8**
- After the 244 starts, only **~4.2% of Body4** remain (LDS/FADD/STS drain of last warps)
- Next-longest are also drain: 33 @ 96.9%, 29 @ 95.7%, 19 @ 99.3% — all last ~10%

Just before: `SIMT_LDS` sch3/warp3 @ 4933.  
Just after (5178): `SIMT_IADD_I` on sch0+sch1 — scheduler resumes addr setup for remaining work.

## Why T512 “max 32” looked like a huge win

T512’s max idle 32 is also a **tail** effect (more warps keep the drain denser). Mid-body max was already ~8 at T128 — T↑ mainly kills this **one long drain hole**, which is why mean idle only moves 2.61→2.46 while max drops 244→32.

## Plot

`plots/sv1_simt_T128_idle_bubble_location.png` — hist (≤30) + bubble length vs body %.

## Method

Idle bubble = consecutive body cycles with Body4 issue-width 0 (same definition as `NOTE_sv1_simt_bubbles_T128_vs_T512.md`). Mid = exclude first/last 5% or 10% of body span.
