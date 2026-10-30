# t582u — Why SoftLib wins at E=4096 (from PR9 running-IPC)

**Dig:** `/workspace/sv8_e4k_simt_win_ipc_t582u/`  
**Plots (PR9 `opcode_lanes_ipc`):** `plots/*_e4096_running_ipc_pr9.png`  
**No crown / no merge.** Dump re-capture only (instr_log) — ranking unchanged vs t579u SHOW.

## SHOW numbers (unchanged)

| arm | E=4096 el/cyc | VF | mid-med IPC (PR9) | peak IPC |
|---|---:|---:|---:|---:|
| **SoftLib fused+noeps** | **7.543** | 543 | **2.65** | 3.2 |
| SIMD keeplive | 7.420 | 552 | 0.95 | 1.2 |
| SIMD pack_ub | 7.038 | 582 | 1.05 | 2.05 |

Cite E=8192 (t548u/t576u PR9): SoftLib mid≈3.4 · pack≈1.1 · keep≈0.9 — SoftLib still denser mid, but pack **wins el/cyc** once N_SCALE==VL (ops/N collapse).

---

## Lok hypothesis — confirm / refute

> SIMD best needs pack-in-UB → ramp-up membar + ramp-down IPC drawdown; SoftLib only wastes per-warp (1/32 SIMT vs 1/64 SIMD).

### A. Pack SMEM_BAR / mid-valley — **CONFIRM (pack-specific)**

PR9 pack lanes show **SMEM_BAR** in the overlay; running IPC has a deep **valley ~ticks 200–280** (IPC → ~0.2) aligned with pack’s compact-store → `VST_VLD` bar → batch VDIV → expand. Extra pad prefill bar at E=4k (`N_SCALE=128 ≪ VL=256`, **50% pad**). That is real and hurts pack vs SoftLib.

### B. “SIMD best = pack → therefore SoftLib wins on membar” — **REFUTE as sole cause**

SoftLib also beats **keeplive** (7.54 > 7.42), and keeplive PR9 has **no SMEM_BAR lane**. Keep’s mid-med IPC is **0.95** — same low density class as pack’s mid plateau (~1.05), without pack’s barrier valley. So SoftLib’s 4k win is **not** “because the only competitive SIMD path pays membar.”

### C. SoftLib win driver at 4k — **IPC density / dual_drain overlap**

SoftLib PR9: long plateau mid-med **2.65** (~2.5× SIMD) with **LDS ∥ REDUX ∥ FDIV_I ∥ STS ∥ FMUL** overlapped (dual_drain fused Abs). SIMD keep/pack mid ~1.0 — sparse vector issue stream. SoftLib’s short ramp-up (~0–180) and steep tail (~420–500) exist, but the **mid body** carries the el/cyc.

### D. “1/32 SIMT vs 1/64 SIMD wastage” — **REFUTE / mis-scale**

At E=4096 SoftLib T=512 fills **16 warps × 32** cleanly (rounds=8); there is no half-warp pad story. Pack’s pad is **N_SCALE/VL = 128/256 = 50%**, not 1/64. Keep has no pad. SoftLib’s edge is **higher sustained IPC**, not a finer 1/32 vs 1/64 wastage grain.

---

## One-line verdict

**Partially confirm pack’s SMEM_BAR mid-valley + 50% pad; refute as the SoftLib-win cause** — SoftLib also beats no-bar keeplive via **~2.65 mid IPC** dual_drain overlap vs SIMD ~1.0. At E=8k pack’s ops/N win overcomes its low mid-IPC; at E=4k it does not.
