# tilelangGym ST report format

Standard spine for ST ANALYSIS.md (SV*/CF*/SP*). Match SV1 section order. Confirm Aims with the author before rewriting a report. Never invent measured numbers.

## 1. Aims
What this ST proves or measures (author-confirmed).

## 2. Test-case design
Kernels, E/TILE/T (or VL/G), datatypes, pass criteria, case matrix.

## 3. Layout and schedule (SIMT and SIMD)
How work maps to UB/MRF, thread/VL, load/store widths, and intended pipeline stages for both arms.

## 4. SIMD PTO-DSL performance
- Measured el/cyc
- Theory-bound / rooftop from the same layout + instruction mapping used in the run
- Sensitivity (tile size, virtual VL / G, other named knobs)
- Instruction mix
- Running IPC (windowed) and input/output element per cycle
- Overall and mid-avg rooftop %

## 5. SIMT (Ascend default) lowering performance
- Stock Ascend / SimtVF path
- Layout mapping and instruction mix (LDS/STS 32 vs 64)
- Theory-bound from that layout + mapping
- Sensitivity, mainly thread dim (T)

## 6. SIMT hand-craft (SoftLib) analysis
- Handwritten CCE / SoftLib
- Best layout that is not LD/ST-bound when possible
- Prefer LDS.32 / STS.32 when that is the SoT width
- Stage-split LD / compute / ST; find best stage schedule
- Update theory rooftop to the same layout mapping + instructions
- IPC timing plots with stage analysis

## 7. Conclusion
SIMD best, SIMT Ascend default, SIMT SoftLib hand-craft; summary table and performance ratios (SIMD ÷ SIMT arms).

## Conventions
- Prefer TILE8192 naming when TILE=E SoT; keep multi-TILE ladders for sensitivity.
- el/cyc = elements / VF cycles (state which VF).
- Mid-avg = interior window (document W); overall = whole-kernel.
- Rooftop % vs the binding resource named in that section.
- Embed running-IPC / stage PNGs inline when present under the ST folder.

Primary path: benches/<st>/ANALYSIS.md. Supporting notes OK beside it; ANALYSIS must stand alone with this spine.
