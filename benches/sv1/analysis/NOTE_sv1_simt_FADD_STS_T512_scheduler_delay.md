# T512 FADD→STS ~55c — scheduler delay, not longer RAW

**Verdict:** T512 FADD→STS ~53c (p50≈mean) is REAL with last-FADD pairing on same (sch,warp). Warp spans balanced (max/min~1.08). STS waitBitMask=0 → not longer RAW; scheduler interleaves other warps before returning to issue STS. LDS→FADD stays ~2–3c (true RAW). Pairing different elements (FIFO across many FADDs) is wrong here — use last-FADD→next-STS; 2× FADD vs STS count is pack geometry.

## Numbers (same sch,warp · last-FADD→next-STS)

| | T128 | T512 |
|--|-----:|-----:|
| warps | 4 | 16 |
| warp-span max/min | 1.002 | 1.082 |
| LDS→FADD mean / p50 | 2.50 / 1.80 | 2.83 / 3.60 |
| FADD→STS mean / p50 | 2.88 / 3.60 | 53.48 / 52.20 |
| per-warp FADD→STS p50 (median across warps) | 3.60 | 51.75 |
| STS % waitBitMask≠0 | 0.0 | 0.0 |

## Why not imbalance / bad pairing?

- **Balance:** equal FADD/warp; span max/min ≈ 1.0–1.08 → no big skew.
- **p50≈mean** on FADD→STS → not a few straggler warps.
- **STS wait=0** → scoreboard clear when STS issues; the ~53c is time the warp was not selected (other warps filling the sch).
- **LDS→FADD** stays ~2–3c (true RAW) at both T.
- FIFO pairing across all FADDs before STSs is nonsense here (2× FADD vs STS from 2R+1W geometry) — do not use it.

Plot: `plots/sv1_simt_warp_balance_FADD_STS_T128_vs_T512.png`
