# REGEN_NOTES t823u — SoftLib LDS.32 best dump → SV1 PR9

**When:** 2026-11-02 HKT (Asia/Hong_Kong)
**Why:** Lok t823u — t822u residual：PR9 之前只用咗 stock SimtVF frag + SIMD twin；SoftLib LDS.32 best dump 未入圖；staged caption 未對齊 SoftLib best。Lok 問點解唔而家 redraw + update PR — 今次兩個一齊做。

## SoftLib dump（SoT）
- KPI：**7.42** el/cyc（ANALYSIS §7 · harvest `elems_per_cyc_vf` · `reports/LDS32_AB_TWO_SOURCE.md`）
- Pack：`/tmp/st_out_cce_lds32_ab_20261019/`
- Arm：`opsim_ab_b32_loadall_barrier_e8192_t512`
- Dump：**`.../OPPROF_20261019162717_PYCDEDBJDNAKDSMF/dump/core0.veccore0.instr_popped_log.dump`**（299379 B；core0.veccore0 **非空**，唔使 fallback 其他 core）

## SIMD twin（保持 t813u/t822u dump）
`/mnt/fluxdata/happybot/glm53_sv1567_sf_mr2_t813u/out/sv1/opsim_sv1dt_e8192_tile8192_vl64/OPPROF_20261031164454_RFRDZRUKZRTQKIED/dump/core0.veccore0.instr_popped_log.dump`

## Plan A（採用）
- **保留** stock/era-2 `sv1_simtvf_running_ipc_pr9.png` (+staged) — 仍然係 t813u frag dump
- **新增** SoftLib-named PR9：
  - `plots/sv1_softlib_lds32_ab_t512_loadall_barrier_running_ipc_pr9.png` (+ `.json`)
  - `plots/sv1_softlib_lds32_ab_t512_loadall_barrier_running_ipc_pr9_staged.png` (+ `.json`)
- ANALYSIS §11 SoftLib **best** 改指 SoftLib dump + SoftLib PNG（t821u prove-supersede）；唔 silent overwrite simtvf 檔名

## Regen cmd
```bash
PY=/home/happybot/projects/venv-ptoas-vmi-0.1.3/bin/python
PYTHONPATH=. $PY scripts/make_pr9_running_ipc_overlay.py \
  --dump <SOFTLIB_DUMP> \
  --out-png benches/sv1/plots/sv1_softlib_lds32_ab_t512_loadall_barrier_running_ipc_pr9.png \
  --title "SV1 / SoftLib A+B LDS.b32 T512 loadall_barrier E8192 — running IPC (W=20) · all ISA groups" \
  --E 8192 --frontend softlib_simt \
  --staged-out benches/sv1/plots/sv1_softlib_lds32_ab_t512_loadall_barrier_running_ipc_pr9_staged.png
# SIMD caption refresh（同一 dump）→ sv1_simdt_running_ipc_pr9{,_staged}.png
```

## Gate 結果
| arm | plot_style | lane_mode | lane_coverage | lanes |
|-----|------------|-----------|--------------:|-------|
| SoftLib LDS.32 best | opcode_lanes_ipc | all_isa_groups | **1.0** | S2R, IADD_I, LEA, SHFI, **LDS**, FADD, STS, END |
| SIMD twin | opcode_lanes_ipc | all_isa_groups | **0.9865** | VLD, VADD, VST |
| stock SimtVF (kept) | opcode_lanes_ipc | all_isa_groups | **1.0** | S2R, IADD_I, LEA, SHFI, LDS, FADD, STS, END, LOP3, MOV |

- SoftLib lanes **有 LDS** + address-math（IADD_I/LEA/SHFI）· cov≥0.95 PASS
- SoftLib plot footer el/cyc≈7.70（popped VF≈1064）；headline KPI 仍用 harvest **7.42**（VF_cyc=1104）
- SoftLib mid-med IPC ≈ **1.90**；SIMD mid-med **2.20**
- **NOT merged** — 等 Lok yes

## Harness
t822u tip 已有 `lane_mode=all_isa_groups` + cov gate；今次 **無改 harness**，只 regen plots + ANALYSIS。
