# Lok t704u — 點解 G64/G128 SoftLib SIMT 唔用細 T（T128）+「WC-only」？

**來源（實測，唔 invent）：**  
`RESULT_multwarp_t686u.tsv` · `LDF3254_WARP_VS_MULTIWARP_t686u.md` ·  
pack `/mnt/fluxdata/happybot/glm53_softlib_simt_absmax_multwarp_t686u/` ·  
CASE_DEF / MEASURE_DEFAULTS_t686u · sources `wc_abs_lds32_redux_g{64,128}_t*.asc`

---

## 短答（粵語）

1. **細 T（T128）已經量過**，唔係「未試」——G64 T128 **2.39** vs T512 **5.54**（~2.3× 差）；G128 T128 **3.36** vs T256 **3.72**。主 SoT 揀大 T 係因為 **el/cyc 實測更高**（多 warp 同時飛 → latency hide / occupancy），唔係理論空講。
2. **「WC-only」（G16 halfmask / G32 一 lane 一 elem）結構上 cover 唔到 G≥64**：warp=32，gold G=64/128 一定要 **per=2 / per=4**（lane 內先 fold 再 `asc_reduce_max`）= **warp-coop-multi**。T128 嗰排已經係同一 motif，只係 launch T 細啲。
3. **真正 single-warp「一 lane 一 elem」G64** = 未作為 Case-1 1×G 獨立 motif 量過（亦唔合理：32 lane 唔夠 64 elem）；peer **wg**（無 warp redux）T128 實測 ~**0.98**，遠差過 wc-multi **2.39**。
4. G128 **T512 未量**（ladder 停喺 T256）；問題問細 T，細 T 已有數。

---

## 表 — SoftLib SIMT Case-1 absmax · axis=1×G · wc-multi T-sweep（t686u）

| G | motif | T | warps | VF_cyc | el/cyc | 角色 |
|--:|-------|--:|------:|-------:|-------:|------|
| 64 | wc-multi per=2 LDS32 | **128** | 4 | 3434 | **2.3856** | stock（已量） |
| 64 | wc-multi per=2 LDS32 | 256 | 8 | 2246 | 3.6474 | |
| 64 | wc-multi per=2 LDS32 | **512** | 16 | 1478 | **5.5426** | **main SoT** |
| 128 | wc-multi per=4 LDS32 | **64** | 2 | 4916 | **1.6664** | stock |
| 128 | wc-multi per=4 LDS32 | **128** | 4 | 2441 | **3.3560** | 已量、輸畀 T256 |
| 128 | wc-multi per=4 LDS32 | **256** | 8 | 2201 | **3.7219** | **main SoT** |
| 128 | — | 512 | — | — | **未量** | ladder 未跑 |

G16/G32 對照（同一 pack，顯示大 T 一樣靠 latency hide）：G32 T128 **1.57** → T512 bare **5.72**（其後 t689u dual-drain **8.22**）。

---

## 點解一定係 warp-coop-multi（唔係 G32 式 WC-only）

| G | 每 gold group | source 註解 | 含義 |
|--:|---------------|-------------|------|
| 16 | 16 lane halfmask · 1 elem/lane | `wc … halfmask` | 單 warp 內 |
| 32 | 32 lane · 1 elem/lane | `wc` full-warp | 單 warp 內 |
| 64 | 32 lane · **2 elem/lane (per=2)** | `motif=warp-coop-multi … per2` | fold **in-group** 再 warp redux |
| 128 | 32 lane · **4 elem/lane (per=4)** | `… per4` | 同上 |

T 只改 `__launch_bounds__(T)` / 同時有幾多 group in-flight；**唔改** per=2/4 reduce 結構。所以「細 T + WC-only」唔成立——細 T 有量而且輸；WC-only（per=1）對 G≥64 唔係 Case-1 1×G 可行 mapping。

---

## 主 SoT 點揀

| G | 揀 | 原因（實測） |
|--:|----|--------------|
| 64 | T**512** wc-multi | T128/256/512 單調升；T512 最高 **5.54**（latency hide） |
| 128 | T**256** wc-multi | T64→T128→T256 升；T256 最高 **3.72**；T512 **未量** |

Pack: `glm53_softlib_simt_absmax_multwarp_t686u` · KPI `el/cyc=E/VF` · `era=fp32-absmax` · E=8192 · no recip.
