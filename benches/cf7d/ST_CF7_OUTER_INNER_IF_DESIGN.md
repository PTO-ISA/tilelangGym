# ST-CF7 design — outer gate slack / AND-merge × inner if (R* × p_gate crossover)

**Date:** 2026-10-30 ~19:32 HKT (rev: crossover = VF cycle, not µs)  
**Suite:** `/workspace/st_simtvf_parallel/`  
**Status:** **first-cut MEASURED** — see `ST_CF7D_FIRSTCUT_RESULT.md` (27/27 PASS; crossover G=0; A≡B)  
**Parent issues:** [tilelangGym#10](https://github.com/PTO-ISA/tilelangGym/issues/10); Lok asks 2026-10-30 (nested outer×inner if → +slack/AND + crossover grid)  
**R*elated:** `ST_CF1_CF6_SLICES.md`, `ST_CF_IMPLEMENT.md`, `ST_CF5_CF6_DIVER*GENCE.md`, `ST_VF_THEOR*Y.md`

---

## 0. Center of this design (Lok rev)

Measure the **crossover** where outer **scalar if (Sreg/ASU)** beats outer **slack/predicate** paths, as a function of:

- **R*** = inner-loop repeat per outer-true trip  
- **p_gate** = fraction of outer trips where gate is true  

**Three arms (same gold), primary = opsim VF cycle / VF length; µs secondary + opcode-bucket:**

| Arm | Outer gate | How outer pred reaches inner | Tag suffix |
|-----|------------|------------------------------|------------|
| **A · pure_slack** | rewrite outer `if` as always-enter; outer false = full slack / lockstep pay both sides | outer gate **not** AND'd into inner mask (inner still runs full Parallel CF under false outer; body may be no-op by data but opcodes still issue) | `_armA` |
| **B · slack_and** | outer gate via **slack/predicate** (no Sreg skip) | after outer slack settles, **AND-merge** outer predicate into inner loop mask (`p_inner &= p_outer`) so false-outer trips **squash** inner CF work at predicate level | `_armB` |
| **C · scalar_if** | outer **Sreg / ASU** `if` — false outer **skips** whole inner nest | n/a (no enter) | `_armC` |

R*ooftop twin **CF7d**: Python/host scalar `if` around VL-chunk + inner always `vcmp+vsel` (arm-C shape explicit).

**Primary question:** on an `R* × p_gate` grid, where does **C** clearly beat **A** and **B** in **VF cycle** (opsim VF length / cycle count), and does **B** close most of the A→C gap via AND-merge alone? Wall µs is secondary context only.

---

## 1. Does CF1–CF6 / CF*d already cover this?

**No.** There is **no** micro where an **outer loop carries a dynamic `if`** *and* an **inner loop carries a dynamic `if`**, and **no** arm that AND-merges an outer gate predicate into an inner Parallel mask.

| Case | Outer nest | Outer CF? | Inner nest | Inner CF? | Notes |
|------|------------|-----------|------------|-----------|-------|
| CF1–CF3 | `serial(K)` | **No** | `Parallel(E)` | **Yes** `if` | CF inside Parallel only |
| CF4 / CF4d | single `Parallel(E)` | n/a | same Parallel | **Yes** nested `if` / nested `vsel` | Nested **predicates**, same loop level — **not** outer×inner loops |
| CF5 / CF5d | — | — | `Parallel(E)` | **Yes** | |
| CF6 / CF6d | — | — | `Parallel` + `serial(N_FAST)` under else | **Yes** on Parallel; serial has **no** `if` | |
| CF6b | — | — | one Parallel if/elif/else | all CF on **one** Parallel | |
| CF*v | Kernel `serial` outside VF (CF5v/6v) | compute hoist, not CF policy cut | `T.Select` | Select-only | |

**CF4 “nested if” ≠ nested loops with if.** Cite: `kernels/cf4_nested_if.py`; `kernels_ptodsl/cf4d_nested_if.py`.

---

## 2. Loop shapes (concrete)

### 2.1 Shared nest (all arms)

```python
# Illustrative — not landed
with T.SimtVF(threads=T):
    for b in T.serial(B):                      # OUTER* trips
        gate = gate_ub[b]                      # dynamic scalar / broadcast
        # === arm-specific OUTER* GATE (below) ===
        for i in T.Parallel(E):                # INNER*
            # INNER* CF — always predicated / Select
            if scores[i] > thr_ub[b]:          # (arm B: effective mask &= p_outer)
                scores[i] = NEG
        for t in T.serial(G):                  # optional fat inner under outer-true
            # Newton-style / CF6-class work; G = primary R* axis when G>0
            ...
```

**Gold:** when outer gate false for trip `b`, that trip must **not** apply the inner kill (nor fat `serial(G)` side effects). When true, inner kill matches CF1 semantics for `thr_ub[b]`.

### 2.2 Arm A — pure_slack

```text
# Pseudocode / expected lowering intent
p_outer = (gate > gate_thr)          # predicate / slack, NO scalar skip
# always enter inner nest
for i in Parallel(E):
    if scores[i] > thr_ub[b]:        # inner pmask ONLY — p_outer NOT AND'd
        scores[i] = NEG
for t in serial(G): ...              # still scheduled; false-outer = slack / dead work
```

False-outer trips still **issue** inner CF + `G` body (flat skew lesson from CF4/CF5/CF6).

### 2.3 Arm B — slack + AND (center arm)

```text
p_outer = (gate > gate_thr)          # slack/predicate settle (no Sreg skip)
# AND-merge into inner
for i in Parallel(E):
    # effective: p_lane = (scores[i] > thr) AND p_outer
    if scores[i] > thr_ub[b]:        # lowering MUST fold p_outer into this mask
        scores[i] = NEG              # false-outer → all-false pmask (squash stores)
for t in serial(G):
    # under p_outer ( predicated serial / masked body — still entered, but
    # dead under false p_outer if compiler honors AND; document as-lowered )
    ...
```

**Implementation note:** if TileLang cannot express “AND outer pred into Parallel if” as source, force via explicit `T.Select` / masked store with `p_outer & p_inner`, or document **as-lowered** opcode mix. Arm B is the **policy** under test (#10 inner pmask path fed by outer gate).

### 2.4 Arm C — scalar_if (Sreg / ASU outer)

```text
if gate > gate_thr:                  # R*EAL scalar skip candidate
    for i in Parallel(E):
        if scores[i] > thr_ub[b]:
            scores[i] = NEG
    for t in serial(G): ...
# else: entire inner nest not entered
```

### 2.5 CF7d twin (PTO-DSL rooftop ≈ arm C shape)

- Outer: host `for b in range(B):` + Python `if gate[b] > thr:` around VL-chunk body.  
- Inner: always `vcmp` + `vsel` (never divergent lane if).  
- Forbidden: AABBCC / `vf_fuse` / inventing SimdVF `T.if` inside Parallel.  
- Optional Layer-B CF7v: Kernel `serial` + Select inside SimdVF (ABI doc only).

---

## 3. Tags, knobs, R* × p_gate grid

### 3.1 Knobs

| Knob | Values | R*ole |
|------|--------|------|
| `E` | 256 | match CF1–CF6 |
| `T` / VL | 32 Simt / 64 *d | |
| `B` | **8** (fixed primary) | outer trip count |
| `G` | {0, 8, 16, 32, 64} | fat inner serial under outer-true; **primary R* when G>0** |
| `p_gate` | {5, 25, 50, 75, 95} | host fraction outer-true |
| arm | A / B / C (+ `cf7d`) | path under test |

**R* definition:**

```text
R* = N_inner per outer-true trip
  = G                         when G > 0   (fat serial; primary sweep axis)
  = E / T                     when G = 0   (thin Parallel only; elems/thread)
R*eport both when G>0 (thin Parallel + G).
```

### 3.2 Tag scheme

```text
cf7_e{E}_b{B}_g{G}_t{T}_pgate{P}_arm{A|B|C}
cf7d_e{E}_b{B}_g{G}_vl{VL}_pgate{P}          # rooftop ≈ C
```

Examples: `cf7_e256_b8_g16_t32_pgate25_armB`, `cf7d_e256_b8_g16_vl64_pgate50`.

### 3.3 R*ecommended grid (first cut)

**Full design grid (when measuring crossover):**

| Axis | Points |
|------|--------|
| `G` (→ R*) | 0, 8, 16, 32, 64 |
| `p_gate` | 5, 25, 50, 75, 95 |
| arms | A, B, C (+ one matching `cf7d` per `(G,p_gate)`) |

→ **5 × 5 × 3 = 75** Simt cells + 25 `cf7d` = 100 oneshots if full; too heavy for first night.

**Primary first-cut slice (land first):**

| | `p_gate=25` | `p_gate=50` | `p_gate=95` |
|--|:-----------:|:-----------:|:-----------:|
| `G=0` (R*≈E/T=8) | A,B,C | A,B,C | A,B,C |
| `G=16` (R*=16) | A,B,C | A,B,C | A,B,C |
| `G=64` (R*=64) | A,B,C | A,B,C | A,B,C |

= **27** Simt + **9** `cf7d` cells. Aligns with `ST_VF_THEOR*Y` unroll cliff at serial **>16**.

**Fill-in later:** `G∈{8,32}`, `p_gate∈{5,75}` on the arm that looks closest to crossover from the first-cut heatmaps.

### 3.4 Provisional crossover hypothesis (unmeasured)

From prior CF7 estimate (`q = 1 - p_gate`; false-outer still pays ~full inner under A):

```text
C beats A when:  q * W_inner  ≳  C_br
B sits between A and C if AND-merge actually squashes inner issue count.
```

| R*egime | Expect |
|--------|--------|
| `p_gate≤50%`, `R*≳16` | **C** clearly &lt; **A** in **VF cycle**; **B** should approach **C** on body_instr / VF cyc if AND works |
| `p_gate=95%`, `R*≤8` | noise / flat — C may not win; need `R*≳32–64` |
| VF cycle flat across `p_gate` on arm A | outer still pure slack (no skip) — expected |
| VF cycle flat on arm B but body_instr drops with `q` | AND-merge works at opcode level even if VF issue tax remains |
| Arm C VF cycle tracks `p_gate` (linear-ish in `q`) | scalar skip real |

**Crossover (primary gate — Lok 2026-10-30):** smallest `R*` (prefer `G`) at a given `p_gate` such that  
`VF_cycle(C) ≤ 0.90 × min(VF_cycle(A), VF_cycle(B))`  
where `VF_cycle` = opsim VF length / cycle count (not wall µs).  
**Also require** arm-C opcode mix shows outer skip (body_instr / VF cyc scales with `p_gate`) while A stays flat.  
**Secondary note:** report wall µs alongside for context; µs alone must **not** declare crossover.

---

## 4. Acceptance — what marks crossover

**Primary metric:** opsim **VF cycle** / VF length (`VF_cycle`). Wall µs is secondary only.

**Crossover definition (Lok):**  
`Crossover = smallest R*` (prefer `G`) at fixed `p_gate` such that  
`VF_cycle(C) ≤ 0.90 × min(VF_cycle(A), VF_cycle(B))`.

Per grid cell, harvest:

| Signal | Arm A | Arm B | Arm C (win) |
|--------|-------|-------|-------------|
| **VF cycle** vs `p_gate` | flat or weakly varying | flat-ish or mild drop | **drops as `p_gate`↓** (skip savings) — **primary** |
| `body_instr` | flat vs `p_gate` | **drops** when AND squashes | drops with `p_gate` |
| wall µs vs `p_gate` | secondary context only | secondary | secondary — do **not** accept crossover on µs alone |
| pred/select bucket | high (inner + slack outer) | high; outer compare + **AND** into inner | inner pred/select only on true trips |
| `SIMT_BR*ANCH` / Sreg | absent or only from `G>16` serial | same | may appear on outer gate **or** only on fat `serial(G)` — attribute in notes |
| gold | pass | pass | pass |

**Crossover cell:** first `(R*, p_gate)` meeting the VF-cycle rule above **and** arm-C opcode mix shows outer skip (body_instr / VF cyc scales with `p_gate`) while A stays flat. R*eport heatmaps `ΔVF(C−A)`, `ΔVF(C−B)`, `ΔVF(B−A)`; optionally also `Δµs` as secondary.

**Fail modes to document (not crossover):**

- A≈B≈C flat → compiler predicated everything; no real outer skip / AND ineffective.  
- B≈A, C wins → AND-merge not lowered; only scalar skip helps.  
- B≈C, A loses → AND-merge recovers most skip benefit without Sreg (important #10 finding).

---

## 5. Logging (default for CF7 grid)

**Default:** opcode-bucket harvest from `core0.veccore0_instr_exe.csv` (extend `harvest_cf_summary.py` / reuse `harvest/harvest_instr_mix_v2.py`):

| Bucket | Opcodes (suite names) | Marks |
|--------|----------------------|-------|
| pred/select | `simt_sel`, `simt_fsetp`, `simt_isetp`, `vsel`, `vcmp`, `rv_vsel`, `rv_vcmp_*` | inner pmask / Select / slack |
| sreg/branch | `SIMT_BR*ANCH`, `SIMT_ISETP_I`, scalar `br`/`jump` if present | Sreg if-else / branchy serial |
| compare | `fsetp`/`isetp`/`vcmp` | CF predicate setup |

**Per cell hooks:**

1. `call_count` per bucket + `body_instr` + **VF cyc (primary)** + wall µs (secondary).  
2. **Skew flatness:** plot / table **VF cycle** and body_instr vs `p_gate` at fixed `G` (as CF6b); µs optional.  
3. Tag embeds arm so grid pivot is mechanical.

**Not default:** full scalar-PC ↔ VF-issue timeline (suite harvest not wired; high overhead). Use only if A/B/C opcode buckets cannot separate AND-merge from pure slack.

---

## 6. Out of scope this pass

- ~~Landing kernels / oneshot / opsim~~ → done (*d first-cut)  
- ~~Claiming measured crossover R* via VF cycle~~ → **G=0** for p∈{25,50,95}; p95 non-monotone at G=16/64  
- Production topk_gate rewrite  

## 7. Implementation order (when approved)

1. Land thin CF7 `G=0` arms A/B/C + CF7d @ `p_gate∈{25,50}` — prove gold + opcode buckets separate arms.  
2. Extend to first-cut grid (`G∈{0,16,64}` × `p_gate∈{25,50,95}` × A/B/C).  
3. Heatmap crossover on **VF cycle** (rule in §3.4 / §4); fill `G∈{8,32}` / `p_gate∈{5,75}` near the cliff.  
4. Link from `ST_CF1_CF6_SLICES.md` + close #10 acceptance bullets.
