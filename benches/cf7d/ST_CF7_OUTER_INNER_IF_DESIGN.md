# ST-CF7 design — global outer gate × inner kill (B preg+PAND / C pto.if_)

**Date:** 2026-11-03 ~15:40 HKT (rev: **global** `gate_sum > GATE_SUM_THR`; 2 arms; kill inside G)  
**Prior design date:** 2026-10-30 ~19:32 HKT (per-b outer; 3 arms A/B/C) — **STALE gold**  
**Suite:** `/workspace/st_simtvf_parallel/` / worktree `benches/cf7d/`  
**Status:** **kernel+docs LANDED** — live VF **UNMEASURED**; prior first-cut RESULT = `era=per-b-outer` **STALE**  
**Parent issues:** [tilelangGym#10](https://github.com/PTO-ISA/tilelangGym/issues/10); Lok redesign 2026-11-03  
**Related:** `ST_CF7_GLOBAL_GATE_REV.md`, `ANALYSIS.md` Appendix H

---

## 0. Center of this design (Lok rev 2026-11-03)

Measure whether outer **in-VF scalar `if_` (arm C)** beats outer **preg+PAND slack (arm B)** when the outer condition is **one global gate** over `sum(gate)`, as a function of inner fat **G**.

**Two arms (same gold), primary = opsim VF cycle; µs secondary + opcode-bucket:**

| Arm | Outer gate | How outer pred reaches inner | Tag |
|-----|------------|------------------------------|-----|
| **B · preg+PAND** | `vcadd(gate)` → bcast → `vcmp` → preg; **no** Sreg skip | `vand(p_outer, scores>thr)` into kill; `vsel(p_outer)` publish; G body still scheduled when false | `_armB` |
| **C · scalar_if** | same reduce → scalar `gate_on`; **`pto.if_(gate_on)`** wraps G nest | false = real VF skip of then-nest — **no** host `n_trips` | `_armC` |

**Dropped:** arm **A** pure_slack (Lok).

**Nest order (Lok):** **B outer**, **G inner**, **kill inside G** (so B needs PAND each kill step).

**Primary question:** on `G × {gate_on, gate_off}`, where does **C** clearly beat **B** in VF cycle? Does B's vand path show a distinct opcode mix vs historical A≡B fail-mode?

---

## 1. Does CF1–CF6 / prior CF7d cover this?

**No** for the **global** outer gate + in-VF `if_`. Prior CF7d first-cut covered **per-b** `gate[b]>thr` with C = host `n_trips` — different gold / different C lowering (`era=per-b-outer`, ANALYSIS Appendix H).

CF1–CF6: Parallel-only / nested predicates same loop — no outer×inner with global reduce gate (see prior design §1 table; retained conceptually).

---

## 2. Loop shapes (concrete)

### 2.1 Shared gold

```python
gate_on = float(np.sum(gate)) > float(GATE_SUM_THR)
for b in range(B):
    if gate_on:
        for _ in range(G):
            out = (out * SCALE).astype(np.float32)
            out = np.where(out > thr[b], -inf, out)
```

### 2.2 Arm B — preg + PAND

```text
gs = vcadd(gate[0:B])
p_outer = (vbrc(gs) > GATE_SUM_THR)          # preg, no Sreg skip
for b in range(B):
    for t in range(G):
        k = scores * SCALE
        k = select(vand(k > thr[b], p_outer), -inf, k)
        scores = select(p_outer, k, scores)  # publish
```

### 2.3 Arm C — pto.if_

```text
gs = vcadd(gate[0:B]); gate_on = load_scalar(gs) > GATE_SUM_THR
for b in range(B):
    with pto.if_(gate_on):
        then: for t in range(G): scale + kill
        else: identity assign
```

### 2.4 Forbidden after this rev

- Host `n_trips = n_true if C else B` as the definition of arm C.
- Per-b `gate[b] > thr` as the outer CF under test.
- Claiming A≠B / B efficacy without a real `vand`/`pand` path (A dropped).

---

## 3. Grid / knobs

| Knob | Value |
|------|-------|
| E | 256 |
| B | 8 |
| VL | 64 |
| G | {0, 8, 16, 32, 64} |
| pgate labels | 25 → gate_off; ≥50 → gate_on (input construction only) |
| GATE_SUM_THR | 0.0 |

First remeasure: prefer G∈{0,8,16,64} × {25,50,95} × {B,C}.

---

## 4. Pass / crossover

- Cell PASS: gold `allclose(..., equal_nan=True)`.
- A1 crossover: smallest G with `VF(C) ≤ 0.90 × VF(B)` at gate_on (document gate_off separately).
- Do not invent VF; mark UNMEASURED until oneshot.

---

## 5. Historical note (t821u)

Prior design text (per-b 3-arm, A/B/C, host C) and measured RESULT tables remain on disk / ANALYSIS Appendix H as **STALE**. They must not be silently deleted or unlabeled-folded into live global-gate KPI.
