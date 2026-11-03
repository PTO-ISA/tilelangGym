# ST-CF7 rev — global outer gate (Lok 2026-11-03)

## Bug in as-landed gold

As-landed outer was **per-b**:

```python
for b in range(B):
    if float(gate[b]) > float(gate_thr):   # per-trip scalar
        ...
```

That forces arm C into **host/compile `n_trips=n_true` specialization** (false trips omitted from `@pto.jit` body) — **not** a real Sreg/ASU scalar `if` **inside** the VF. Arm B also cannot cleanly `create_bcast` a single compare into a preg and `pand` with the inner kill mask, because outer truth varies per `b`.

## Intended outer (global)

```python
# gold
gate_sum = float(np.sum(gate))            # reduce_sum(gate[b])
if gate_sum > GATE_SUM_THR:               # ONE global outer if
    for b in range(B):                    # thr still per-b
        for _ in range(G):
            out *= SCALE
        out = np.where(out > thr[b], -inf, out)
```

`p_gate` / inputs: build `gate[]` so `sum(gate)` is **above** thr for “on” cells and **below** for “off” (binary kernel-level bit; optional magnitude sweep later). First-cut grid stays `G × {gate_on, gate_off}` (or keep pgate labels as on≈95 / off≈25 proxies via sum).

## Three *d lowerings (same gold)

| Arm | Outer handling inside VF |
|-----|--------------------------|
| **A · pure_slack** | Always enter B×(G+kill). May still compute `gate_sum` / compare and discard — **no** skip, **no** AND into kill. |
| **B · preg+PAND** | `gate_sum = reduce_sum(gate)`; `p_outer = (gate_sum > thr)`; **`create_bcast` → preg**; `p_kill = pand(p_outer, scores>thr[b])` then `vsel`. False outer → all-false preg, squash stores; body still scheduled. |
| **C · scalar_if** | Same reduce+cmp to **Sreg**; **`if gate_sum > thr:`** around the whole B×(G+kill) nest — false = real VF skip (no host `n_trips` specialization). |

## Forbidden / deprecated after this rev

- Host `n_trips = n_true if C else B` kernel specialization as the arm-C definition.
- Per-b `gate[b] > thr` as the outer CF under test.
- Claiming A≠B without a real `pand`/`create_bcast` path on B.

## Next land

1. Patch `ST_CF7_OUTER_INNER_IF_DESIGN.md` §2 + gold.
2. Rewrite `cf7d_outer_inner_if.py` arms A/B/C + `reference()`.
3. Re-opsim first-cut; mark prior A≡B / crossover numbers **stale** until remeasure.
4. Refresh ANALYSIS §2b slices + SEMANTICS_GATE.
