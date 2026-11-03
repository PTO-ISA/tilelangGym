# CF7d sibling — reduce-in-b (no hoist)

**Date:** 2026-11-03 (Asia/Hong_Kong) · Lok pt44u  
**Kernel:** `benches/cf7d/cf7d_reduce_in_b.py`  
**Smoke:** `benches/cf7d/smoke_thin_rib.py`  
**Era:** `era=cf7d-rib-ptodsl` (live KPI UNMEASURED)

## Why

Hoisted case (`cf7d_outer_inner_if.py`) computes `gate_on` / `p_outer` **once** then reuses across `B` trips. Arm C can then look cheap by paying reduce tax once while still wrapping each `b` with `pto.if_(gate_on)` that has no reduce dep. Lok: that can fake a C win.

## Spec

- Keep hoisted case as-is.
- Gold value identical: `sum(gate) > GATE_SUM_THR`.
- But **recompute** that reduce **inside** each outer `b` iteration (no hoist).
- Arm **B:** `vcadd→vbrc→vcmp→vand` (+ `vsel` publish) **each b**.
- Arm **C:** per-b reduce → `load_scalar` → `pto.if_(gate_on)` wrapping G nest **inside** b.
- `B=8` fixed; `G ∈ {0,8,16,32,64}` primary sweep.
- Tag: `cf7d_rib_e{E}_b{B}_g{G}_vl{VL}_pgate{P}_arm{B|C}`.

## Gold (host)

```text
for b in range(B):
    gate_on_b = sum(gate) > GATE_SUM_THR   # same value each trip; kernel must recompute
    if gate_on_b:
        for _ in range(G):
            scale; kill(thr[b])
```

## Status

Compile smoke intended on pto-b10. Live VF **UNMEASURED**. SEMANTICS_GATE INCONCLUSIVE until dumps.
