# Benches layout

Each sensitivity ST lives in its own folder:

```
benches/<id>/
  <kernel>.py          # simt-asc / simd-pto source
  <kernel_vmi>.py      # optional VMI twin
  <kernel_dsl>.py      # optional handwritten SIMD / PTO-DSL twin
  config.toml          # shapes / arms / argv (gate + explore)
  VERDICT.md           # per-case testing verdict (SV1: ANALYSIS.md only)
```

- **Primary focus:** simt-asc, simd-pto, handwritten SIMD DSL.
- **Deferred:** simt-pto SoftLib lowering (sources still under benches/; deferred path not a focus).
- Oneshots / harness call `benches/<id>/…` directly (no separate `kernels/` symlink layer).

Run a single gate case (example SV1):

```bash
python benches/sv1/sv1_stream_eltwise.py 256 32
# or via symlink:
python benches/sv1/sv1_stream_eltwise.py 256 32
```

Case matrices are declared in each `config.toml` `[[cases]]` (see SV1 `suite = "explore"` for the size/thread ladder).

SV1 full Simt/Simd explore case list, sensitivity plots (tile/thread), instruction breakdown, and SIMD keynote (~85% of ~44 el/cyc): [`benches/sv1/ANALYSIS.md`](sv1/ANALYSIS.md).
