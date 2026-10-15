# Benches layout

Each sensitivity ST lives in its own folder:

```
benches/<id>/
  <kernel>.py          # simt-asc / simd-pto source
  <kernel_vmi>.py      # optional VMI twin
  <kernel_dsl>.py      # optional handwritten SIMD / PTO-DSL twin
  config.toml          # shapes / arms / argv (gate + explore)
  VERDICT.md           # per-case testing verdict
```

- **Primary focus:** simt-asc, simd-pto, handwritten SIMD DSL.
- **Deferred:** simt-pto SoftLib lowering (kernels still runnable via legacy oneshots).
- Legacy paths `kernels/` and `kernels_ptodsl/` are **symlinks** into these folders so existing `oneshot_*.sh` keep working.

Run a single gate case (example SV1):

```bash
python benches/sv1/sv1_stream_eltwise.py 256 32
# or via symlink:
python kernels/sv1_stream_eltwise.py 256 32
```

Case matrices are declared in each `config.toml` `[[cases]]` (see SV1 `suite = "explore"` for the size/thread ladder).

SV1 full Simt/Simd explore case list, sensitivity plots (tile/thread), instruction breakdown, and LDS/STS.64/.128 rooftop: [`benches/sv1/ANALYSIS.md`](sv1/ANALYSIS.md).
