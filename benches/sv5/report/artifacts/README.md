# report/artifacts

Checked-in VF / instr logs for spine arms (real opsim only — no fakes).

| arm dir | kind | key files |
|---|---|---|
| `simd_dsl_best/` | DSL VL128 VAND storeA | `trace.json`, `instr_exe.csv`, `instr_popped_log.dump` |
| `simt_asc_default/` | Asc TileLang ub_stream T=128 | same |
| `softlib_simt/` | SoftLib contig LDS.b64 T=128 | same |

Each dir has `SOURCE.txt` with peanut path + OPPROF id.

```bash
python -m scripts.st_report plot-ipc --st sv5 --window 20
```
