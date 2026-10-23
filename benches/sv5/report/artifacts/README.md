# report/artifacts

Drop per-arm VF / instr logs here (or point `arms.yaml` `artifacts:` at absolute peanut paths).

Expected names (optional):
- `trace.json` — chrome-trace from opsim
- `instr_exe.csv` — `core0.veccore0_instr_exe.csv` (VF cycle rows)
- `instr_popped_log.dump` — cycle-accurate popped log

Then:
```bash
python -m scripts.st_report plot --st <id> --window 20
```
