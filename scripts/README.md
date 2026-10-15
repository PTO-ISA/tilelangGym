# scripts/

Grouped runners and helpers (kept out of the repo root).

| Path | Purpose |
|------|---------|
| `regression/` | Weekly / CI gates. Start with `oneshot_sv1_e8192_t128_t512.sh`. |
| `oneshot/` | Family oneshots (SV / CF / SP / PTO-DSL / vf-mode). |
| `laptop/` | PowerShell relays that SSH/SCP to the Ascend lab host. |
| `tools/` | Harvest / opsim utilities (`legacy/` = old scratch). |

Repo-root `oneshot_sv1_e8192_t128_t512.sh` is a thin wrapper that `exec`s
`regression/oneshot_sv1_e8192_t128_t512.sh` so the historic path still works.

Oneshot and regression scripts `cd` to the repo root and prepend it to
`PYTHONPATH`, so `common_*_harness.py` / `vf_mode.py` / `kernels/` keep working.
