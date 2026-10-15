# SV1 Jupyter on pto-b10

Interactive report: `SV1_report.ipynb` (plots from `data/` offline; optional opsim cells).

## One-time setup (login node)

```bash
cd ~/projects/tilelangGym
git fetch origin && git checkout chore/reorg-scripts-layout && git pull

# Prefer a small venv so we don't disturb .venv-npu torch stack
python3 -m venv ~/venvs/sv1-jupyter
source ~/venvs/sv1-jupyter/bin/activate
pip install -U pip
pip install jupyterlab notebook pandas matplotlib tomli numpy
```

## Start Lab (token required — do not use empty token)

```bash
cd ~/projects/tilelangGym/benches/sv1
source ~/venvs/sv1-jupyter/bin/activate
export JUPYTER_TOKEN=$(openssl rand -hex 24)
mkdir -p ~/sv1_jupyter_logs
nohup jupyter lab --no-browser --ip=0.0.0.0 --port=8899 \
  --ServerApp.token="$JUPYTER_TOKEN" \
  --ServerApp.password='' \
  --ServerApp.root_dir="$HOME/projects/tilelangGym" \
  > ~/sv1_jupyter_logs/lab_8899.log 2>&1 &
echo "PID=$! TOKEN=$JUPYTER_TOKEN"
echo "URL=http://pto-b10.ddns.net:8899/lab?token=$JUPYTER_TOKEN"
```

Prefer SSH tunnel from Windows (safer than exposing WAN):

```powershell
# on DESKTOP-V7R2A6I
ssh -N -L 8899:127.0.0.1:8899 pto-b10
# browser: http://127.0.0.1:8899/lab?token=<TOKEN>
```

Open notebook: `benches/sv1/SV1_report.ipynb`.

## Stop

```bash
pkill -f 'jupyter-lab.*8899' || true
```

## Opsim re-run (optional)

Use camodel env from the TileLang Ascend skill (`.venv-npu` + `cann_91b3`, `TORCH_DEVICE_BACKEND_AUTOLOAD=0`). Plot cells do not need CANN.
