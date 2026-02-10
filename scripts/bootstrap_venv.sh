#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR=".venv"

if [[ -x "${VENV_DIR}/bin/python" && -x "${VENV_DIR}/bin/pip" ]]; then
  echo "[setup] .venv already exists"
else
  echo "[setup] Creating venv without pip: ${VENV_DIR}"
  if ! python3 -m venv --without-pip "${VENV_DIR}"; then
    echo "[setup] ERROR: failed to create venv. On Ubuntu, you may need: sudo apt-get install -y python3-venv" >&2
    exit 1
  fi

  GET_PIP="${VENV_DIR}/get-pip.py"
  if [[ ! -f "${GET_PIP}" ]]; then
    echo "[setup] Downloading get-pip.py"
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL -o "${GET_PIP}" https://bootstrap.pypa.io/get-pip.py
    elif command -v wget >/dev/null 2>&1; then
      wget -qO "${GET_PIP}" https://bootstrap.pypa.io/get-pip.py
    else
      python3 - <<'PY'
import pathlib, urllib.request
dst = pathlib.Path(".venv/get-pip.py")
dst.parent.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen("https://bootstrap.pypa.io/get-pip.py") as r:
    dst.write_bytes(r.read())
print(f"[setup] Wrote {dst}")
PY
    fi
  fi

  echo "[setup] Bootstrapping pip inside venv"
  "${VENV_DIR}/bin/python" "${GET_PIP}" "pip==25.0.1" "setuptools==75.8.0" "wheel==0.45.1"
fi

PIP="${VENV_DIR}/bin/pip"

TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
if [[ -n "${TORCH_INDEX_URL}" ]]; then
  echo "[setup] Installing requirements with TORCH_INDEX_URL=${TORCH_INDEX_URL} (as extra index)"
  "${PIP}" install --upgrade -r requirements.txt --extra-index-url "${TORCH_INDEX_URL}"
else
  echo "[setup] Installing requirements"
  "${PIP}" install --upgrade -r requirements.txt
fi

echo "[setup] Done"
