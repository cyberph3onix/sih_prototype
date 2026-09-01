#!/usr/bin/env bash
#
# Start the prototype. Safe to run straight after `git clone` -- if the virtual
# environment is missing it is created and the dependencies installed first.
#
# Needs Python 3.10+ and, for the FIRST run only, an internet connection to
# fetch the dependencies. The demo itself runs entirely offline: no GPU, no
# database, no API keys, no network calls.
#
#   ./run_demo.sh              # http://localhost:8501
#   PORT=8600 ./run_demo.sh    # somewhere else

set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
PY="$VENV/bin/python"
PORT="${PORT:-8501}"

find_python() {
  for c in python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1; then echo "$c"; return 0; fi
  done
  echo "ERROR: no python3 on PATH. Install Python 3.10 or newer." >&2
  exit 1
}

if [ ! -x "$PY" ]; then
  BASE_PY="$(find_python)"
  echo "First run: creating $VENV with $("$BASE_PY" --version 2>&1)"

  if "$BASE_PY" -m venv "$VENV" 2>/dev/null && [ -x "$VENV/bin/pip" ]; then
    "$PY" -m pip install --quiet --upgrade pip
    "$PY" -m pip install --quiet -r requirements.txt
  elif command -v uv >/dev/null 2>&1; then
    echo "  (python's venv module is unavailable; falling back to uv)"
    rm -rf "$VENV"
    uv venv "$VENV"
    uv pip install --python "$PY" -r requirements.txt
  else
    rm -rf "$VENV"
    cat >&2 <<'MSG'

ERROR: could not create a virtual environment.

Your Python installation is missing the `venv` module. Install it, or install
uv, then run this script again:

  sudo apt install python3-venv                      # Debian / Ubuntu
  sudo dnf install python3-virtualenv                # Fedora
  curl -LsSf https://astral.sh/uv/install.sh | sh    # or: uv, no sudo needed

MSG
    exit 1
  fi
  echo "Setup complete."
fi

echo "Dashboard: http://localhost:${PORT}   (Ctrl-C to stop)"
exec "$VENV/bin/streamlit" run app.py --server.port "$PORT"
