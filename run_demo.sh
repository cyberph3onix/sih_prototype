#!/usr/bin/env bash
# Start the judge-facing prototype. No internet, GPU, or database required.
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "No virtual environment found. Create one with:"
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  echo "  (or, with uv:  uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt)"
  exit 1
fi

exec .venv/bin/streamlit run app.py --server.port "${PORT:-8501}"
