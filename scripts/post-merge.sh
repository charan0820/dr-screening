#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Keep the connector bridge available after task merges. npm ci is
# deterministic because package-lock.json is committed with package.json.
if [[ -f package-lock.json ]]; then
  npm ci --no-audit --no-fund --ignore-scripts
elif [[ -f package.json ]]; then
  npm install --no-audit --no-fund --ignore-scripts
fi

python -m py_compile app/streamlit_app.py