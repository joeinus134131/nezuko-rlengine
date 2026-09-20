#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ENV_DIR="$PROJECT_DIR/.venv311"
PYTHON_BIN=${PYTHON_BIN:-/Users/user/.local/bin/python3.11}

"$PYTHON_BIN" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel swig
PATH="$ENV_DIR/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
  "$ENV_DIR/bin/python" -m pip install --no-build-isolation box2d-py==2.3.5
"$ENV_DIR/bin/python" -m pip install pygame==2.6.1
"$ENV_DIR/bin/python" -m pip install gym==0.26.2 --no-deps
"$ENV_DIR/bin/python" -m pip install gym-notices==0.1.0
"$ENV_DIR/bin/python" -m pip install elegantrl==0.3.6 --no-deps
PATH="$ENV_DIR/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
  "$ENV_DIR/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"
"$ENV_DIR/bin/python" "$PROJECT_DIR/scripts/check_setup.py"

echo "Run dashboard: $ENV_DIR/bin/streamlit run $PROJECT_DIR/finrl/dashboard.py"
