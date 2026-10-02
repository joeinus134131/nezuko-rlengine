#!/usr/bin/env bash
# Entrypoint for the finance container: wait DB -> migrate -> start Streamlit.
set -euo pipefail

cd /app

if [ -n "${DATABASE_URL:-}" ] || [ -n "${POSTGRES_HOST:-}" ]; then
  echo "[entrypoint] waiting for Postgres..."
  python scripts/migrate_db.py
  echo "[entrypoint] migrations done."
else
  echo "[entrypoint] DATABASE_URL not set, running in local-file mode."
fi

exec streamlit run finrl/dashboard.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.headless=true \
  --browser.gatherUsageStats=false
