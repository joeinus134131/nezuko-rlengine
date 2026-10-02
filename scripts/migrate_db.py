"""Run pending Alembic migrations. Used by entrypoint, CI, and manual ops."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from finrl.db.database import get_database_url, wait_for_db

    if not get_database_url():
        print("DATABASE_URL belum diset, migrasi dilewati (mode file lokal).")
        return 0
    wait_for_db(timeout_seconds=60)
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(ROOT / "alembic.ini"), "upgrade", "head"],
        cwd=str(ROOT),
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
