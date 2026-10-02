"""Alembic environment: runs finrl.db.models metadata against DATABASE_URL."""
from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# Ensure repo root is importable (both local and /app in Docker).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finrl.db.database import get_database_url  # noqa: E402
from finrl.db.models import Base  # noqa: E402

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _url() -> str:
    url = get_database_url()
    if not url:
        raise RuntimeError(
            "DATABASE_URL (atau POSTGRES_HOST/USER/PASSWORD/DB) belum diset."
        )
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    ini_section = config.get_section(config.config_ini_section, {})
    ini_section["sqlalchemy.url"] = _url()
    connectable = engine_from_config(
        ini_section, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
