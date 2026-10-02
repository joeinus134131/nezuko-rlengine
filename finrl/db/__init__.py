"""PostgreSQL persistence layer for NEZU workbench.

When ``DATABASE_URL`` is unset the dashboard keeps using the legacy
``configs/*.json`` + OS keyring storage, so local development still works
without a database. When it is set, auth / providers / app settings /
secrets / experiments are read from Postgres.
"""
from finrl.db.database import get_database_url, is_db_configured

__all__ = ["get_database_url", "is_db_configured"]
