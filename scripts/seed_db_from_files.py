"""One-time import: configs/*.json + users.json -> Postgres.

Usage:
  DATABASE_URL=... APP_SECRET_KEY=... .venv311/bin/python scripts/seed_db_from_files.py

Idempotent: skips existing users/providers, overwrites app_settings.
Secrets in OS keyring are NOT auto-migrated (must re-enter via UI once,
they will then be encrypted in DB). File mirrors remain as fallback.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_DIR = ROOT / "configs"


def _read_json(name: str) -> dict | None:
    path = CONFIG_DIR / name
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    from finrl.db.database import get_database_url, wait_for_db
    from finrl.db.database import session_factory
    from finrl.db import repositories as repo
    from finrl.integrations import auth as local_auth

    if not get_database_url():
        print("DATABASE_URL belum diset.")
        return 1
    wait_for_db(30)
    session_factory().dispose() if hasattr(session_factory(), "dispose") else None

    with session_factory()() as session:
        # --- users ---
        try:
            store = local_auth.load_users(CONFIG_DIR / "users.json")
        except local_auth.AuthError:
            store = local_auth.empty_store()
        migrated_users = 0
        for username, record in store.get("users", {}).items():
            if repo.get_user(session, username) is None:
                # Re-hash? No: preserve existing salt/hash directly.
                from finrl.db.models import User

                session.add(
                    User(
                        username=username,
                        algorithm=record.get("algorithm", "pbkdf2_hmac_sha256"),
                        iterations=int(record.get("iterations", 240000)),
                        salt_hex=record["salt"],
                        hash_hex=record["hash"],
                        role=record.get("role", "admin"),
                    )
                )
                migrated_users += 1

        # --- providers + active_* ---
        try:
            from finrl.integrations.provider_adapter import load_provider_store

            pstore = load_provider_store(CONFIG_DIR / "providers.json")
        except Exception:
            pstore = {"providers": []}
        migrated_providers = 0
        for item in pstore.get("providers", []):
            try:
                repo.upsert_provider(session, item)
                migrated_providers += 1
            except Exception as error:
                print(f"skip provider {item.get('id')}: {error}")
        if pstore.get("active_market_data"):
            repo.set_setting(session, "active_market_data", pstore["active_market_data"])
        if pstore.get("active_broker"):
            repo.set_setting(session, "active_broker", pstore.get("active_broker", ""))

        # --- app settings ---
        migrated_settings = 0
        for name in ("ai_research.json", "paper_trading.json", "telegram_monitor.json"):
            payload = _read_json(name)
            if payload is not None:
                repo.set_setting(session, name[:-5], payload)
                migrated_settings += 1

        session.commit()
        print(
            f"Seed selesai: {migrated_users} users, "
            f"{migrated_providers} providers, {migrated_settings} settings."
        )
        print("Secrets keyring TIDAK auto-migrasi — input ulang sekali via UI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
