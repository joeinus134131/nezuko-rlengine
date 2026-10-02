"""Compatibility bridge: DB-first, JSON/keyring fallback for local dev.

Dashboard code calls these helpers instead of touching files directly.
If DATABASE_URL is unset everything behaves exactly like before.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from finrl.db.database import is_db_configured, session_factory


def _session():
    return session_factory()()


# ------------------------------------------------------------ app config ---
def read_app_config(name: str, config_dir: Path) -> dict[str, Any] | None:
    """name like 'ai_research.json' -> dict or None."""
    if is_db_configured():
        try:
            from finrl.db import repositories as repo

            key = name[:-5] if name.endswith(".json") else name
            with _session() as session:
                value = repo.get_setting(session, key, None)
            return dict(value) if isinstance(value, dict) else value
        except Exception:
            pass  # fall through to file
    path = config_dir / name
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_app_config(name: str, payload: dict[str, Any], config_dir: Path) -> str:
    """Persist to DB when configured AND mirror to file for offline fallback."""
    if is_db_configured():
        try:
            from finrl.db import repositories as repo

            key = name[:-5] if name.endswith(".json") else name
            with _session() as session:
                repo.set_setting(session, key, payload)
                session.commit()
        except Exception:
            pass
    # Always mirror to file so VPS has a local copy and local dev keeps working.
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if is_db_configured():
        return f"PostgreSQL + {path}"
    return str(path)


# -------------------------------------------------------------- providers ---
def load_provider_store(config_dir: Path) -> dict[str, Any]:
    from finrl.integrations.provider_adapter import load_provider_store as load_file

    if not is_db_configured():
        return load_file(config_dir / "providers.json")
    try:
        from finrl.db import repositories as repo

        with _session() as session:
            providers = repo.list_provider_dicts(session)
            active_market = repo.get_setting(session, "active_market_data", "") or ""
            active_broker = repo.get_setting(session, "active_broker", "") or ""
        if providers or active_market or active_broker:
            return {
                "active_market_data": active_market,
                "active_broker": active_broker,
                "providers": providers,
            }
    except Exception:
        pass
    return load_file(config_dir / "providers.json")


def save_provider_store(config_dir: Path, payload: dict[str, Any]) -> None:
    from finrl.integrations.provider_adapter import save_provider_store as save_file

    save_file(config_dir / "providers.json", payload)  # validates + local mirror
    if not is_db_configured():
        return
    try:
        from finrl.db import repositories as repo

        with _session() as session:
            for item in payload.get("providers", []):
                repo.upsert_provider(session, item)
            repo.set_setting(
                session, "active_market_data", payload.get("active_market_data", "")
            )
            repo.set_setting(session, "active_broker", payload.get("active_broker", ""))
            session.commit()
    except Exception:
        pass


# ------------------------------------------------------------------ auth ---
def load_auth_store(config_dir: Path) -> dict[str, Any] | None:
    """Return legacy-shaped store dict so dashboard code is unchanged."""
    from finrl.integrations import auth as local_auth

    path = config_dir / "users.json"
    if not is_db_configured():
        try:
            return local_auth.load_users(path)
        except local_auth.AuthError as error:
            raise error
    try:
        from finrl.db import repositories as repo

        with _session() as session:
            users = repo.list_users(session)
            store: dict[str, Any] = {"version": 1, "users": {}}
            for user in users:
                store["users"][user.username] = {
                    "algorithm": user.algorithm,
                    "iterations": user.iterations,
                    "salt": user.salt_hex,
                    "hash": user.hash_hex,
                    "role": user.role,
                }
            return store
    except Exception:
        try:
            return local_auth.load_users(path)
        except local_auth.AuthError:
            return {"version": 1, "users": {}}


def create_user_bridge(
    config_dir: Path, username: str, password: str, role: str = "admin"
) -> None:
    from finrl.integrations import auth as local_auth

    # Always keep file mirror for fallback.
    try:
        store = local_auth.load_users(config_dir / "users.json")
    except local_auth.AuthError:
        store = local_auth.empty_store()
    local_auth.create_user(store, username, password, role=role)
    local_auth.save_users(config_dir / "users.json", store)
    if not is_db_configured():
        return
    from finrl.db import repositories as repo

    try:
        with _session() as session:
            # Skip if already migrated (file was source of truth on first run).
            if repo.get_user(session, username.strip()) is None:
                repo.create_user_db(session, username, password, role=role)
                session.commit()
    except Exception:
        pass


def authenticate_bridge(config_dir: Path, username: str, password: str) -> bool:
    if not is_db_configured():
        from finrl.integrations import auth as local_auth

        try:
            store = local_auth.load_users(config_dir / "users.json")
        except local_auth.AuthError:
            return False
        return local_auth.authenticate(store, username, password)
    # DB-first, file fallback if DB down.
    try:
        from finrl.db import repositories as repo

        with _session() as session:
            ok = repo.authenticate_db(session, username, password)
        if ok:
            return True
        # If user table empty (fresh DB), fall back to file once.
        with _session() as session:
            if repo.count_users_db(session) == 0:
                from finrl.integrations import auth as local_auth

                try:
                    store = local_auth.load_users(config_dir / "users.json")
                except local_auth.AuthError:
                    return False
                return local_auth.authenticate(store, username, password)
        return False
    except Exception:
        from finrl.integrations import auth as local_auth

        try:
            store = local_auth.load_users(config_dir / "users.json")
        except local_auth.AuthError:
            return False
        return local_auth.authenticate(store, username, password)


def user_count_bridge(config_dir: Path) -> int:
    if not is_db_configured():
        from finrl.integrations import auth as local_auth

        try:
            return local_auth.user_count(local_auth.load_users(config_dir / "users.json"))
        except local_auth.AuthError:
            return 0
    try:
        from finrl.db import repositories as repo

        with _session() as session:
            count = repo.count_users_db(session)
        if count == 0:
            from finrl.integrations import auth as local_auth

            try:
                file_count = local_auth.user_count(
                    local_auth.load_users(config_dir / "users.json")
                )
                return file_count
            except local_auth.AuthError:
                return 0
        return count
    except Exception:
        return 0


# ---------------------------------------------------------------- secrets ---
def load_secret_bridge(name: str) -> str | None:
    if is_db_configured():
        try:
            from finrl.db import repositories as repo

            with _session() as session:
                value = repo.load_secret_db(session, name)
            if value:
                return value
        except Exception:
            pass
    try:
        from finrl.integrations.secure_store import load_secret

        return load_secret(name)
    except Exception:
        return None


def save_secret_bridge(name: str, value: str) -> None:
    # DB first (encrypted), keyring as mirror for local dev.
    if is_db_configured():
        try:
            from finrl.db import repositories as repo

            with _session() as session:
                repo.save_secret_db(session, name, value)
                session.commit()
        except Exception as error:
            # Missing APP_SECRET_KEY etc: surface clearly, then try keyring.
            raise error
    try:
        from finrl.integrations.secure_store import save_secret

        save_secret(name, value)
    except Exception:
        if not is_db_configured():
            raise
        pass
