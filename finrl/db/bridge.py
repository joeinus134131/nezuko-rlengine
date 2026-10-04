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
    stored_in_db = False
    if is_db_configured():
        try:
            from finrl.db import repositories as repo

            key = name[:-5] if name.endswith(".json") else name
            with _session() as session:
                repo.set_setting(session, key, payload)
                session.commit()
            stored_in_db = True
        except Exception as error:
            raise RuntimeError(
                f"Gagal menyimpan konfigurasi {name} ke PostgreSQL. "
                "Periksa DATABASE_URL dan migrasi app_settings. "
                f"Detail: {error}"
            ) from error
    # Always mirror to file so VPS has a local copy and local dev keeps working.
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if stored_in_db:
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

    db_mode = is_db_configured()
    db_ok = False
    if db_mode:
        from finrl.db import repositories as repo

        try:
            with _session() as session:
                repo.create_user_db(session, username, password, role=role)
                session.commit()
            db_ok = True
        except Exception as error:
            if isinstance(error, local_auth.AuthError):
                raise
            # DB unreachable: fall through to file-only so bootstrap/setup
            # tetap jalan lokal. Deploy server tetap gagal loudly di migrate.
            db_ok = False

    # Keep a local mirror for disaster recovery after the authoritative write.
    try:
        store = local_auth.load_users(config_dir / "users.json")
    except local_auth.AuthError:
        store = local_auth.empty_store()
    if db_mode and db_ok and username.strip() in store.get("users", {}):
        pass
    else:
        local_auth.create_user(store, username, password, role=role)
    local_auth.save_users(config_dir / "users.json", store)


def authenticate_bridge(config_dir: Path, username: str, password: str) -> bool:
    if not is_db_configured():
        return _file_authenticate(config_dir, username, password)
    # PostgreSQL first; fall back to file mirror when DB unreachable so local
    # E2E keeps working. Deploy entrypoint still fails loudly on migrate.
    try:
        from finrl.db import repositories as repo

        with _session() as session:
            ok = repo.authenticate_db(session, username, password)
        if ok:
            return True
        # If user table empty (fresh DB), fall back to file once.
        with _session() as session:
            if repo.count_users_db(session) == 0:
                return _file_authenticate(config_dir, username, password)
        return False
    except Exception:
        return _file_authenticate(config_dir, username, password)


def _file_user_count(config_dir: Path) -> int:
    from finrl.integrations import auth as local_auth

    try:
        return local_auth.user_count(local_auth.load_users(config_dir / "users.json"))
    except local_auth.AuthError:
        return 0


def _file_authenticate(config_dir: Path, username: str, password: str) -> bool:
    from finrl.integrations import auth as local_auth

    try:
        store = local_auth.load_users(config_dir / "users.json")
    except local_auth.AuthError:
        return False
    return local_auth.authenticate(store, username, password)


def effective_storage_mode(config_dir: Path | None = None) -> str:
    """Human-readable storage state: DB, file, or file-fallback when DB is down."""
    if not is_db_configured():
        return "File lokal"
    try:
        from finrl.db.database import wait_for_db

        wait_for_db(timeout_seconds=3)
        return "PostgreSQL"
    except Exception:
        return "File lokal (DB tidak reachable)"


def user_count_bridge(config_dir: Path) -> int:
    if not is_db_configured():
        return _file_user_count(config_dir)
    try:
        from finrl.db import repositories as repo

        with _session() as session:
            count = repo.count_users_db(session)
        if count == 0:
            return _file_user_count(config_dir) or 0
        return count
    except Exception:
        # DB configured but unreachable (e.g. local dev without Postgres):
        # fall back to file mirror so login/E2E keeps working. Writes still
        # mirror to file; server deploy uses entrypoint migrate which fails
        # loudly instead of silently falling back.
        return _file_user_count(config_dir)


def get_user_role_bridge(config_dir: Path, username: str) -> str | None:
    """Return the stored role for ``username`` or None when unknown."""
    username = (username or "").strip()
    if not username:
        return None
    if not is_db_configured():
        from finrl.integrations import auth as local_auth

        try:
            store = local_auth.load_users(config_dir / "users.json")
        except local_auth.AuthError:
            return None
        record = store.get("users", {}).get(username)
        if isinstance(record, dict):
            role = record.get("role", "admin")
            return str(role) if role else "admin"
        return None
    try:
        from finrl.db import repositories as repo

        with _session() as session:
            user = repo.get_user(session, username)
            if user is not None:
                return user.role or "admin"
            if repo.count_users_db(session) == 0:
                from finrl.integrations import auth as local_auth

                try:
                    store = local_auth.load_users(config_dir / "users.json")
                except local_auth.AuthError:
                    return None
                record = store.get("users", {}).get(username)
                if isinstance(record, dict):
                    return str(record.get("role", "admin") or "admin")
            return None
    except Exception:
        from finrl.integrations import auth as local_auth

        try:
            store = local_auth.load_users(config_dir / "users.json")
        except local_auth.AuthError:
            return None
        record = store.get("users", {}).get(username)
        if isinstance(record, dict):
            return str(record.get("role", "admin") or "admin")
        return None


def list_users_bridge(config_dir: Path) -> list[dict[str, str]]:
    """List users as ``[{username, role}]`` from DB (authoritative) or file."""
    if not is_db_configured():
        from finrl.integrations import auth as local_auth

        try:
            store = local_auth.load_users(config_dir / "users.json")
        except local_auth.AuthError:
            return []
        return [
            {"username": name, "role": str(rec.get("role", "admin") or "admin")}
            for name, rec in store.get("users", {}).items()
            if isinstance(rec, dict)
        ]
    try:
        from finrl.db import repositories as repo

        with _session() as session:
            users = repo.list_users(session)
            if users:
                return [{"username": u.username, "role": u.role or "admin"} for u in users]
        from finrl.integrations import auth as local_auth

        try:
            store = local_auth.load_users(config_dir / "users.json")
        except local_auth.AuthError:
            return []
        return [
            {"username": name, "role": str(rec.get("role", "admin") or "admin")}
            for name, rec in store.get("users", {}).items()
            if isinstance(rec, dict)
        ]
    except Exception:
        from finrl.integrations import auth as local_auth

        try:
            store = local_auth.load_users(config_dir / "users.json")
        except local_auth.AuthError:
            return []
        return [
            {"username": name, "role": str(rec.get("role", "admin") or "admin")}
            for name, rec in store.get("users", {}).items()
            if isinstance(rec, dict)
        ]


def change_password_bridge(
    config_dir: Path, username: str, current_password: str, new_password: str
) -> None:
    """Change password in DB (authoritative) and mirror to file when in DB mode."""
    from finrl.integrations import auth as local_auth

    username = (username or "").strip()
    if not username:
        raise local_auth.AuthError("Username wajib diisi.")
    if not is_db_configured():
        store = local_auth.load_users(config_dir / "users.json")
        local_auth.change_password(store, username, current_password, new_password)
        local_auth.save_users(config_dir / "users.json", store)
        return
    try:
        from finrl.db import repositories as repo

        with _session() as session:
            repo.change_password_db(session, username, current_password, new_password)
            session.commit()
    except Exception as error:
        if isinstance(error, local_auth.AuthError):
            raise
        # DB unreachable: fall back to file-only so user tidak terkunci.
        # AuthError dari file (password lama salah) tetap dipropagasi di bawah.
        try:
            store = local_auth.load_users(config_dir / "users.json")
            local_auth.change_password(store, username, current_password, new_password)
            local_auth.save_users(config_dir / "users.json", store)
            return
        except local_auth.AuthError:
            raise
        raise
    # Mirror to file so disaster recovery keeps working.
    try:
        try:
            store = local_auth.load_users(config_dir / "users.json")
        except local_auth.AuthError:
            store = local_auth.empty_store()
        users = store.setdefault("users", {})
        if username in users and isinstance(users[username], dict):
            try:
                local_auth.change_password(store, username, current_password, new_password)
            except local_auth.AuthError:
                # File copy out of sync (different password): overwrite hash from DB.
                with _session() as session:
                    from finrl.db import repositories as repo

                    user = repo.get_user(session, username)
                    if user is not None:
                        users[username] = {
                            "algorithm": user.algorithm,
                            "iterations": user.iterations,
                            "salt": user.salt_hex,
                            "hash": user.hash_hex,
                            "role": user.role,
                        }
            local_auth.save_users(config_dir / "users.json", store)
    except Exception:
        pass


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
