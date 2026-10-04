"""Engine / session helpers. DATABASE_URL is the single source of truth.

Automatically loads ``PROJECT_ROOT/.env`` once per process so local
``streamlit run`` behaves like deploy (which uses compose ``env_file``).
Existing process environment always wins over ``.env`` values.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_LOADED = False


def _load_project_dotenv() -> None:
    """Load ``.env`` without overriding already-exported variables."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=env_path, override=False)
        return
    except Exception:
        pass
    # Minimal fallback parser so auth/DB still works without python-dotenv.
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if not key or key in os.environ:
                continue
            value = value.strip().strip('"').strip("'")
            # Skip docker-style `path:` lines accidentally left in .env.
            if not key.replace("_", "").isalnum():
                continue
            os.environ.setdefault(key, value)
    except OSError:
        pass


def _resolve_docker_host(url: str) -> str:
    """When running inside Docker, rewrite localhost/127.0.0.1 to host.docker.internal."""
    if not url or not (os.path.exists("/.dockerenv") or os.getenv("RUNNING_IN_DOCKER")):
        return url
    from urllib.parse import urlsplit, urlunsplit

    try:
        parts = urlsplit(url)
        if parts.hostname in ("127.0.0.1", "localhost"):
            netloc = parts.netloc
            auth_prefix = ""
            if "@" in netloc:
                auth_prefix, _ = netloc.rsplit("@", 1)
                auth_prefix += "@"
            port_suffix = f":{parts.port}" if parts.port else ""
            new_netloc = f"{auth_prefix}host.docker.internal{port_suffix}"
            return urlunsplit((parts.scheme, new_netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        pass
    return url


def _normalize_local_sslmode(url: str) -> str:
    """Allow local Postgres without weakening SSL policy for remote databases.

    Some shells retain an older ``DATABASE_URL`` with ``sslmode=require`` even
    after ``.env`` is updated. A loopback Postgres commonly has no TLS enabled;
    ``prefer`` attempts TLS first and safely falls back to the local socket.
    Remote hosts keep their explicitly configured SSL policy unchanged.
    """
    if not url:
        return url
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    try:
        parts = urlsplit(url)
        if parts.hostname not in {"127.0.0.1", "localhost", "::1"}:
            return url
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        if query.get("sslmode", "").lower() in {"require", "verify-ca", "verify-full"}:
            query["sslmode"] = "prefer"
        return urlunsplit(
            (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
        )
    except Exception:
        return url


def get_database_url() -> str:
    _load_project_dotenv()
    url = os.getenv("DATABASE_URL", "").strip()
    # Allow separate PG* vars as fallback for VPS .env style.
    if not url:
        host = os.getenv("POSTGRES_HOST", "").strip()
        if host:
            user = os.getenv("POSTGRES_USER", "nezu")
            password = os.getenv("POSTGRES_PASSWORD", "")
            db = os.getenv("POSTGRES_DB", "nezu")
            port = os.getenv("POSTGRES_PORT", "5432")
            url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return _resolve_docker_host(_normalize_local_sslmode(url))


def is_db_configured() -> bool:
    return bool(get_database_url())


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    url = get_database_url()
    if not url:
        raise RuntimeError("DATABASE_URL belum diset.")
    # psycopg2 driver; pool_pre_ping avoids stale connections after deploy.
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)


def session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def wait_for_db(timeout_seconds: int = 60) -> None:
    """Block until Postgres accepts connections (used by entrypoint)."""
    url = get_database_url()
    if not url:
        return
    engine = get_engine()
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return
        except Exception as error:  # noqa: BLE001 - retry on any connectivity error
            last_error = error
            time.sleep(2)
    raise RuntimeError(f"Postgres tidak reachable dalam {timeout_seconds}s: {last_error}")
