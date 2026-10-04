from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("APP_SECRET_KEY", "test-only-key-please-override-in-prod-123456")

from finrl.db import repositories as repo
from finrl.db import bridge
from finrl.db.database import _normalize_local_sslmode
from finrl.db.crypto import decrypt_secret, encrypt_secret
from finrl.db.models import Base


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as sess:
        yield sess


def test_user_create_and_authenticate(session) -> None:
    repo.create_user_db(session, "admin", "strong-pass-1")
    session.commit()
    assert repo.authenticate_db(session, "admin", "strong-pass-1")
    assert not repo.authenticate_db(session, "admin", "salah")
    assert not repo.authenticate_db(session, "ghost", "strong-pass-1")


def test_settings_roundtrip(session) -> None:
    repo.set_setting(session, "ai_research", {"model": "m"})
    session.commit()
    assert repo.get_setting(session, "ai_research") == {"model": "m"}


def test_secret_encrypt_roundtrip() -> None:
    token = encrypt_secret("abc-123")
    assert token != "abc-123"
    assert decrypt_secret(token) == "abc-123"


def test_provider_upsert_and_list(session) -> None:
    repo.upsert_provider(
        session,
        {
            "id": "p1",
            "name": "P",
            "kind": "market_data",
            "tier": "licensed",
            "base_url": "https://example.com",
        },
    )
    session.commit()
    rows = repo.list_provider_dicts(session)
    assert [r["id"] for r in rows] == ["p1"]


def test_experiment_create_and_list(session) -> None:
    repo.create_experiment(
        session, {"name": "e1", "tickers": ["BBCA.JK"], "created_by": "admin"}
    )
    session.commit()
    assert repo.list_experiments(session)[0].name == "e1"


def test_local_database_sslmode_falls_back_to_prefer() -> None:
    url = "postgresql://user:pass@127.0.0.1:5432/db?sslmode=require"
    assert "sslmode=prefer" in _normalize_local_sslmode(url)


def test_remote_database_sslmode_is_preserved() -> None:
    url = "postgresql://user:pass@db.example.com:5432/db?sslmode=require"
    assert _normalize_local_sslmode(url) == url


def test_config_write_does_not_hide_database_failure(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bridge, "is_db_configured", lambda: True)
    monkeypatch.setattr(
        bridge,
        "_session",
        lambda: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )

    with pytest.raises(RuntimeError, match="Gagal menyimpan konfigurasi"):
        bridge.write_app_config("telegram_monitor.json", {"chat_id": "1"}, tmp_path)
    assert not (tmp_path / "telegram_monitor.json").exists()


def test_user_count_falls_back_to_file_when_database_is_unavailable(
    monkeypatch, tmp_path
) -> None:
    # Kontrak baru: DB tidak reachable -> fallback ke file mirror agar login
    # lokal tetap jalan. Deploy server tetap gagal loudly di migrate_db.
    monkeypatch.setattr(bridge, "is_db_configured", lambda: True)
    monkeypatch.setattr(
        bridge,
        "_session",
        lambda: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )

    assert bridge.user_count_bridge(tmp_path) == 0


def test_create_user_falls_back_to_file_when_database_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bridge, "is_db_configured", lambda: True)
    monkeypatch.setattr(
        bridge,
        "_session",
        lambda: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )

    bridge.create_user_bridge(tmp_path, "admin", "strong-pass-1")
    assert (tmp_path / "users.json").exists()
    assert bridge.authenticate_bridge(tmp_path, "admin", "strong-pass-1")


def test_resolve_docker_host(monkeypatch) -> None:
    from finrl.db.database import _resolve_docker_host

    monkeypatch.setenv("RUNNING_IN_DOCKER", "1")
    url = "postgresql://finrlab_app:secret@127.0.0.1:5432/finrlab?sslmode=require"
    resolved = _resolve_docker_host(url)
    assert resolved == "postgresql://finrlab_app:secret@host.docker.internal:5432/finrlab?sslmode=require"

    url_localhost = "postgresql://user:pass@localhost:5432/db"
    assert _resolve_docker_host(url_localhost) == "postgresql://user:pass@host.docker.internal:5432/db"

    url_external = "postgresql://user:pass@postgres.internal:5432/db"
    assert _resolve_docker_host(url_external) == url_external
