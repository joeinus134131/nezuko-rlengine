from __future__ import annotations

import pytest

from finrl.integrations import secure_store


def test_server_db_failure_does_not_fall_back_to_os_keyring(monkeypatch) -> None:
    monkeypatch.setattr(secure_store, "_db_configured", lambda: True)
    monkeypatch.setattr(
        "finrl.db.database.session_factory",
        lambda: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )
    monkeypatch.setattr(
        secure_store,
        "_backend",
        lambda: (_ for _ in ()).throw(AssertionError("must not use keyring")),
    )

    with pytest.raises(RuntimeError, match="Secure vault database gagal"):
        secure_store.save_secret("telegram_bot_token", "secret")


def test_server_db_read_failure_does_not_fall_back_to_os_keyring(monkeypatch) -> None:
    monkeypatch.setattr(secure_store, "_db_configured", lambda: True)
    monkeypatch.setattr(
        "finrl.db.database.session_factory",
        lambda: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )
    monkeypatch.setattr(
        secure_store,
        "_backend",
        lambda: (_ for _ in ()).throw(AssertionError("must not use keyring")),
    )

    with pytest.raises(RuntimeError, match="gagal membaca secret"):
        secure_store.load_secret("telegram_bot_token")
