from __future__ import annotations

import json

import pytest

from finrl.integrations import auth


def test_hash_and_verify_roundtrip() -> None:
    record = auth.hash_password("S3cur3-Pass!")
    assert record["algorithm"] == auth.ALGORITHM
    assert "S3cur3-Pass!" not in json.dumps(record)
    assert auth.verify_password("S3cur3-Pass!", record)
    assert not auth.verify_password("wrong-password", record)


def test_hash_uses_unique_salt_per_call() -> None:
    first = auth.hash_password("same-password")
    second = auth.hash_password("same-password")
    assert first["salt"] != second["salt"]
    assert first["hash"] != second["hash"]


def test_verify_rejects_malformed_record() -> None:
    assert not auth.verify_password("x", {})
    assert not auth.verify_password("x", {"salt": "zz", "hash": "zz"})
    assert not auth.verify_password("x", {"salt": "00", "hash": "00", "iterations": 0})


def test_create_user_and_authenticate() -> None:
    store = auth.empty_store()
    auth.create_user(store, "admin", "strong-pass-1", role="admin")
    assert auth.user_count(store) == 1
    assert auth.authenticate(store, "admin", "strong-pass-1")
    assert not auth.authenticate(store, "admin", "nope")
    assert not auth.authenticate(store, "ghost", "strong-pass-1")


def test_create_user_rejects_duplicate_and_weak_password() -> None:
    store = auth.empty_store()
    auth.create_user(store, "admin", "strong-pass-1")
    with pytest.raises(auth.AuthError):
        auth.create_user(store, "admin", "another-pass-2")
    with pytest.raises(auth.AuthError):
        auth.create_user(store, "user2", "short")
    with pytest.raises(auth.AuthError):
        auth.create_user(store, "bad name", "strong-pass-1")


def test_save_and_load_store_roundtrip(tmp_path) -> None:
    path = tmp_path / "configs" / "users.json"
    store = auth.empty_store()
    auth.create_user(store, "admin", "strong-pass-1")
    auth.save_users(path, store)

    loaded = auth.load_users(path)
    assert auth.user_count(loaded) == 1
    assert auth.authenticate(loaded, "admin", "strong-pass-1")


def test_load_missing_store_is_empty(tmp_path) -> None:
    store = auth.load_users(tmp_path / "missing.json")
    assert auth.user_count(store) == 0


def test_load_invalid_store_raises(tmp_path) -> None:
    path = tmp_path / "users.json"
    path.write_text(json.dumps({"users": "not-a-dict"}), encoding="utf-8")
    with pytest.raises(auth.AuthError):
        auth.load_users(path)


def test_change_password_requires_current_password() -> None:
    store = auth.empty_store()
    auth.create_user(store, "admin", "strong-pass-1")
    auth.change_password(store, "admin", "strong-pass-1", "new-strong-pass-2")
    assert auth.authenticate(store, "admin", "new-strong-pass-2")
    assert not auth.authenticate(store, "admin", "strong-pass-1")
    with pytest.raises(auth.AuthError):
        auth.change_password(store, "admin", "wrong-current", "another-pass-3")
