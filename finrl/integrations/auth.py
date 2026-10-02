"""Local user authentication for the IDN Maker FINRLAB dashboard.

Credentials are stored as salted PBKDF2-HMAC-SHA256 hashes in a JSON file
(``configs/users.json`` by default), which is excluded from version control.
No plaintext password is ever persisted or logged.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

ALGORITHM = "pbkdf2_hmac_sha256"
DEFAULT_ITERATIONS = 240_000
SALT_BYTES = 16
MIN_PASSWORD_LENGTH = 8
MAX_USERNAME_LENGTH = 64
_ALLOWED_USERNAME_CHARS = set("._-")


class AuthError(ValueError):
    """Raised when a credential or user-store operation is invalid."""


def hash_password(password: str, iterations: int = DEFAULT_ITERATIONS) -> dict[str, Any]:
    """Derive a salted password record suitable for JSON serialization."""
    if not password:
        raise AuthError("Password tidak boleh kosong.")
    salt = secrets.token_bytes(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return {
        "algorithm": ALGORITHM,
        "iterations": iterations,
        "salt": salt.hex(),
        "hash": digest.hex(),
    }


def verify_password(password: str, record: dict[str, Any]) -> bool:
    """Check ``password`` against a stored record using a constant-time compare."""
    if not isinstance(record, dict):
        return False
    salt_hex = record.get("salt")
    hash_hex = record.get("hash")
    if not isinstance(salt_hex, str) or not isinstance(hash_hex, str):
        return False
    try:
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        iterations = int(record.get("iterations", DEFAULT_ITERATIONS))
    except (TypeError, ValueError):
        return False
    if iterations <= 0:
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(digest, expected)


def _burn_time(password: str) -> None:
    """Perform a dummy derivation so unknown usernames cost similar time."""
    hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), b"\x00" * SALT_BYTES, DEFAULT_ITERATIONS
    )


def empty_store() -> dict[str, Any]:
    return {"version": 1, "users": {}}


def load_users(path: str | Path) -> dict[str, Any]:
    """Load the user store, returning an empty store when the file is absent."""
    path = Path(path)
    if not path.exists():
        return empty_store()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuthError(f"File user tidak dapat dibaca: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("users"), dict):
        raise AuthError("Struktur file user tidak valid.")
    payload.setdefault("version", 1)
    return payload


def save_users(path: str | Path, store: dict[str, Any]) -> Path:
    """Atomically persist the user store with owner-only permissions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        # Best-effort on filesystems that do not support POSIX permissions.
        pass
    tmp.replace(path)
    return path


def validate_username(username: str) -> str:
    username = (username or "").strip()
    if not username:
        raise AuthError("Username wajib diisi.")
    if len(username) > MAX_USERNAME_LENGTH:
        raise AuthError(f"Username maksimal {MAX_USERNAME_LENGTH} karakter.")
    if not all(char.isalnum() or char in _ALLOWED_USERNAME_CHARS for char in username):
        raise AuthError("Username hanya boleh berisi huruf, angka, titik, underscore, dan dash.")
    return username


def validate_password(password: str) -> None:
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Password minimal {MIN_PASSWORD_LENGTH} karakter.")


def create_user(
    store: dict[str, Any],
    username: str,
    password: str,
    *,
    role: str = "admin",
) -> dict[str, Any]:
    """Add a new user to ``store`` in memory. Persist with :func:`save_users`."""
    username = validate_username(username)
    validate_password(password)
    users = store.setdefault("users", {})
    if username in users:
        raise AuthError("Username sudah terdaftar.")
    record = hash_password(password)
    record.update({"role": role, "created_at": int(time.time())})
    users[username] = record
    return store


def change_password(
    store: dict[str, Any],
    username: str,
    current_password: str,
    new_password: str,
) -> dict[str, Any]:
    """Replace a user's password after verifying the current one."""
    username = (username or "").strip()
    users = store.get("users", {})
    record = users.get(username)
    if not isinstance(record, dict) or not verify_password(current_password, record):
        raise AuthError("Password lama tidak cocok.")
    validate_password(new_password)
    updated = hash_password(new_password)
    updated.update(
        {
            "role": record.get("role", "admin"),
            "created_at": record.get("created_at", int(time.time())),
        }
    )
    users[username] = updated
    return store


def authenticate(store: dict[str, Any], username: str, password: str) -> bool:
    """Return True only when the username exists and the password matches."""
    record = store.get("users", {}).get((username or "").strip())
    if not isinstance(record, dict):
        _burn_time(password or "")
        return False
    return verify_password(password or "", record)


def user_count(store: dict[str, Any]) -> int:
    return len(store.get("users", {}))
