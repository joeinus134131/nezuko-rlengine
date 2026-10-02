"""Fernet encryption for secrets stored in Postgres."""
from __future__ import annotations

import base64
import hashlib
import os


def _fernet():
    key_material = os.getenv("APP_SECRET_KEY", "").strip()
    if not key_material:
        return None
    try:
        from cryptography.fernet import Fernet
    except ImportError as error:
        raise RuntimeError("Paket cryptography belum terpasang.") from error
    # Accept a raw Fernet key or derive one from an arbitrary passphrase.
    try:
        return Fernet(key_material.encode())
    except Exception:  # noqa: BLE001 - fall back to derived key
        digest = hashlib.sha256(key_material.encode()).digest()
        return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_secret(plaintext: str) -> str:
    fernet = _fernet()
    if fernet is None:
        raise RuntimeError("APP_SECRET_KEY belum diset, tidak bisa enkripsi secret.")
    return fernet.encrypt(plaintext.encode()).decode()


def decrypt_secret(ciphertext: str) -> str:
    fernet = _fernet()
    if fernet is None:
        raise RuntimeError("APP_SECRET_KEY belum diset, tidak bisa dekripsi secret.")
    return fernet.decrypt(ciphertext.encode()).decode()


def is_encryption_configured() -> bool:
    return bool(os.getenv("APP_SECRET_KEY", "").strip())
