"""Store credentials in the operating-system credential vault via keyring."""
from __future__ import annotations

SERVICE_NAME = "finrl-workbench"


def _backend():
    try:
        import keyring
    except ImportError as error:
        raise RuntimeError(
            "Paket keyring belum terpasang. Jalankan: .venv311/bin/pip install keyring"
        ) from error
    return keyring


def save_secret(name: str, value: str) -> None:
    if not value:
        raise ValueError("Secret tidak boleh kosong.")
    _backend().set_password(SERVICE_NAME, name, value)


def load_secret(name: str) -> str | None:
    return _backend().get_password(SERVICE_NAME, name)


def delete_secret(name: str) -> bool:
    keyring = _backend()
    try:
        keyring.delete_password(SERVICE_NAME, name)
    except keyring.errors.PasswordDeleteError:
        return False
    return True
