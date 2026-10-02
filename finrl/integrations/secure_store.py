"""Store credentials: Postgres (encrypted) when DATABASE_URL set, else OS keyring."""
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


def _db_configured() -> bool:
    try:
        from finrl.db.database import is_db_configured

        return is_db_configured()
    except Exception:
        return False


def save_secret(name: str, value: str) -> None:
    if not value:
        raise ValueError("Secret tidak boleh kosong.")
    if _db_configured():
        try:
            from finrl.db.database import session_factory
            from finrl.db.repositories import save_secret_db

            with session_factory()() as session:
                save_secret_db(session, name, value)
                session.commit()
            # Mirror to keyring for local fallback; ignore failures on server.
            try:
                _backend().set_password(SERVICE_NAME, name, value)
            except Exception:
                pass
            return
        except Exception as error:
            raise RuntimeError(
                "Secure vault database gagal menyimpan secret. Periksa DATABASE_URL, "
                "APP_SECRET_KEY, dan migrasi tabel secrets."
            ) from error
    try:
        _backend().set_password(SERVICE_NAME, name, value)
    except Exception as error:
        raise RuntimeError(
            "Secure vault tidak tersedia. Untuk server/container, konfigurasi DATABASE_URL "
            "dan APP_SECRET_KEY; OS keyring hanya digunakan untuk instalasi desktop."
        ) from error


def load_secret(name: str) -> str | None:
    if _db_configured():
        try:
            from finrl.db.database import session_factory
            from finrl.db.repositories import load_secret_db

            with session_factory()() as session:
                value = load_secret_db(session, name)
            return value or None
        except Exception as error:
            raise RuntimeError(
                "Secure vault database gagal membaca secret. Periksa DATABASE_URL, "
                "APP_SECRET_KEY, dan migrasi tabel secrets."
            ) from error
    try:
        return _backend().get_password(SERVICE_NAME, name)
    except Exception:
        return None


def delete_secret(name: str) -> bool:
    keyring = _backend()
    try:
        keyring.delete_password(SERVICE_NAME, name)
    except keyring.errors.PasswordDeleteError:
        return False
    return True
