"""Repository layer: DB-backed users / providers / settings / secrets / experiments."""
from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from finrl.db import crypto
from finrl.db.models import AppSetting, Experiment, ProviderProfileRow, Secret, User
from finrl.integrations import auth as local_auth


# ---------------------------------------------------------------- users ---
def list_users(session: Session) -> list[User]:
    return list(session.scalars(select(User).order_by(User.username)).all())


def get_user(session: Session, username: str) -> User | None:
    username = (username or "").strip()
    if not username:
        return None
    return session.scalar(select(User).where(User.username == username))


def create_user_db(session: Session, username: str, password: str, role: str = "admin") -> User:
    username = local_auth.validate_username(username)
    local_auth.validate_password(password)
    if get_user(session, username) is not None:
        raise local_auth.AuthError("Username sudah terdaftar.")
    record = local_auth.hash_password(password)
    user = User(
        username=username,
        algorithm=record["algorithm"],
        iterations=record["iterations"],
        salt_hex=record["salt"],
        hash_hex=record["hash"],
        role=role,
    )
    session.add(user)
    session.flush()
    return user


def authenticate_db(session: Session, username: str, password: str) -> bool:
    user = get_user(session, username)
    if user is None:
        local_auth._burn_time(password or "")
        return False
    return local_auth.verify_password(
        password or "",
        {"salt": user.salt_hex, "hash": user.hash_hex, "iterations": user.iterations},
    )


def change_password_db(
    session: Session, username: str, current_password: str, new_password: str
) -> None:
    user = get_user(session, username)
    if user is None or not authenticate_db(session, username, current_password):
        raise local_auth.AuthError("Password lama tidak cocok.")
    local_auth.validate_password(new_password)
    record = local_auth.hash_password(new_password)
    user.algorithm = record["algorithm"]
    user.iterations = record["iterations"]
    user.salt_hex = record["salt"]
    user.hash_hex = record["hash"]
    session.flush()


def count_users_db(session: Session) -> int:
    return len(list_users(session))


# ------------------------------------------------------------- settings ---
def get_setting(session: Session, key: str, default: Any = None) -> Any:
    row = session.get(AppSetting, key)
    return row.value if row is not None else default


def set_setting(session: Session, key: str, value: Any) -> None:
    row = session.get(AppSetting, key)
    if row is None:
        session.add(AppSetting(key=key, value=value))
    else:
        row.value = value
    session.flush()


# ------------------------------------------------------------ providers ---
def list_provider_dicts(session: Session) -> list[dict[str, Any]]:
    rows = session.scalars(select(ProviderProfileRow).order_by(ProviderProfileRow.id)).all()
    return [_row_to_dict(row) for row in rows]


def _row_to_dict(row: ProviderProfileRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "kind": row.kind,
        "tier": row.tier,
        "base_url": row.base_url,
        "quote_path": row.quote_path,
        "history_path": row.history_path,
        "account_path": row.account_path,
        "auth_type": row.auth_type,
        "auth_header": row.auth_header,
        "auth_prefix": row.auth_prefix,
        "secret_name": row.secret_name,
        "strip_suffix": row.strip_suffix,
        "timeout_seconds": row.timeout_seconds,
        "stale_after_seconds": row.stale_after_seconds,
        "quote_root": row.quote_root,
        "history_root": row.history_root,
        "account_root": row.account_root,
        "quote_mapping": row.quote_mapping,
        "history_mapping": row.history_mapping,
        "history_params": row.history_params,
    }


def upsert_provider(session: Session, payload: dict[str, Any]) -> None:
    from finrl.integrations.provider_adapter import ProviderProfile

    profile = ProviderProfile.from_dict(payload)  # validates
    row = session.get(ProviderProfileRow, profile.id)
    data = {
        "name": profile.name,
        "kind": profile.kind,
        "tier": profile.tier,
        "base_url": profile.base_url,
        "quote_path": profile.quote_path,
        "history_path": profile.history_path,
        "account_path": profile.account_path,
        "auth_type": profile.auth_type,
        "auth_header": profile.auth_header,
        "auth_prefix": profile.auth_prefix,
        "secret_name": profile.secret_name,
        "strip_suffix": profile.strip_suffix,
        "timeout_seconds": profile.timeout_seconds,
        "stale_after_seconds": profile.stale_after_seconds,
        "quote_root": profile.quote_root,
        "history_root": profile.history_root,
        "account_root": profile.account_root,
        "quote_mapping": dict(profile.quote_mapping),
        "history_mapping": dict(profile.history_mapping),
        "history_params": dict(profile.history_params),
    }
    if row is None:
        session.add(ProviderProfileRow(id=profile.id, **data))
    else:
        for key, value in data.items():
            setattr(row, key, value)
    session.flush()


def delete_provider(session: Session, profile_id: str) -> bool:
    row = session.get(ProviderProfileRow, profile_id)
    if row is None:
        return False
    session.delete(row)
    session.flush()
    return True


# -------------------------------------------------------------- secrets ---
def save_secret_db(session: Session, name: str, value: str) -> None:
    if not value:
        raise ValueError("Secret tidak boleh kosong.")
    ciphertext = crypto.encrypt_secret(value)
    row = session.get(Secret, name)
    if row is None:
        session.add(Secret(name=name, ciphertext=ciphertext))
    else:
        row.ciphertext = ciphertext
    session.flush()


def load_secret_db(session: Session, name: str) -> str | None:
    row = session.get(Secret, name)
    if row is None:
        return None
    return crypto.decrypt_secret(row.ciphertext)


# ----------------------------------------------------------- experiments ---
def create_experiment(session: Session, payload: dict[str, Any]) -> Experiment:
    allowed = {c.name for c in Experiment.__table__.columns if c.name != "id"}
    filtered = {k: v for k, v in payload.items() if k in allowed}
    row = Experiment(**filtered)
    session.add(row)
    session.flush()
    return row


def update_experiment(session: Session, experiment_id: str, **fields: Any) -> Experiment | None:
    from uuid import UUID

    row = session.get(Experiment, UUID(str(experiment_id)))
    if row is None:
        return None
    allowed = {c.name for c in Experiment.__table__.columns if c.name != "id"}
    for key, value in fields.items():
        if key in allowed:
            setattr(row, key, value)
    session.flush()
    return row


def list_experiments(session: Session, limit: int = 50) -> list[Experiment]:
    return list(
        session.scalars(
            select(Experiment).order_by(Experiment.created_at.desc()).limit(limit)
        ).all()
    )
