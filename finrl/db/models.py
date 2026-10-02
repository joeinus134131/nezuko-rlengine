"""SQLAlchemy models. Mirrors the legacy JSON stores 1:1 + experiment audit."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _json_type():
    # JSONB on Postgres, plain JSON on SQLite (unit tests).
    try:
        from sqlalchemy.dialects.postgresql import dialect as pg_dialect  # noqa: F401
        return JSONB().with_variant(JSON(), "sqlite")
    except Exception:  # pragma: no cover
        return JSON()


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    algorithm: Mapped[str] = mapped_column(String(32), nullable=False, default="pbkdf2_hmac_sha256")
    iterations: Mapped[int] = mapped_column(Integer, nullable=False, default=240000)
    salt_hex: Mapped[str] = mapped_column(String(128), nullable=False)
    hash_hex: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="admin")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ProviderProfileRow(Base):
    __tablename__ = "provider_profiles"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    tier: Mapped[str] = mapped_column(String(32), nullable=False)
    base_url: Mapped[str] = mapped_column(Text, nullable=False)
    quote_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    history_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    account_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    auth_type: Mapped[str] = mapped_column(String(32), nullable=False, default="bearer")
    auth_header: Mapped[str] = mapped_column(String(128), nullable=False, default="Authorization")
    auth_prefix: Mapped[str] = mapped_column(String(64), nullable=False, default="Bearer ")
    secret_name: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    strip_suffix: Mapped[str] = mapped_column(String(32), nullable=False, default=".JK")
    timeout_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=15)
    stale_after_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=120)
    quote_root: Mapped[str] = mapped_column(Text, nullable=False, default="")
    history_root: Mapped[str] = mapped_column(Text, nullable=False, default="")
    account_root: Mapped[str] = mapped_column(Text, nullable=False, default="")
    quote_mapping = mapped_column(_json_type(), nullable=False, default=dict)
    history_mapping = mapped_column(_json_type(), nullable=False, default=dict)
    history_params = mapped_column(_json_type(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class AppSetting(Base):
    """Generic key-value store: ai_research, paper_trading, telegram_monitor,
    active_market_data, active_broker, ... Value is JSON."""

    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value = mapped_column(_json_type(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Secret(Base):
    """Encrypted secrets (Fernet). Replaces OS keyring on server."""

    __tablename__ = "secrets"

    name: Mapped[str] = mapped_column(String(128), primary_key=True)
    ciphertext: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Experiment(Base):
    """Audit trail for training / backtest runs."""

    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    tickers = mapped_column(_json_type(), nullable=False, default=list)
    data_source: Mapped[str] = mapped_column(String(64), nullable=False, default="yahoofinance")
    interval: Mapped[str] = mapped_column(String(16), nullable=False, default="1D")
    train_start: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    train_end: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    test_start: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    test_end: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    indicators = mapped_column(_json_type(), nullable=False, default=list)
    use_vix: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    drl_lib: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    model_name: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    model_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    timesteps: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    agent_params = mapped_column(_json_type(), nullable=False, default=dict)
    universe: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    risk_free_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.06)
    buy_cost_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0016)
    sell_cost_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0035)
    lot_size: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    stop_loss_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="created")
    metrics = mapped_column(_json_type(), nullable=False, default=dict)
    created_by: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
