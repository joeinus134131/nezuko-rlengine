"""Initial schema: users, provider_profiles, app_settings, secrets, experiments."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("username", sa.String(64), nullable=False),
        sa.Column("algorithm", sa.String(32), nullable=False, server_default="pbkdf2_hmac_sha256"),
        sa.Column("iterations", sa.Integer(), nullable=False, server_default="240000"),
        sa.Column("salt_hex", sa.String(128), nullable=False),
        sa.Column("hash_hex", sa.String(256), nullable=False),
        sa.Column("role", sa.String(32), nullable=False, server_default="admin"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)

    op.create_table(
        "provider_profiles",
        sa.Column("id", sa.String(128), primary_key=True),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("tier", sa.String(32), nullable=False),
        sa.Column("base_url", sa.Text(), nullable=False),
        sa.Column("quote_path", sa.Text(), nullable=False, server_default=""),
        sa.Column("history_path", sa.Text(), nullable=False, server_default=""),
        sa.Column("account_path", sa.Text(), nullable=False, server_default=""),
        sa.Column("auth_type", sa.String(32), nullable=False, server_default="bearer"),
        sa.Column("auth_header", sa.String(128), nullable=False, server_default="Authorization"),
        sa.Column("auth_prefix", sa.String(64), nullable=False, server_default="Bearer "),
        sa.Column("secret_name", sa.String(128), nullable=False, server_default=""),
        sa.Column("strip_suffix", sa.String(32), nullable=False, server_default=".JK"),
        sa.Column("timeout_seconds", sa.Integer(), nullable=False, server_default="15"),
        sa.Column("stale_after_seconds", sa.Integer(), nullable=False, server_default="120"),
        sa.Column("quote_root", sa.Text(), nullable=False, server_default=""),
        sa.Column("history_root", sa.Text(), nullable=False, server_default=""),
        sa.Column("account_root", sa.Text(), nullable=False, server_default=""),
        sa.Column("quote_mapping", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("history_mapping", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("history_params", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "app_settings",
        sa.Column("key", sa.String(128), primary_key=True),
        sa.Column("value", postgresql.JSONB(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "secrets",
        sa.Column("name", sa.String(128), primary_key=True),
        sa.Column("ciphertext", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "experiments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(256), nullable=False, server_default=""),
        sa.Column("tickers", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("data_source", sa.String(64), nullable=False, server_default="yahoofinance"),
        sa.Column("interval", sa.String(16), nullable=False, server_default="1D"),
        sa.Column("train_start", sa.String(32), nullable=False, server_default=""),
        sa.Column("train_end", sa.String(32), nullable=False, server_default=""),
        sa.Column("test_start", sa.String(32), nullable=False, server_default=""),
        sa.Column("test_end", sa.String(32), nullable=False, server_default=""),
        sa.Column("indicators", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("use_vix", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("drl_lib", sa.String(64), nullable=False, server_default=""),
        sa.Column("model_name", sa.String(64), nullable=False, server_default=""),
        sa.Column("model_path", sa.Text(), nullable=False, server_default=""),
        sa.Column("timesteps", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("agent_params", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("universe", sa.String(128), nullable=False, server_default=""),
        sa.Column("risk_free_rate", sa.Float(), nullable=False, server_default="0.06"),
        sa.Column("buy_cost_pct", sa.Float(), nullable=False, server_default="0.0016"),
        sa.Column("sell_cost_pct", sa.Float(), nullable=False, server_default="0.0035"),
        sa.Column("lot_size", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("stop_loss_pct", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("status", sa.String(32), nullable=False, server_default="created"),
        sa.Column("metrics", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_by", sa.String(64), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("experiments")
    op.drop_table("secrets")
    op.drop_table("app_settings")
    op.drop_table("provider_profiles")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")
