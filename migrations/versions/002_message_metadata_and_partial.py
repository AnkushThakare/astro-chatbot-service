"""add message metadata columns

Revision ID: 002_message_metadata_and_partial
Revises: 001_initial
Create Date: 2026-05-13
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "002_message_metadata_and_partial"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("prompt_versions", sa.JSON(), nullable=True))
    op.add_column("messages", sa.Column("model_used", sa.String(length=128), nullable=True))
    op.add_column("messages", sa.Column("route_taken", sa.String(length=32), nullable=True))
    op.add_column("messages", sa.Column("tool_called", sa.String(length=64), nullable=True))
    op.add_column("messages", sa.Column("variant_id", sa.String(length=32), nullable=True))
    op.add_column("messages", sa.Column("total_tokens_input", sa.Integer(), nullable=True))
    op.add_column("messages", sa.Column("total_tokens_output", sa.Integer(), nullable=True))
    op.add_column("messages", sa.Column("latency_ms", sa.Integer(), nullable=True))
    op.add_column(
        "messages",
        sa.Column("partial", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("messages", "partial")
    op.drop_column("messages", "latency_ms")
    op.drop_column("messages", "total_tokens_output")
    op.drop_column("messages", "total_tokens_input")
    op.drop_column("messages", "variant_id")
    op.drop_column("messages", "tool_called")
    op.drop_column("messages", "route_taken")
    op.drop_column("messages", "model_used")
    op.drop_column("messages", "prompt_versions")
