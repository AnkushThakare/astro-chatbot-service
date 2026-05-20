"""drop local birth_charts table

Revision ID: 003_drop_birth_charts_table
Revises: 002_message_metadata_and_partial
Create Date: 2026-05-13
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "003_drop_birth_charts_table"
down_revision = "002_message_metadata_and_partial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_birth_charts_conversation_id", table_name="birth_charts")
    op.drop_index("ix_birth_charts_user_id", table_name="birth_charts")
    op.drop_table("birth_charts")


def downgrade() -> None:
    op.create_table(
        "birth_charts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("conversation_id", sa.Integer(), sa.ForeignKey("conversations.id"), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("timezone_name", sa.String(length=64), nullable=True),
        sa.Column("birth_datetime", sa.DateTime(), nullable=False),
        sa.Column("chart_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_birth_charts_user_id", "birth_charts", ["user_id"])
    op.create_index("ix_birth_charts_conversation_id", "birth_charts", ["conversation_id"])
