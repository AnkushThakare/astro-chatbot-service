"""add energy flow behavior tables

Revision ID: 006_add_energy_flow_tables
Revises: 005_add_embeddings_full_text_index
Create Date: 2026-05-18
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "006_add_energy_flow_tables"
down_revision = "005_add_embeddings_full_text_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "behavior_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("conversation_id", sa.Integer(), nullable=True),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=True),
        sa.Column("occurred_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_behavior_events_user_id"), "behavior_events", ["user_id"], unique=False)
    op.create_index(op.f("ix_behavior_events_conversation_id"), "behavior_events", ["conversation_id"], unique=False)
    op.create_index(op.f("ix_behavior_events_session_id"), "behavior_events", ["session_id"], unique=False)
    op.create_index(op.f("ix_behavior_events_event_type"), "behavior_events", ["event_type"], unique=False)
    op.create_index(op.f("ix_behavior_events_occurred_at"), "behavior_events", ["occurred_at"], unique=False)
    op.create_index(op.f("ix_behavior_events_created_at"), "behavior_events", ["created_at"], unique=False)

    op.create_table(
        "behavior_profiles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("scope_key", sa.String(length=160), nullable=False),
        sa.Column("scope_type", sa.String(length=16), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("conversation_id", sa.Integer(), nullable=True),
        sa.Column("session_id", sa.String(length=128), nullable=True),
        sa.Column("overall_alignment", sa.Integer(), nullable=False),
        sa.Column("stress_score", sa.Integer(), nullable=False),
        sa.Column("focus_score", sa.Integer(), nullable=False),
        sa.Column("emotional_drift_score", sa.Integer(), nullable=False),
        sa.Column("cognitive_overload_score", sa.Integer(), nullable=False),
        sa.Column("clarity_score", sa.Integer(), nullable=False),
        sa.Column("behavioral_consistency_score", sa.Integer(), nullable=False),
        sa.Column("emotional_state", sa.String(length=32), nullable=False),
        sa.Column("focus_state", sa.String(length=32), nullable=False),
        sa.Column("behavioral_state", sa.String(length=32), nullable=False),
        sa.Column("signal_count", sa.Integer(), nullable=False),
        sa.Column("summary_text", sa.Text(), nullable=True),
        sa.Column("signals_json", sa.JSON(), nullable=True),
        sa.Column("last_event_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("scope_key"),
    )
    op.create_index(op.f("ix_behavior_profiles_scope_key"), "behavior_profiles", ["scope_key"], unique=True)
    op.create_index(op.f("ix_behavior_profiles_scope_type"), "behavior_profiles", ["scope_type"], unique=False)
    op.create_index(op.f("ix_behavior_profiles_user_id"), "behavior_profiles", ["user_id"], unique=False)
    op.create_index(op.f("ix_behavior_profiles_conversation_id"), "behavior_profiles", ["conversation_id"], unique=False)
    op.create_index(op.f("ix_behavior_profiles_session_id"), "behavior_profiles", ["session_id"], unique=False)
    op.create_index(op.f("ix_behavior_profiles_last_event_at"), "behavior_profiles", ["last_event_at"], unique=False)
    op.create_index(op.f("ix_behavior_profiles_created_at"), "behavior_profiles", ["created_at"], unique=False)
    op.create_index(op.f("ix_behavior_profiles_updated_at"), "behavior_profiles", ["updated_at"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_behavior_profiles_updated_at"), table_name="behavior_profiles")
    op.drop_index(op.f("ix_behavior_profiles_created_at"), table_name="behavior_profiles")
    op.drop_index(op.f("ix_behavior_profiles_last_event_at"), table_name="behavior_profiles")
    op.drop_index(op.f("ix_behavior_profiles_session_id"), table_name="behavior_profiles")
    op.drop_index(op.f("ix_behavior_profiles_conversation_id"), table_name="behavior_profiles")
    op.drop_index(op.f("ix_behavior_profiles_user_id"), table_name="behavior_profiles")
    op.drop_index(op.f("ix_behavior_profiles_scope_type"), table_name="behavior_profiles")
    op.drop_index(op.f("ix_behavior_profiles_scope_key"), table_name="behavior_profiles")
    op.drop_table("behavior_profiles")

    op.drop_index(op.f("ix_behavior_events_created_at"), table_name="behavior_events")
    op.drop_index(op.f("ix_behavior_events_occurred_at"), table_name="behavior_events")
    op.drop_index(op.f("ix_behavior_events_event_type"), table_name="behavior_events")
    op.drop_index(op.f("ix_behavior_events_session_id"), table_name="behavior_events")
    op.drop_index(op.f("ix_behavior_events_conversation_id"), table_name="behavior_events")
    op.drop_index(op.f("ix_behavior_events_user_id"), table_name="behavior_events")
    op.drop_table("behavior_events")
