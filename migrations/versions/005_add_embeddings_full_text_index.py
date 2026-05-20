"""add postgres full text index for embeddings

Revision ID: 005_add_embeddings_full_text_index
Revises: 004_add_pgvector_support
Create Date: 2026-05-16
"""

from __future__ import annotations

from alembic import op


revision = "005_add_embeddings_full_text_index"
down_revision = "004_add_pgvector_support"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_embeddings_content_tsv_simple "
        "ON embeddings USING gin (to_tsvector('simple', content))"
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute("DROP INDEX IF EXISTS ix_embeddings_content_tsv_simple")
