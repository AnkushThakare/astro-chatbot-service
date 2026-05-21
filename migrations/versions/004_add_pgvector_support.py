"""add pgvector support for embeddings

Revision ID: 004_add_pgvector_support
Revises: 003_drop_birth_charts_table
Create Date: 2026-05-14
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from src.core.config import settings

try:
    from pgvector.sqlalchemy import Vector as PgVector
except ImportError:  # pragma: no cover - optional dependency during migration authoring
    PgVector = None


revision = "004_add_pgvector_support"
down_revision = "003_drop_birth_charts_table"
branch_labels = None
depends_on = None
_IVFFLAT_MAX_DIMENSIONS = 2000


def upgrade() -> None:
    bind = op.get_bind()
    dialect_name = bind.dialect.name

    if dialect_name == "postgresql" and PgVector is not None:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        op.add_column(
            "embeddings",
            sa.Column("vector_pg", PgVector(settings.RAG_EMBEDDING_DIMENSIONS), nullable=True),
        )
        if settings.RAG_EMBEDDING_DIMENSIONS <= _IVFFLAT_MAX_DIMENSIONS:
            op.execute(
                "CREATE INDEX IF NOT EXISTS ix_embeddings_vector_pg "
                "ON embeddings USING ivfflat (vector_pg vector_cosine_ops) WITH (lists = 100)"
            )
    else:
        op.add_column("embeddings", sa.Column("vector_pg", sa.Text(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    dialect_name = bind.dialect.name
    if dialect_name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_embeddings_vector_pg")
    op.drop_column("embeddings", "vector_pg")
