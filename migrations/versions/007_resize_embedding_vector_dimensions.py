"""resize pgvector embeddings column to configured dimensions

Revision ID: 007_resize_embedding_vector_dimensions
Revises: 006_add_energy_flow_tables
Create Date: 2026-05-20
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from src.core.config import settings

try:
    from pgvector.sqlalchemy import Vector as PgVector
except ImportError:  # pragma: no cover - optional dependency during migration authoring
    PgVector = None


revision = "007_resize_embedding_vector_dimensions"
down_revision = "006_add_energy_flow_tables"
branch_labels = None
depends_on = None

_PREVIOUS_DIMENSIONS = 128
_IVFFLAT_MAX_DIMENSIONS = 2000


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql" or PgVector is None:
        return

    op.execute("DROP INDEX IF EXISTS ix_embeddings_vector_pg")
    op.drop_column("embeddings", "vector_pg")
    op.add_column(
        "embeddings",
        sa.Column("vector_pg", PgVector(settings.RAG_EMBEDDING_DIMENSIONS), nullable=True),
    )
    if settings.RAG_EMBEDDING_DIMENSIONS <= _IVFFLAT_MAX_DIMENSIONS:
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_embeddings_vector_pg "
            "ON embeddings USING ivfflat (vector_pg vector_cosine_ops) WITH (lists = 100)"
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql" or PgVector is None:
        return

    op.execute("DROP INDEX IF EXISTS ix_embeddings_vector_pg")
    op.drop_column("embeddings", "vector_pg")
    op.add_column(
        "embeddings",
        sa.Column("vector_pg", PgVector(_PREVIOUS_DIMENSIONS), nullable=True),
    )
    if _PREVIOUS_DIMENSIONS <= _IVFFLAT_MAX_DIMENSIONS:
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_embeddings_vector_pg "
            "ON embeddings USING ivfflat (vector_pg vector_cosine_ops) WITH (lists = 100)"
        )
