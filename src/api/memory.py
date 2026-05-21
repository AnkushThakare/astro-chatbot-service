from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.auth.jwt import AuthenticatedUser, get_current_user
from src.core.logging import get_logger
from src.db.repositories.conversations import ConversationRepository
from src.db.repositories.users import UserRepository
from src.db.session import get_db

router = APIRouter()
logger = get_logger(__name__)


def _resolve_internal_user_id(
    user_repo: UserRepository,
    current_user: AuthenticatedUser,
) -> int | None:
    user = user_repo.get_by_external_id(current_user.user_id)
    return user.id if user is not None else None


@router.get("/chat/memories")
def list_memories(
    db: Session = Depends(get_db),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, Any]:
    """List all memories for the authenticated user."""
    user_repo = UserRepository(db)
    internal_id = _resolve_internal_user_id(user_repo, current_user)
    if internal_id is None:
        return {"memories": []}
    repo = ConversationRepository(db)
    facts = repo.list_facts_for_user(internal_id)
    return {
        "memories": [
            {
                "id": fact.id,
                "key": fact.fact_key,
                "value": fact.fact_value,
                "updated_at": fact.updated_at.isoformat() if fact.updated_at else None,
            }
            for fact in facts
        ]
    }


@router.delete("/chat/memories/{memory_id}")
def delete_memory(
    memory_id: int,
    db: Session = Depends(get_db),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, str]:
    """Delete a specific memory by ID."""
    user_repo = UserRepository(db)
    internal_id = _resolve_internal_user_id(user_repo, current_user)
    if internal_id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")
    repo = ConversationRepository(db)
    deleted = repo.delete_fact_by_id(memory_id, user_id=internal_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")
    return {"status": "deleted"}


@router.delete("/chat/memories")
def clear_all_memories(
    db: Session = Depends(get_db),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Clear all memories for the authenticated user."""
    user_repo = UserRepository(db)
    internal_id = _resolve_internal_user_id(user_repo, current_user)
    if internal_id is None:
        return {"deleted_count": 0}
    repo = ConversationRepository(db)
    count = repo.delete_all_facts_for_user(internal_id)
    return {"deleted_count": count}
