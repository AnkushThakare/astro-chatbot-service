from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BirthDetails(BaseModel):
    name: str | None = None
    latitude: float
    longitude: float
    birth_datetime: datetime
    ayanamsha: str = "LAHIRI"
    house_system: str = "W"


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    include_astrology_context: bool = True
    birth_details: BirthDetails | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    model: str
    memory_turns_used: int
    retrieved_document_ids: list[int]
    astrology_summary: str | None = None


class MemoryTurnRead(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime


class MemoryListResponse(BaseModel):
    session_id: str
    turns: list[MemoryTurnRead]


class KnowledgeDocumentCreate(BaseModel):
    source: str
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)


class KnowledgeDocumentBatchCreate(BaseModel):
    documents: list[KnowledgeDocumentCreate]


class RetrievalMatch(BaseModel):
    id: int
    source: str
    title: str
    excerpt: str
    score: int
    tags: list[str]


class RetrievalResponse(BaseModel):
    matches: list[RetrievalMatch]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class FineTuneDatasetRequest(BaseModel):
    session_ids: list[str] = Field(default_factory=list)
    limit_sessions: int = Field(default=50, ge=1, le=1000)
    system_prompt: str | None = None


class FineTuneJobRequest(BaseModel):
    training_file: str
    base_model: str = "llama-3.3-70b-versatile"
    suffix: str = "astro-chatbot-v1"
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
