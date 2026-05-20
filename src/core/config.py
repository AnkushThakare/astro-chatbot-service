from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = os.getenv("ENV_FILE", ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_NAME: str = Field(default="astro-chatbot-service", alias="APP_NAME")
    APP_ENV: str = Field(default="development", alias="APP_ENV")
    APP_VERSION: str = Field(default="0.1.0", alias="APP_VERSION")
    API_V1_PREFIX: str = Field(default="/api/v1", alias="API_V1_PREFIX")
    DOCS_ENABLED: bool = Field(default=True, alias="DOCS_ENABLED")
    APP_HOST: str = Field(default="0.0.0.0", alias="APP_HOST")
    APP_PORT: int = Field(default=8010, alias="APP_PORT")
    DATABASE_URL: str = Field(default="sqlite:///./astro_chatbot.db", alias="DATABASE_URL")
    ASYNC_DATABASE_URL: str | None = Field(default=None, alias="ASYNC_DATABASE_URL")
    LOG_LEVEL: str = Field(default="INFO", alias="LOG_LEVEL")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        alias="LOG_FORMAT",
    )
    LOG_FORMAT_TYPE: str = Field(default="auto", alias="LOG_FORMAT_TYPE")
    GROQ_API_KEY: str | None = Field(default=None, alias="GROQ_API_KEY")
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    GROQ_PLANNER_MODEL: str = Field(default="llama-3.1-8b-instant", alias="GROQ_PLANNER_MODEL")
    GROQ_BASE_URL: str = Field(
        default="https://api.groq.com/openai/v1",
        alias="GROQ_BASE_URL",
    )
    GROQ_TIMEOUT_SECONDS: int = Field(default=30, alias="GROQ_TIMEOUT_SECONDS")
    CORE_SERVICE_BASE_URL: str = Field(
        default="http://localhost:8000/api/v1",
        alias="CORE_SERVICE_BASE_URL",
    )
    CORE_SERVICE_TIMEOUT_SECONDS: int = Field(default=20, alias="CORE_SERVICE_TIMEOUT_SECONDS")
    BIRTH_DETAILS_CACHE_TTL_SECONDS: int = Field(
        default=300,
        alias="BIRTH_DETAILS_CACHE_TTL_SECONDS",
    )
    REDIS_URL: str | None = Field(default=None, alias="REDIS_URL")
    LANGFUSE_SECRET_KEY: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    LANGFUSE_PUBLIC_KEY: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    LANGFUSE_BASE_URL: str | None = Field(
        default="https://us.cloud.langfuse.com",
        alias="LANGFUSE_BASE_URL",
    )
    LANGFUSE_PROJECT_EVAL: str | None = Field(default=None, alias="LANGFUSE_PROJECT_EVAL")
    JWT_SECRET_KEY: str = Field(default="development-insecure-secret", alias="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = Field(default="HS256", alias="JWT_ALGORITHM")
    JWT_ISSUER: str | None = Field(default=None, alias="JWT_ISSUER")
    JWT_AUDIENCE: str | None = Field(default=None, alias="JWT_AUDIENCE")
    JWT_EXP_GRACE_SECONDS: int = Field(default=30, alias="JWT_EXP_GRACE_SECONDS")
    JWT_STREAM_MIN_REMAINING_SECONDS: int = Field(
        default=60,
        alias="JWT_STREAM_MIN_REMAINING_SECONDS",
    )
    ASTROLOGY_ENGINE_MODE: str = Field(default="remote", alias="ASTROLOGY_ENGINE_MODE")
    ASTROLOGY_SERVICE_URL: str = Field(
        default="http://localhost:8000/api/v1",
        alias="ASTROLOGY_SERVICE_URL",
    )
    ASTROLOGY_ENGINE_PATH: str = Field(
        default="../astrology-service/src",
        alias="ASTROLOGY_ENGINE_PATH",
    )
    MEMORY_WINDOW_SIZE: int = Field(default=8, alias="MEMORY_WINDOW_SIZE")
    RAG_TOP_K: int = Field(default=3, alias="RAG_TOP_K")
    FAST_RAG_TOP_K: int = Field(default=2, alias="FAST_RAG_TOP_K")
    RAG_CHUNK_SIZE_WORDS: int = Field(default=180, alias="RAG_CHUNK_SIZE_WORDS")
    RAG_CHUNK_OVERLAP_WORDS: int = Field(default=30, alias="RAG_CHUNK_OVERLAP_WORDS")
    RAG_EMBEDDING_DIMENSIONS: int = Field(default=128, alias="RAG_EMBEDDING_DIMENSIONS")
    RAG_EMBEDDING_PROVIDER: str = Field(default="local_hash", alias="RAG_EMBEDDING_PROVIDER")
    RAG_EMBEDDING_MODEL: str = Field(default="local-hash-v1", alias="RAG_EMBEDDING_MODEL")
    RAG_EMBEDDING_BASE_URL: str = Field(
        default="https://api.openai.com/v1",
        alias="RAG_EMBEDDING_BASE_URL",
    )
    RAG_EMBEDDING_API_KEY: str | None = Field(default=None, alias="RAG_EMBEDDING_API_KEY")
    RAG_EMBEDDING_TIMEOUT_SECONDS: int = Field(
        default=30,
        alias="RAG_EMBEDDING_TIMEOUT_SECONDS",
    )
    RAG_VECTOR_BACKEND: str = Field(default="auto", alias="RAG_VECTOR_BACKEND")
    RAG_TEXT_SEARCH_CONFIG: str = Field(default="simple", alias="RAG_TEXT_SEARCH_CONFIG")
    RAG_RERANKER_PROVIDER: str = Field(default="heuristic", alias="RAG_RERANKER_PROVIDER")
    RAG_RERANKER_MODEL: str = Field(default="heuristic-v1", alias="RAG_RERANKER_MODEL")
    RAG_RERANKER_BASE_URL: str = Field(
        default="https://api.groq.com/openai/v1",
        alias="RAG_RERANKER_BASE_URL",
    )
    RAG_RERANKER_API_KEY: str | None = Field(default=None, alias="RAG_RERANKER_API_KEY")
    RAG_RERANKER_TIMEOUT_SECONDS: int = Field(default=20, alias="RAG_RERANKER_TIMEOUT_SECONDS")
    SOFT_PRODUCT_RECOMMENDATIONS_ENABLED: bool = Field(
        default=True,
        alias="SOFT_PRODUCT_RECOMMENDATIONS_ENABLED",
    )
    TOOL_TIMEOUT_SECONDS: int = Field(default=12, alias="TOOL_TIMEOUT_SECONDS")
    IDEMPOTENCY_TTL_SECONDS: int = Field(default=300, alias="IDEMPOTENCY_TTL_SECONDS")
    RESPONSE_VARIANT_ID: str = Field(default="base", alias="RESPONSE_VARIANT_ID")
    INTERNAL_API_KEY: str = Field(default="development-internal-key", alias="INTERNAL_API_KEY")
    EVAL_MODE: bool = Field(default=False, alias="EVAL_MODE")
    GROQ_API_KEY_EVAL: str | None = Field(default=None, alias="GROQ_API_KEY_EVAL")

    # Multi-provider LLM config — defaults fall back to GROQ_* values
    PLANNER_LLM_PROVIDER: str = Field(default="groq", alias="PLANNER_LLM_PROVIDER")
    PLANNER_LLM_API_KEY: str | None = Field(default=None, alias="PLANNER_LLM_API_KEY")
    PLANNER_LLM_BASE_URL: str | None = Field(default=None, alias="PLANNER_LLM_BASE_URL")
    PLANNER_LLM_MODEL: str | None = Field(default=None, alias="PLANNER_LLM_MODEL")

    RESPONSE_LLM_PROVIDER: str = Field(default="groq", alias="RESPONSE_LLM_PROVIDER")
    RESPONSE_LLM_API_KEY: str | None = Field(default=None, alias="RESPONSE_LLM_API_KEY")
    RESPONSE_LLM_BASE_URL: str | None = Field(default=None, alias="RESPONSE_LLM_BASE_URL")
    RESPONSE_LLM_MODEL: str | None = Field(default=None, alias="RESPONSE_LLM_MODEL")
    DEFAULT_SYSTEM_PROMPT: str = Field(
        default=(
            "You are an astrology assistant. Be direct, careful, and explicit about "
            "uncertainty. Use retrieved knowledge, kundali context, and tools when available."
        ),
        alias="DEFAULT_SYSTEM_PROMPT",
    )

    @property
    def is_production(self) -> bool:
        return self.APP_ENV.lower() == "production"

    @property
    def prompts_dir(self) -> Path:
        return PROJECT_ROOT / "prompts"

    @property
    def rag_data_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "astrology_texts"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(
            self.LANGFUSE_SECRET_KEY and self.LANGFUSE_PUBLIC_KEY and self.LANGFUSE_BASE_URL
        )

    @property
    def sync_database_url(self) -> str:
        return self.DATABASE_URL

    @property
    def async_database_url(self) -> str:
        if self.ASYNC_DATABASE_URL:
            return self.ASYNC_DATABASE_URL
        if self.DATABASE_URL.startswith("sqlite:///"):
            return self.DATABASE_URL.replace("sqlite:///", "sqlite+aiosqlite:///")
        if self.DATABASE_URL.startswith("postgresql://"):
            return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
        if self.DATABASE_URL.startswith("postgresql+psycopg://"):
            return self.DATABASE_URL.replace("postgresql+psycopg://", "postgresql+asyncpg://")
        return self.DATABASE_URL


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
