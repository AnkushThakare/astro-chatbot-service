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
    GROQ_BASE_URL: str = Field(
        default="https://api.groq.com/openai/v1",
        alias="GROQ_BASE_URL",
    )
    GROQ_TIMEOUT_SECONDS: int = Field(default=30, alias="GROQ_TIMEOUT_SECONDS")
    LANGFUSE_SECRET_KEY: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    LANGFUSE_PUBLIC_KEY: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    LANGFUSE_BASE_URL: str | None = Field(
        default="https://us.cloud.langfuse.com",
        alias="LANGFUSE_BASE_URL",
    )
    JWT_SECRET_KEY: str = Field(default="development-insecure-secret", alias="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = Field(default="HS256", alias="JWT_ALGORITHM")
    JWT_ISSUER: str | None = Field(default=None, alias="JWT_ISSUER")
    JWT_AUDIENCE: str | None = Field(default=None, alias="JWT_AUDIENCE")
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
