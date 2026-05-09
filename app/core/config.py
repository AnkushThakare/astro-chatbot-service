from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    DATABASE_URL: str = Field(default="sqlite:///./astro_chatbot.db", alias="DATABASE_URL")

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
            "You are an astrology assistant. Give direct, careful answers. "
            "Use retrieved knowledge and chart context when available, and say "
            "when information is uncertain."
        ),
        alias="DEFAULT_SYSTEM_PROMPT",
    )

    @property
    def is_production(self) -> bool:
        return self.APP_ENV.lower() == "production"

    @property
    def is_development(self) -> bool:
        return self.APP_ENV.lower() == "development"

    @property
    def database_connect_args(self) -> dict[str, object]:
        if self.DATABASE_URL.startswith("sqlite"):
            return {"check_same_thread": False}
        return {}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

