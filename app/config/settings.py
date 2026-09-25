from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application Config
    APP_NAME: str = "Universal AI Action Agent"
    APP_ENV: Literal["development", "testing", "production"] = "development"
    DEBUG: bool = True
    PORT: int = 8000

    # Database (Phase 2)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/action_agent_db"

    # Gemini AI (Phase 4)
    GEMINI_API_KEY: str = ""

    # Provider Options
    DEFAULT_RIDE_PROVIDER: Literal["mock", "uber"] = "mock"
    UBER_CLIENT_ID: str = ""
    UBER_CLIENT_SECRET: str = ""
    UBER_SANDBOX_MODE: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance to avoid reading disk repeatedly."""
    return Settings()
