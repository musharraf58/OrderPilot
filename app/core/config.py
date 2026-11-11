from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables (and .env locally)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "postgresql+asyncpg://orderpilot:orderpilot@localhost:5432/orderpilot"
    GEMINI_API_KEY: SecretStr = SecretStr("")
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"
    STORE_API_URL: str = "http://localhost:8001"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # Agent and conversations.
    AGENT_MAX_STEPS: int = 6
    AGENT_TIMEOUT_S: float = 60.0
    HISTORY_WINDOW: int = 20  # messages sent to the model; everything is still stored
    MAX_MESSAGE_CHARS: int = 1000

    # Write actions (proposed by the model, confirmed by the customer through the API).
    AGENT_ALLOW_WRITES: bool = True
    ACTION_TTL_S: float = 600.0

    # Protection (in-memory, per process).
    AUTH_CACHE_TTL_S: float = 60.0
    RATE_LIMIT_MESSAGES: int = 20
    RATE_LIMIT_WINDOW_S: float = 600.0
    ACTION_RATE_LIMIT: int = 20  # confirm/cancel calls per customer per window


@lru_cache
def get_settings() -> Settings:
    return Settings()
