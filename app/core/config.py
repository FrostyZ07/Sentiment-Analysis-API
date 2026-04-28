"""Application configuration via environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings loaded from environment / .env file."""

    # Model
    model_path: str = "./models/distilbert-sentiment"
    model_path_v2: str = ""  # Optional second model for A/B testing

    # Server
    port: int = 8000
    log_level: str = "INFO"
    allowed_origins: str = "*"

    # Rate limiting
    rate_limit_per_minute: int = 60

    # API Keys (comma-separated SHA-256 hashes; empty = no auth required)
    api_keys: str = ""

    # W&B (used for model metadata endpoint)
    wandb_project: str = "sentiment-analysis-distilbert"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def api_keys_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
