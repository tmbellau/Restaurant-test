"""Application configuration via Pydantic Settings.

All environment variables are declared here. No hardcoded keys, URLs, or paths
elsewhere in the codebase. Import the `settings` singleton to read config.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # Database
    database_url: str = Field(
        default="postgresql+psycopg://wagamama:wagamama@localhost:5432/wagamama_forecast"
    )
    database_url_sync: str = Field(
        default="postgresql+psycopg://wagamama:wagamama@localhost:5432/wagamama_forecast"
    )

    # Redis (Layer 2+)
    redis_url: str = Field(default="redis://localhost:6379/0")

    # External APIs
    open_meteo_base_url: str = Field(default="https://api.open-meteo.com/v1")
    open_meteo_historical_url: str = Field(default="https://archive-api.open-meteo.com/v1")
    besttime_api_key: str | None = Field(default=None)
    ticketmaster_api_key: str | None = Field(default=None)
    football_data_api_key: str | None = Field(default=None)
    tfl_app_key: str | None = Field(default=None)
    deliverect_api_key: str | None = Field(default=None)

    # MLflow (Layer 2)
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")

    # POS ingestion paths
    pos_inbox_dir: Path = Field(default=Path("./data/pos_inbox"))
    pos_processed_dir: Path = Field(default=Path("./data/pos_processed"))

    # Training freshness (hours after which POS data is considered final)
    pos_finalisation_hours: int = Field(default=24)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
