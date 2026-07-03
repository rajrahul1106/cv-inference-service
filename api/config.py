"""
Application configuration.

Loads settings from environment variables (and a local `.env` file) using
pydantic-settings v2. Import the module-level `settings` singleton anywhere
config is needed:

    from api.config import settings
    engine = create_async_engine(settings.database_url)

Every field has a development default that mirrors `.env.example`, so importing
this module never crashes when `.env` is absent (e.g. in CI or tests). Real
values come from `.env` (gitignored) or the process environment, which override
the defaults. Environment variable matching is case-insensitive, so the
UPPERCASE keys in `.env` (DATABASE_URL) bind to the lowercase fields here.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings, populated from the environment / `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated keys present in the environment
    )

    # ─── Database ───────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://cvuser:cvpass@localhost:5432/cv_inference"

    # ─── Redis (Celery broker + pub/sub) ────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ─── Celery ─────────────────────────────────────────────────
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # ─── API server ─────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    environment: str = "development"

    # ─── Storage paths ──────────────────────────────────────────
    upload_dir: str = "./uploads"
    annotated_dir: str = "./annotated"
    models_cache_dir: str = "./models_cache"


# Module-level singleton. Import this, not the class.
settings = Settings()
