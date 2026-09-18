from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MODELS = ("tiny", "base", "small", "medium", "large-v3")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    max_upload_mb: int = Field(default=500, gt=0)
    max_media_duration_seconds: int = Field(default=10_800, gt=0)
    result_retention_minutes: int = Field(default=60, gt=0)
    default_model: str = "small"
    default_language: str = "pt"
    model_cache_dir: Path = Path("/models")
    jobs_dir: Path = Path("/tmp/whisper-ui")
    compute_type: str = "int8"
    batch_size: int = Field(default=4, gt=0)
    cleanup_interval_seconds: int = Field(default=60, gt=0)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @model_validator(mode="after")
    def validate_defaults(self) -> "Settings":
        if self.default_model not in MODELS:
            raise ValueError(f"DEFAULT_MODEL must be one of {', '.join(MODELS)}")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
