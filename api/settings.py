"""Runtime configuration via environment variables.

Note the OpenRouter key itself (``OPENROUTER_API_KEY``) is read directly by
``src/llm/client.py``, not duplicated here — this only covers API-layer
concerns.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    cors_allow_origins: str = "*"   # comma-separated, or "*"
    max_upload_mb: int = 10
    ocr_lang: str = "eng"

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_allow_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


settings = Settings()
