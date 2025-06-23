"""Settings configuration for the Ironhide framework, including API endpoints, keys, models, and general options."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings

from ironhide.utils import Provider


class Settings(BaseSettings):
    """Configuration settings for the Ironhide framework, including API endpoints, keys, models, and general options."""

    # LLM
    ironhide_provider_url: str | None = None
    ironhide_provider: Provider = Provider.openai
    ironhide_api_key: SecretStr = SecretStr("")
    ironhide_completions_model: str = "gpt-4o-mini"
    ironhide_audio_to_text_model: str = "whisper-1"

    # General
    log_level: str = "INFO"
    ironhide_request_timeout: int = 60
    ironhide_max_retries: int = 3
    ironhide_retry_delay: int = 3


settings = Settings()
