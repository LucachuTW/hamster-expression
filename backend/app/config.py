from pydantic_settings import BaseSettings

from backend.ai import config as ai_config


class Settings(BaseSettings):
    app_name: str = "Emotion Detection Service"
    version: str = "0.1.0"
    api_prefix: str = "/api"
    weights_path: str = str(ai_config.WEIGHTS_PATH)
    log_level: str = "INFO"


settings = Settings()
