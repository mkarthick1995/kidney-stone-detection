from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str
    model_weights_path: str
    allowed_origins: str = "http://localhost:5173"
    claude_model: str = "claude-sonnet-4-6"
    claude_timeout_seconds: int = 30

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()
