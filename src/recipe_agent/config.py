from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    google_api_key: str
    ollama_base_url: str
    ollama_model: str
    ollama_translate_model: str
    supabase_url: str
    supabase_service_key: str


settings = Settings()