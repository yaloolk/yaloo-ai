"""
app/core/config.py
Centralised settings loaded from environment variables.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import SecretStr



class Settings(BaseSettings):
    # --- Supabase ---
    supabase_url: str
    supabase_service_key: str          # service-role key (never expose to client)
    supabase_webhook_secret: str = ""  # shared secret for DB webhook verification

    # --- Embedding model ---
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_device: str = "cpu"      # "cuda" if GPU available on VPS
    embedding_dim: int = 768

    # --- Gemini (chatbot) ---
    PRIMARY_GEMINI_API_KEY:   SecretStr
    SECONDARY_GEMINI_API_KEY: SecretStr
    TERTIARY_GEMINI_API_KEY:  SecretStr

    # --- Recommendation engine ---
    top_k: int = 5
    rerank_vec_weight: float = 0.80
    rerank_budget_weight: float = 0.12
    rerank_active_weight: float = 0.08   # activities only

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
