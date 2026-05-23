from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    All configuration lives here.

    WHY pydantic_settings:
    - Automatically reads from .env file
    - Type-validates every variable at startup
    - Fails loudly if a required variable is missing
    - One source of truth for all config
    """

    # OpenAI
    OPENAI_API_KEY: str

    # MongoDB Atlas
    MONGODB_URI: str
    DATABASE_NAME: str = "walkWithJesusData"
    COLLECTION_NAME: str = "anxietyData"
    VECTOR_INDEX_NAME: str = "autoembed_index"

    # RAG tuning
    RETRIEVAL_TOP_K: int = 4           # how many verses to retrieve
    LLM_MODEL: str = "gpt-4o-mini"    # swap model here without touching other files
    LLM_MAX_TOKENS: int = 700
    LLM_TEMPERATURE: float = 0.7

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """
    WHY lru_cache:
    Settings object is created once and reused.
    Without this, every request would re-read the .env file — wasteful.
    """
    return Settings()
