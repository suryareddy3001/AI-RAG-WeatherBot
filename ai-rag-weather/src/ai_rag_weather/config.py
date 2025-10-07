from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache
from pathlib import Path
import os

"""Configuration management for the RAGChain-WeatherBot application.

This module defines a Pydantic settings class to manage configuration variables
and provides utilities to load environment variables from a .env file.
It ensures critical settings like API keys are properly loaded and cached.
"""

ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".env"

def _load_env_exact(path: Path) -> None:
    """Load environment variables from a specific .env file path.

    Attempts to load the .env file using python-dotenv. If the OPENAI_API_KEY
    is not found in the environment after loading, it parses the file and
    manually injects the key into os.environ as a fallback.

    Args:
        path: The path to the .env file.
    """
    try:
        from dotenv import load_dotenv, dotenv_values
    except Exception:
        print("[config] python-dotenv not installed; `pip install python-dotenv` recommended")
        return

    loaded = load_dotenv(dotenv_path=path, override=False, verbose=True)
    print(f"[config] load_dotenv({path}) -> {loaded} (exists={path.exists()})")

    if not os.getenv("OPENAI_API_KEY"):
        vals = dotenv_values(dotenv_path=path)
        if "OPENAI_API_KEY" in vals and vals["OPENAI_API_KEY"]:
            os.environ.setdefault("OPENAI_API_KEY", vals["OPENAI_API_KEY"])
            print("[config] Injected OPENAI_API_KEY into os.environ from parsed .env")
        else:
            print("[config] OPENAI_API_KEY not found in parsed .env or empty")

_load_env_exact(ENV_PATH)

class Settings(BaseSettings):
    """Configuration settings for the RAGChain-WeatherBot application.

    Defines environment variables for LLM providers, embeddings, LangSmith tracing,
    Qdrant vector database, OpenWeatherMap API, and application settings.
    """
    LLM_PROVIDER: str = Field("openai")
    OPENAI_API_KEY: str = Field("")
    EMBEDDINGS_PROVIDER: str = Field("openai")
    SENTENCE_MODEL: str = Field("sentence-transformers/all-MiniLM-L6-v2")
    LANGCHAIN_TRACING_V2: bool = Field(True)
    LANGCHAIN_ENDPOINT: str = Field("https://api.smith.langchain.com")
    LANGCHAIN_API_KEY: str = Field("")
    LANGSMITH_PROJECT: str = Field("ai-rag-weather")
    QDRANT_HOST: str = Field("qdrant")
    QDRANT_PORT: int = Field(6333)
    QDRANT_COLLECTION: str = Field("ai_rag_weather_docs")
    OPENWEATHER_API_KEY: str = Field("")
    OPENWEATHER_UNITS: str = Field("metric")
    APP_ENV: str = Field("dev")

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

@lru_cache(maxsize=1)
def get_settings() -> "Settings":
    """Retrieve cached application settings.

    Creates and caches a Settings instance, logging the presence and masked value
    of the OPENAI_API_KEY, as well as the root and .env file paths.

    Returns:
        A Settings instance containing the application's configuration.
    """
    s = Settings()
    masked = "****" + s.OPENAI_API_KEY[-6:] if s.OPENAI_API_KEY else None
    print(f"[config] OPENAI_API_KEY present? {bool(s.OPENAI_API_KEY)} value(masked)={masked}")
    print(f"[config] ROOT={ROOT}  ENV_PATH={ENV_PATH}")
    return s