from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import os

from ai_rag_weather.config import get_settings

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings

"""Language model and embeddings provider for AI RAG WeatherBot.

This module provides classes to instantiate language models and embeddings providers
based on configuration settings, supporting both OpenAI and HuggingFace models.
It includes a utility to retrieve and validate the OpenAI API key from multiple sources.
"""

def _get_openai_key(settings):
    """Retrieve and validate the OpenAI API key from multiple sources.

    Checks for the API key in Streamlit secrets, environment variables, or settings.
    Raises an error if no valid key is found.

    Args:
        settings: Configuration settings object.

    Returns:
        The OpenAI API key as a string.

    Raises:
        RuntimeError: If no valid OPENAI_API_KEY is found.
    """
    key = None
    try:
        import streamlit as st
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        pass
    key = key or os.getenv("OPENAI_API_KEY") or getattr(settings, "OPENAI_API_KEY", "")
    key = (key or "").strip()
    print(f"[providers] key sources -> secrets/env/settings | present? {bool(key)}")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put it in <repo>/.env, export it in your shell, "
            "or add it to .streamlit/secrets.toml."
        )
    return key

@dataclass
class EmbeddingsProvider:
    """Provider for text embedding models."""
    settings: Any = get_settings()
    def get(self):
        """Instantiate an embeddings model based on configuration.

        Returns:
            An embeddings model instance (OpenAIEmbeddings or HuggingFaceEmbeddings).
        """
        provider = (getattr(self.settings, "EMBEDDINGS_PROVIDER", "huggingface") or "huggingface").lower()
        if provider == "openai":
            model = getattr(self.settings, "OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
            return OpenAIEmbeddings(model=model, openai_api_key=_get_openai_key(self.settings))
        model_name = getattr(self.settings, "SENTENCE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model_name)

@dataclass
class LLMProvider:
    """Provider for language models."""
    settings: Any = get_settings()
    def get(self):
        """Instantiate a language model based on configuration.

        Returns:
            A ChatOpenAI model instance with configured model and temperature.
        """
        model = getattr(self.settings, "OPENAI_CHAT_MODEL", "gpt-4o-mini")
        temperature = float(getattr(self.settings, "OPENAI_TEMPERATURE", 0.2))
        return ChatOpenAI(model=model, temperature=temperature, openai_api_key=_get_openai_key(self.settings))