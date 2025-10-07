from __future__ import annotations

import re
from typing import TypedDict, Optional, Any, Dict
from difflib import SequenceMatcher

from ai_rag_weather.llm.providers import LLMProvider, EmbeddingsProvider
from ai_rag_weather.vectordb.qdrant_store import QdrantStore
from ai_rag_weather.rag.retriever import RAGRetriever
from ai_rag_weather.weather.client import WeatherClient

"""Graph nodes for AI RAG WeatherBot query processing.

This module defines the state and nodes for a LangChain graph that processes user queries,
classifying intent, fetching weather data, retrieving document contexts, and synthesizing responses.
It includes utilities for intent classification and city extraction with fuzzy matching.
"""

class GraphState(TypedDict, total=False):
    """State dictionary for the LangChain graph.

    Attributes:
        user_input: The user's query string.
        intent: The classified intent ("weather" or "doc_qa").
        weather: Weather data dictionary, if applicable.
        retrieval: Retrieved document contexts, if applicable.
        answer: The final synthesized answer.
    """
    user_input: str
    intent: Optional[str]
    weather: Optional[Dict[str, Any]]
    retrieval: Optional[Dict[str, Any]]
    answer: Optional[str]

_WEATHER_HINTS = (
    "weather", "temperature", "temp", "forecast", "rain",
    "humidity", "wind", "snow", "sun", "cloud", "visibility"
)

def _fuzzy_contains(text: str, keywords: tuple, threshold: float = 0.85) -> bool:
    """Check if any word in the text fuzzy-matches a keyword.

    Args:
        text: Input text to check.
        keywords: Tuple of keywords to match against.
        threshold: Minimum similarity ratio for a match (default: 0.85).

    Returns:
        True if a word in the text matches a keyword with similarity >= threshold, False otherwise.
    """
    for word in text.lower().split():
        for kw in keywords:
            if SequenceMatcher(None, word, kw).ratio() >= threshold:
                return True
    return False

def _classify_intent(text: str) -> str:
    """Classify the intent of a user query.

    Args:
        text: The user query string.

    Returns:
        "weather" if the query is weather-related, "doc_qa" otherwise.
    """
    t = text.lower()
    if _fuzzy_contains(t, _WEATHER_HINTS):
        return "weather"
    if re.search(r"\b(what('?s)?|how)( is|s|’s)? the (weather|temperature)\b", t):
        return "weather"
    return "doc_qa"

def _extract_city(text: str) -> Optional[str]:
    """Extract a city name from the user query.

    Args:
        text: The user query string.

    Returns:
        The extracted city name, or None if no city is found.
    """
    m = re.search(r"\b(?:in|at)\s+([A-Z][a-zA-Z\-\.’']+(?:\s+[A-Z][a-zA-Z\-\.’']+)*)", text)
    if m:
        city = m.group(1).strip(" .?!,;:").strip()
        return city
    m = re.search(r"weather\s+in\s+([A-Z][a-zA-Z\-\.’']+(?:\s+[A-Z][a-zA-Z\-\.’']+)*)", text, flags=re.I)
    if m:
        return m.group(1).strip(" .?!,;:")
    tokens = re.findall(r"[A-Z][a-zA-Z\-\.’']+(?:\s+[A-Z][a-zA-Z\-\.’']+)*", text)
    return tokens[-1] if tokens else None

def router_node(state: GraphState) -> GraphState:
    """Route the query based on its intent.

    Args:
        state: The current graph state.

    Returns:
        Updated state with the classified intent.
    """
    intent = state.get("intent") or _classify_intent(state["user_input"])
    return {**state, "intent": intent}

def weather_node(state: GraphState) -> GraphState:
    """Process weather-related queries.

    Args:
        state: The current graph state.

    Returns:
        Updated state with weather data or an error message.
    """
    city = _extract_city(state["user_input"])
    if not city:
        return {**state, "answer": "Please mention a city, e.g., “What’s the weather in Mumbai?”"}

    client = WeatherClient()
    resp = client.fetch(city)
    if resp:
        weather = {
            "city": resp.city,
            "country": resp.country,
            "temp": resp.temp,
            "feels_like": resp.feels_like,
            "description": resp.description,
            "humidity": resp.humidity,
            "wind_speed": resp.wind_speed,
        }
        return {**state, "weather": weather}

    suggestions = client.search_cities(city)
    if suggestions:
        sug_text = ", ".join([f"{s['name']}, {s.get('country', '?')}" for s in suggestions])
        return {**state, "answer": f"Couldn’t find '{city}'. Did you mean one of these? {sug_text}. Try again with a suggestion."}
    
    return {**state, "answer": f"Couldn’t fetch weather for {city} or find similar cities. Check spelling or try another city."}

def rag_node(state: GraphState) -> GraphState:
    """Process document-based question-answering queries.

    Args:
        state: The current graph state.

    Returns:
        Updated state with retrieved contexts and answer, or an error message.
    """
    try:
        vectordb = QdrantStore()
        embeddings = EmbeddingsProvider().get()
        llm = LLMProvider().get()

        retriever = RAGRetriever(vectordb=vectordb, embeddings=embeddings, llm=llm)
        result = retriever.retrieve(state["user_input"])
        answer_text = retriever.summarize(state["user_input"], result["contexts"])

        return {**state, "retrieval": result, "answer": answer_text}
    except Exception as e:
        return {**state, "retrieval": None, "answer": f"⚠️ Retrieval step failed: {e}"}

def synthesis_node(state: GraphState) -> GraphState:
    """Synthesize the final response from weather or retrieval data.

    Args:
        state: The current graph state.

    Returns:
        Updated state with a synthesized answer, if not already present.
    """
    if state.get("weather") and not state.get("answer"):
        w = state["weather"]
        ans = (
            f"{w['city']}, {w['country']}: {w['description']}. "
            f"Temp {w['temp']}° ({w['feels_like']}° feels like). "
            f"Humidity {w['humidity']}%, wind {w['wind_speed']} m/s."
        )
        return {**state, "answer": ans}
    return state

def route_intent(state: GraphState) -> str:
    """Determine the routing path based on the state's intent.

    Args:
        state: The current graph state.

    Returns:
        The intent ("weather" or "doc_qa") to route to the appropriate node.
    """
    return state.get("intent") or "doc_qa"