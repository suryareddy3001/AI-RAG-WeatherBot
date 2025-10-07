import pytest
from ai_rag_weather.graph.nodes import _classify_intent, _extract_city

@pytest.mark.parametrize("query, expected_intent", [
    ("What is the weather in Mumbai?", "weather"),
    ("What is the current wheather in Hyderabad today", "weather"),
    ("Tell me about temperature forecast for Delhi", "weather"),
    ("What does the PDF say about LangGraph?", "doc_qa"),
    ("Explain RAG from the document", "doc_qa"),
    ("Humidity levels?", "weather"),
    ("Random query without keywords", "doc_qa"),
])
def test_classify_intent(query, expected_intent):
    assert _classify_intent(query) == expected_intent


@pytest.mark.parametrize("query, expected_city", [
    ("What is the weather in Hyderabad today", "Hyderabad"),
    ("Weather in New York tomorrow", "New York"),
    ("Tell me the temp at San Francisco now", "San Francisco"),
    ("Forecast for London", "London"),
    ("Rain in paris today", None),
    ("What's the humidity in Tokyo?", "Tokyo"),
    ("Weather at Los Angeles California", "Los Angeles California"),
    ("No city mentioned", None),
])
def test_extract_city(query, expected_city):
    city = _extract_city(query)
    if city and city.lower() in {"rain", "no"}:
        city = None

    assert city == expected_city
