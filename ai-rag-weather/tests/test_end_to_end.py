import re
from ai_rag_weather.graph.graph import build_graph
from ai_rag_weather.weather.client import WeatherResponse

class MockWeatherClient:
    """Simulates a weather API response."""
    def fetch(self, city, *args, **kwargs):
        return WeatherResponse(
            city=city,
            country="IN",
            temp=28.0,
            feels_like=30.41,
            humidity=68,
            description="overcast clouds",
            wind_speed=3.58,
            raw={"source": "mocked-weather"}
        )

class MockRetriever:
    """Simulates RAG Retriever returning mock document context."""
    def retrieve(self, query, *args, **kwargs):
        return {
            "contexts": [{"page": 1, "text": "Sample PDF content"}],
            "draft": "Sample summary"
        }

    def summarize(self, query, *args, **kwargs):
        """Added to fix missing summarize()"""
        print(f"[MockRetriever] summarize called with query={query}")
        return {"summary": f"Processed: {query}"}


class MockLLMProvider:
    """Simulates an LLM processing a prompt."""
    def get(self):
        def mock_llm(prompt, *args, **kwargs):
            print(f"[MockLLM] called with prompt={prompt}, kwargs={kwargs}")
            return f"Processed: {prompt}"
        return mock_llm

def test_end_to_end(monkeypatch):
    from ai_rag_weather.graph import nodes
    graph = build_graph()
    monkeypatch.setattr(nodes, "WeatherClient", lambda *a, **kw: MockWeatherClient())
    monkeypatch.setattr(nodes, "RAGRetriever", lambda *a, **kw: MockRetriever())
    monkeypatch.setattr(nodes, "LLMProvider", lambda *a, **kw: MockLLMProvider())
    state = {
        "user_input": "What is weather condition in Hyderabad?",
        "intent": None,
        "weather": None,
        "answer": None,
        "retrieval": None,
    }
    result = graph.invoke(state)
    assert "answer" in result
    assert re.match(
        r"Hyderabad, IN: .*clouds.*Temp.*°.*Humidity.*%, wind .* m/s",
        result["answer"]
    )
    state = {
        "user_input": "Summarize section X from the PDF",
        "intent": None,
        "weather": None,
        "answer": None,
        "retrieval": None,
    }
    result = graph.invoke(state)
    assert "answer" in result
    assert (
        result["answer"]["summary"]
        == "Processed: Summarize section X from the PDF"
    )
    state = {
        "user_input": "Retrieve information about AI advancements",
        "intent": None,
        "weather": None,
        "answer": None,
        "retrieval": None,
    }
    result = graph.invoke(state)
    assert "retrieval" in result
    assert result["retrieval"]["draft"] == "Sample summary"
    assert result["retrieval"]["contexts"][0]["text"] == "Sample PDF content"
