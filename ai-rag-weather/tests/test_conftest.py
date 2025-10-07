import pytest
from ai_rag_weather.rag.retriever import RAGRetriever
from ai_rag_weather.llm.providers import LLMProvider
from ai_rag_weather.weather.client import WeatherClient

class MockVectorDB:
    def search(self, query_vector, top_k=5, score_threshold=0.0):
        return [
            {"payload": {"page": 1, "text": "Sample text"}}
        ]

class MockEmbeddings:
    def embed_query(self, text):
        return [0.1] * 384

class MockLLM:
    def __call__(self, prompt):
        return "Sample summary"

@pytest.fixture
def mock_rag_retriever(monkeypatch):
    vectordb = MockVectorDB()
    embeddings = MockEmbeddings()
    llm = MockLLM()
    return RAGRetriever(vectordb=vectordb, embeddings=embeddings, llm=llm)

@pytest.fixture
def mock_weather_client(monkeypatch):
    monkeypatch.setattr(
        "ai_rag_weather.weather.client.WeatherClient.fetch",
        lambda self, city: {"city": "London", "temp": 15, "description": "sunny"} if city == "London" else None
    )

@pytest.fixture
def mock_llm_provider(monkeypatch):
    monkeypatch.setattr(
        "ai_rag_weather.llm.providers.LLMProvider.get",
        lambda self: lambda prompt: f"Processed: {prompt}"
    )

def test_rag_retriever(mock_rag_retriever):
    result = mock_rag_retriever.retrieve("Find information about AI")
    assert "contexts" in result
    assert isinstance(result["contexts"], list)
    assert result["contexts"][0]["text"] == "Sample text"

def test_weather_client(mock_weather_client):
    client = WeatherClient()
    result = client.fetch("London")
    assert result["city"] == "London"
    assert result["temp"] == 15
    assert result["description"] == "sunny"

def test_llm_provider(mock_llm_provider):
    provider = LLMProvider()
    process = provider.get()
    result = process("Generate a summary")
    assert result == "Processed: Generate a summary"
