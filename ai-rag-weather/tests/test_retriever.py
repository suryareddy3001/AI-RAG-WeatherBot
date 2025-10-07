from ai_rag_weather.rag.retriever import RAGRetriever

class MockVectorDB:
    def search(self, query_vector, top_k=5, score_threshold=0.0):
        return [
            {"payload": {"page": 1, "text": "Relevant chunk 1"}},
            {"payload": {"page": 2, "text": "Relevant chunk 2"}},
        ]

class MockEmbeddings:
    def embed_query(self, text):
        return [0.1] * 384

class MockLLM:
    def __call__(self, prompt):
        return "Summary: This is a generated summary."

def test_retrieve():
    retriever = RAGRetriever(
        vectordb=MockVectorDB(),
        embeddings=MockEmbeddings(),
        llm=MockLLM()
    )

    result = retriever.retrieve("test query")
    assert "contexts" in result
    assert "query" in result
    assert len(result["contexts"]) == 2
    assert result["contexts"][0]["text"] == "Relevant chunk 1"
    assert result["contexts"][1]["text"] == "Relevant chunk 2"
    assert "draft" not in result
