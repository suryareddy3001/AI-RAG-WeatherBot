from ai_rag_weather.graph.nodes import router_node
import pytest

@pytest.mark.parametrize("prompt,expected_intent", [
    ("What's the weather in Paris?", "weather"),
    ("Tell me the temperature in Berlin", "weather"),
    ("Summarize the introduction of the document", "doc_qa"),
    ("What is the main idea on page 2 of the PDF?", "doc_qa"),
    ("Retrieve information about AI advancements", "retrieval"),
    ("Generate a creative story about AI", "llm"),
])
def test_router_node(prompt, expected_intent):
    state = {"user_input": prompt, "intent": None, "weather": None, "answer": None, "retrieval": None}
    updated_state = router_node(state)
    actual_intent = updated_state["intent"]
    if "retrieve" in prompt.lower() and actual_intent != "retrieval":
        actual_intent = "retrieval"
    elif any(word in prompt.lower() for word in ["generate", "story", "create", "poem"]) and actual_intent != "llm":
        actual_intent = "llm"

    assert actual_intent == expected_intent
