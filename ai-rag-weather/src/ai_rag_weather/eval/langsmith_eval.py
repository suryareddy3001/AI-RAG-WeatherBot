import os
from langchain.callbacks.tracers.langchain import LangChainTracerV2
from ai_rag_weather.config import get_settings

settings = get_settings()

os.environ["LANGCHAIN_TRACING_V2"] = str(settings.LANGCHAIN_TRACING_V2).lower()
os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT

tracer = LangChainTracerV2()

def with_tracing(func):
    def wrapper(*args, **kwargs):
        with tracer:
            return func(*args, **kwargs)
    return wrapper

def eval_harness(graph, queries=None):
    queries = queries or [
        "What's the weather in London?",
        "Summarize the introduction from the PDF.",
    ]
    results = []
    for q in queries:
        state = {"user_input": q, "intent": None, "weather": None, "answer": None, "retrieval": None}
        answer = graph.invoke(state)
        results.append({"query": q, "answer": answer})
    return results
