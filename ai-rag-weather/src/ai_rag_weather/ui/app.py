"""AI RAG WeatherBot Streamlit application.

This application provides a user interface for interacting with a Retrieval-Augmented Generation (RAG) system
that answers queries about weather or content from uploaded PDFs. It supports PDF ingestion, vector-based retrieval,
and weather data integration using a LangChain graph. The interface includes configuration options, chat history,
and evidence rendering for transparency.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st
from ai_rag_weather.config import get_settings
from ai_rag_weather.graph.graph import build_graph
from ai_rag_weather.ingestion.pdf_ingest import load_and_chunk_pdf, embed_and_upsert
from ai_rag_weather.llm.providers import EmbeddingsProvider
from ai_rag_weather.vectordb.qdrant_store import QdrantStore
from ai_rag_weather.logging import get_logger
from typing import cast

settings = get_settings()
logger = get_logger(__name__)
graph = build_graph()

st.set_page_config(page_title="AI RAG WeatherBot", layout="wide")
st.title("AI RAG WeatherBot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = cast(List[Dict[str, Any]], [])
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.2
if "top_k" not in st.session_state:
    st.session_state.top_k = 5
if "force_weather" not in st.session_state:
    st.session_state.force_weather = False

with st.sidebar:
    st.header("Config")
    st.selectbox("Embeddings Provider (UI hint)", ["openai", "sentence-transformers"], index=0, key="emb_ui_hint")
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.05)
    st.session_state.top_k = st.slider("Top-K (retrieval)", 1, 10, st.session_state.top_k)
    enable_tracing = st.checkbox("Enable LangSmith Tracing", value=settings.LANGCHAIN_TRACING_V2)
    st.session_state.force_weather = st.checkbox("Force weather (debug)", value=st.session_state.force_weather,
                                                help="Bypass intent classifier for the next question.")

    st.markdown("---")
    st.subheader("PDF Ingestion")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if st.button("Re-ingest PDF") and uploaded:
        try:
            data_dir = Path("data")
            data_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = data_dir / uploaded.name
            with open(pdf_path, "wb") as f:
                f.write(uploaded.read())

            docs = load_and_chunk_pdf(pdf_path)
            embeddings = EmbeddingsProvider().get()
            probe = embeddings.embed_query("probe")
            if hasattr(probe, "tolist"):
                probe = probe.tolist()
            vec_size = len(probe)
            vectordb = QdrantStore(check_compatibility=False)
            vectordb.ensure_collection(vec_size, recreate_if_mismatch=True)
            embed_and_upsert(docs, embeddings, vectordb)
            st.success(f"Ingested {len(docs)} chunks from {uploaded.name}")
        except Exception as e:
            logger.exception("PDF ingestion failed")
            st.error(f"PDF ingestion failed: {e}")

    if st.button("Clear chat"):
        st.session_state.chat_history = []

def _render_evidence(ev: Any) -> None:
    """Render evidence from RAG contexts or weather data.

    Args:
        ev: Evidence data, either a list of RAG contexts (dictionaries with page/text) or a weather dictionary.
    """
    if not ev:
        return
    with st.expander("Evidence"):
        if isinstance(ev, dict) and {"city", "temp", "description"} <= set(ev.keys()):
            st.markdown(
                f"**Weather** — {ev['city']}, {ev.get('country','?')}\n\n"
                f"- {ev['description']}\n"
                f"- Temp: {ev['temp']}°  (Feels like: {ev['feels_like']}°)\n"
                f"- Humidity: {ev['humidity']}%\n"
                f"- Wind: {ev['wind_speed']} m/s"
            )
            return
        if isinstance(ev, list):
            for i, c in enumerate(ev, start=1):
                page = c.get("page", "?")
                text = (c.get("text") or "")[:500].replace("\n", " ")
                st.markdown(f"**Ctx {i} — Page {page}**")
                st.code(text + ("..." if len(c.get('text','')) > 500 else ""), language="markdown")
            return
        st.code(json.dumps(ev, indent=2, ensure_ascii=False))

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        _render_evidence(msg.get("evidence"))

st.markdown("---")

prompt = st.chat_input("Ask about the weather or your PDF…")
if prompt:
    state: Dict[str, Any] = {
        "user_input": prompt,
        "intent": "weather" if st.session_state.force_weather else None,
        "weather": None,
        "answer": None,
        "retrieval": None,
        "temperature": float(st.session_state.temperature),
        "top_k": int(st.session_state.top_k),
        "enable_tracing": bool(enable_tracing),
    }

    st.session_state.chat_history.append({"role": "user", "content": prompt})

    try:
        result: Dict[str, Any] = graph.invoke(state)
        answer_text: str = result.get("answer") or "I couldn’t produce an answer."
        evidence = None
        if result.get("retrieval") and isinstance(result["retrieval"], dict):
            evidence = result["retrieval"].get("contexts")
        if not evidence and result.get("weather"):
            evidence = result["weather"]

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer_text,
            "evidence": evidence
        })
    except Exception as e:
        logger.exception("Graph invocation failed")
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"Error while generating answer: {e}"
        })

    st.session_state.force_weather = False
    st.rerun()