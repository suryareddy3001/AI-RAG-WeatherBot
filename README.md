# ai-rag-weather

A production-ready RAG + WeatherBot pipeline using LangChain, LangGraph, Qdrant, Streamlit, and LangSmith.

## Overview

```
+-------------------+
|   User Message    |
+-------------------+
          |
          v
   +--------------+
   |  RouterNode  | <--- classifies intent (weather/doc_qa)
   +--------------+
    |           |
    v           v
+--------+   +--------+
|Weather |   |  RAG   |
| Node   |   | Node   |
+--------+   +--------+
    |           |
    +-----+-----+
          v
   +--------------+
   | SynthesisNode|
   +--------------+
          |
          v
   +--------------+
   |   Response   |
   +--------------+
```

- **Weather**: Calls OpenWeatherMap API for real-time data.
- **RAG**: Retrieves and summarizes PDF content via Qdrant + LLM.
- **Synthesis**: Crafts a final, cited answer.

## Setup

1. **Clone & Install**

```sh
git clone <repo-url>
cd ai-rag-weather
make venv
make install
cp .env.example .env
```

2. **Configure Environment**

Edit `.env` with your API keys (OpenAI, OpenWeatherMap, LangSmith, etc).

3. **Run Qdrant & App (Docker Compose)**

```sh
make up
```

4. **Ingest a PDF**

```sh
make ingest PDF=data/assignment.pdf
```
Or upload via the Streamlit UI sidebar.

5. **Launch Streamlit UI**

```sh
make run
```

## LangSmith Tracing

- Set `LANGCHAIN_TRACING_V2=true` in `.env` to enable tracing.
- View runs at [LangSmith](https://smith.langchain.com/).

## Testing

```sh
make test
```

- Fast, deterministic unit tests (pytest + coverage).
- Mocks for network/LLM/vector store.

## Linting & Formatting

```sh
make lint
make fmt
```

## Troubleshooting

- **Qdrant not reachable?** Ensure Docker is running and port 6333 is free.
- **No API key?** Use `sentence-transformers` for local embeddings.
- **LangSmith not logging?** Check `LANGCHAIN_API_KEY` and project name.

## Deliverables Checklist

- [x] Python codebase compiles & runs
- [x] Streamlit chat demo functional
- [x] Qdrant stores embeddings and serves retrieval
- [x] LangGraph routes correctly
- [x] LangSmith tracing on and a tiny eval harness included
- [x] Unit tests pass
- [x] Docker Compose works (qdrant + app)
- [x] Lint, type-check, format clean
