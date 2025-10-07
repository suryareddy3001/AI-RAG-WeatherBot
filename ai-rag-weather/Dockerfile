# syntax=docker/dockerfile:1
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY Makefile .
COPY .env.example .
COPY src/ ./src/
COPY data/ ./data/

RUN pip install --upgrade pip
RUN pip install -e .[dev]

FROM base as prod
CMD ["streamlit", "run", "src/ai_rag_weather/ui/app.py"]
