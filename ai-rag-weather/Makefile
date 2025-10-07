VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

venv:
	python3.11 -m venv $(VENV)

install: venv
	$(PIP) install -e .[dev]

lint:
	ruff src tests
	black --check src tests
	mypy src

fmt:
	ruff --fix src tests
	black src tests

run:
	streamlit run src/ai_rag_weather/ui/app.py

ingest:
	$(PYTHON) -m src.ai_rag_weather.ingestion.pdf_ingest --pdf data/assignment.pdf

test:
	pytest -q

up:
	docker compose up --build

down:
	docker compose down -v
