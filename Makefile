PY = python
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate

.PHONY: venv install format lint typecheck test test-docker eval-planner-docker check run ingest-rag

venv:
	$(PY) -m venv $(VENV)
	@echo "Run: source $(VENV)/bin/activate"

install:
	$(ACTIVATE); pip install -U pip
	$(ACTIVATE); pip install -r requirements.txt -r dev-requirements.txt

format:
	$(ACTIVATE); ruff format src tests

lint:
	$(ACTIVATE); ruff check src tests

typecheck:
	$(ACTIVATE); mypy src

test:
	$(ACTIVATE); PYTHONPATH=. pytest

test-docker:
	docker build -f Dockerfile.test -t astro-chatbot-service-test .
	docker run --rm astro-chatbot-service-test tests

eval-planner-docker:
	docker build -f Dockerfile.test -t astro-chatbot-service-test .
	docker run --rm --entrypoint python astro-chatbot-service-test finetune/eval.py

check:
	$(ACTIVATE); $(MAKE) format
	$(ACTIVATE); $(MAKE) lint
	$(ACTIVATE); $(MAKE) typecheck
	$(ACTIVATE); $(MAKE) test

run:
	$(ACTIVATE); uvicorn src.main:app --reload --port 8010

ingest-rag:
	$(ACTIVATE); python -m scripts.embed_astrology_texts
