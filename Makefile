PY = python
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate

.PHONY: venv install format lint typecheck test test-docker check run

venv:
	$(PY) -m venv $(VENV)
	@echo "Run: source $(VENV)/bin/activate"

install:
	$(ACTIVATE); pip install -U pip
	$(ACTIVATE); pip install -r requirements.txt -r dev-requirements.txt

format:
	$(ACTIVATE); ruff format app src tests

lint:
	$(ACTIVATE); ruff check app src tests

typecheck:
	$(ACTIVATE); mypy app src

test:
	$(ACTIVATE); PYTHONPATH=src pytest

test-docker:
	docker build -f Dockerfile.test -t astro-chatbot-service-test .
	docker run --rm astro-chatbot-service-test tests

check:
	$(ACTIVATE); $(MAKE) lint
	$(ACTIVATE); $(MAKE) typecheck
	$(ACTIVATE); PYTHONPATH=src $(MAKE) test

run:
	$(ACTIVATE); PYTHONPATH=src uvicorn app.main:app --reload --port 8010
