PY = python
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate

.PHONY: venv install format lint typecheck test test-docker eval-planner-docker eval-retrieval-docker eval-retrieval-heuristic-docker eval-retrieval-groq-docker eval-retrieval-compare-docker eval-soft-product-docker check run ingest-rag compose-up migrate-db ingest-rag-docker

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

eval-retrieval-docker:
	docker build -f Dockerfile.test -t astro-chatbot-service-test .
	docker run --rm --entrypoint python astro-chatbot-service-test finetune/retrieval_eval.py

eval-retrieval-heuristic-docker:
	docker build -f Dockerfile.test -t astro-chatbot-service-test .
	docker run --rm --entrypoint python astro-chatbot-service-test finetune/retrieval_eval.py --reranker-provider heuristic --reranker-model heuristic-v1

eval-retrieval-groq-docker:
	docker build -f Dockerfile.test -t astro-chatbot-service-test .
	docker run --rm --entrypoint python astro-chatbot-service-test finetune/retrieval_eval.py --reranker-provider groq_listwise

eval-retrieval-compare-docker:
	docker build -f Dockerfile.test -t astro-chatbot-service-test .
	docker run --rm --entrypoint python astro-chatbot-service-test finetune/retrieval_eval.py --compare-rerankers heuristic,groq_listwise

eval-soft-product-docker:
	docker build -f Dockerfile.test -t astro-chatbot-service-test .
	docker run --rm --entrypoint python astro-chatbot-service-test finetune/product_recommendation_eval.py

check:
	$(ACTIVATE); $(MAKE) format
	$(ACTIVATE); $(MAKE) lint
	$(ACTIVATE); $(MAKE) typecheck
	$(ACTIVATE); $(MAKE) test

run:
	$(ACTIVATE); uvicorn src.main:app --reload --port 8010

ingest-rag:
	$(ACTIVATE); python -m scripts.embed_astrology_texts

compose-up:
	docker compose up -d postgres redis

migrate-db:
	docker compose run --rm api python -m alembic upgrade head

ingest-rag-docker:
	docker compose run --rm api python -m scripts.embed_astrology_texts
