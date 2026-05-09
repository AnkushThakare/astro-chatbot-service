# astro-chatbot-service

Standalone chatbot backend for astrology use cases. This repo is separate from `astrology-service` and focuses on orchestration while following the same baseline service conventions used in `core-service` and `astrology-service`:

- Groq LLM integration
- Swiss Ephemeris-backed astrology context via local or remote adapter
- RAG document ingestion and retrieval
- Conversation memory persisted in a database
- Prompt assembly and chat orchestration
- Fine-tuning dataset export scaffolding

## Stack

- Python 3.10+
- FastAPI
- SQLAlchemy
- Groq OpenAI-compatible chat API
- Optional integration with the neighboring `astrology-service` repo

## Repository Layout

```text
astro-chatbot-service/
+-- app/                         FastAPI entrypoint, middleware, and API routes
+-- src/astro_chatbot_service/   Adapters, models, and orchestration services
+-- tests/                       API-oriented test coverage
+-- .env.example
+-- .env.test
+-- Dockerfile
+-- Makefile
+-- pyproject.toml
+-- pytest.ini
+-- README.md
```

## Quick Start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r dev-requirements.txt
cp .env.example .env
```

Run the API:

```bash
PYTHONPATH=src uvicorn app.main:app --reload --port 8010
```

Or use the same workflow style as the reference repos:

```bash
make format
make lint
make typecheck
make test
```

If there is no usable host Python interpreter in your workspace, run tests in Docker instead:

```bash
docker build -f Dockerfile.test -t astro-chatbot-service-test .
docker run --rm astro-chatbot-service-test tests
```

On Windows PowerShell you can use the committed helper:

```powershell
./scripts/test_in_docker.ps1
./scripts/test_in_docker.ps1 tests/test_planner.py
```

## Astrology Engine Modes

This service keeps astrology behind an adapter boundary so the repo stays separate:

- `ASTROLOGY_ENGINE_MODE=remote`
  - Calls `POST {ASTROLOGY_SERVICE_URL}/kundli/generate`
- `ASTROLOGY_ENGINE_MODE=local`
  - Imports `astrology_engine` from `ASTROLOGY_ENGINE_PATH`

The local mode is intended for development when `astrology-service` exists nearby and exposes its `src/` folder.

## API Surface

- `GET /health`
- `GET /ready`
- `GET /api/v1/health`
- `POST /api/v1/chat`
- `GET /api/v1/memory/{session_id}`
- `POST /api/v1/retrieval/documents`
- `POST /api/v1/retrieval/search`
- `POST /api/v1/fine-tuning/dataset`
- `POST /api/v1/fine-tuning/job-payload`

## Notes

- Without `GROQ_API_KEY`, the chat endpoint still works but returns a scaffold response.
- Default database is local SQLite for easy bootstrapping.
- Point `DATABASE_URL` at your shared database if this repo should reuse an existing store.
- This workspace may not always have a usable host Python interpreter. The Docker test workflow above is the supported fallback for local verification in that case.
- Correlation IDs, request IDs, process-time headers, app lifespan bootstrapping, and Makefile-based checks are included to keep the repo aligned with the existing backend standard.
