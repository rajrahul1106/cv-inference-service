# CV Inference Service

A production-grade distributed computer vision inference microservice. Built to demonstrate distributed systems and ML serving patterns for portfolio purposes.

## Stack

- **API:** FastAPI 0.115+ (async, Python 3.11)
- **ORM:** SQLAlchemy 2.0 async
- **DB:** PostgreSQL 16 (with Alembic migrations)
- **Queue:** Redis 7.4 (broker for Celery)
- **Workers:** Celery 5.4
- **ML:** ultralytics (YOLOv8), mediapipe (face detection), torch (fire detection)
- **Real-time:** WebSockets (FastAPI native) + Redis pub/sub
- **Metrics:** prometheus-client
- **Frontend:** React 18 + Vite + Tailwind
- **Container:** Docker Compose v2

## Folder Structure

```
api/          - FastAPI gateway (routes, schemas, db, websocket, metrics)
workers/      - Celery workers + inference models (yolo, face, fire)
frontend/     - React dashboard
infra/        - Docker, compose files, prometheus config
scripts/      - Load test, benchmarks, model seed
tests/        - pytest suites
```

## Commands

```bash
# spin up full stack
docker compose -f infra/docker-compose.yml up -d

# spin up dev (only postgres + redis, run api/worker locally)
docker compose -f infra/docker-compose.dev.yml up -d

# run api locally
PYTHONPATH=. uvicorn api.main:app --reload --port 8000

# run celery worker locally
PYTHONPATH=. celery -A workers.celery_app worker --loglevel=info --concurrency=2

# run tests
pytest tests/ -v

# run migrations
alembic upgrade head

# create new migration
alembic revision --autogenerate -m "description"

# load test
python scripts/load_test.py --users 50 --duration 60s
```

## Conventions

- Python: PEP 8, type hints required on all function signatures
- Async functions in API layer, sync code in worker tasks (Celery is sync by design)
- Pydantic v2 for all request/response schemas
- SQLAlchemy models live in `api/db/models.py`, shared with workers via package import
- Logging: structlog with JSON output, never `print()`
- No `print()` statements in production code
- Tests: arrange-act-assert pattern, fixtures in `conftest.py`
- Commit messages: conventional commits format (`feat:`, `fix:`, `docs:`, `chore:`)

## Architecture Rules

- Models load **once per worker process** using a module-level registry. Never load inside task functions.
- WebSocket clients subscribe to Redis pub/sub channels via a background task in the API. Workers never know about WebSocket connections directly.
- Database sessions are **per-request** in API and **per-task** in workers. Never share sessions across requests.
- Results are stored as JSONB in PostgreSQL. Annotated images go to local volume (or S3 in a future iteration).
- All retryable failures use exponential backoff with jitter via Celery's `retry_backoff` and `retry_jitter`.

## Things to Never Do

- Do not load ML models inside Celery task functions (use the registry)
- Do not use `print()` for logging
- Do not commit `.env` (only `.env.example`)
- Do not bypass Alembic for schema changes
- Do not store binary data in PostgreSQL (use paths/URLs)
- Do not use synchronous DB calls in async API routes (use `async_session`)
- Do not skip tests when the build is failing

## Plan Mode First

For any task touching more than 2 files or changing architecture, use Plan Mode (Shift+Tab) before writing code. The author wants to read the plan and push back before code is written.

## Author's Goals

This project exists to demonstrate distributed systems, async patterns, ML serving, and observability. The author is preparing for software internship interviews. Code clarity and idiomatic patterns matter more than clever optimizations. When given a choice between a clever one-liner and explicit, readable code, choose readable.

## Things the Author Will Write by Hand

The author wants to write these pieces themselves to deeply understand them. Do not auto-generate these unless explicitly asked:

- The Celery base task class with retry logic
- The ConnectionManager for WebSocket tracking
- The model registry singleton pattern
- The Redis pub/sub bridge between workers and WebSockets

If asked to work on these, generate a skeleton with comments explaining the design, then let the author fill in the implementation.

## Current Phase

(Update this as you progress. Example: "Day 5 of 14, working on YOLO detector implementation. Database schema and basic job CRUD complete.")
